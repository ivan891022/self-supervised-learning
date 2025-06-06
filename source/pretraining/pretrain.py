#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimCLR pre-training with PyTorch-Lightning + lightly
* 單機多 GPU（DDP）
* 預設自動偵測可用 GPU；CPU 亦可 fallback
* **NEW**: 線性 warm-up → cosine decay
"""

import os, argparse, datetime, shutil
from pathlib import Path

import torch, torch.nn as nn, torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from lightly.data import LightlyDataset
from lightly.data.collate import SimCLRCollateFunction
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss

import numpy as np, pandas as pd
from PIL import Image

# ------------------------------------------------------------
# 1)  自訂增強
# ------------------------------------------------------------
class HistogramNormalize:
    def __init__(self, number_bins: int = 256):
        self.number_bins = number_bins

    def __call__(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image)
        if img_np.ndim == 3:
            img_np = img_np[:, :, 0]

        hist, bins = np.histogram(img_np.flatten(), self.number_bins, density=True)
        cdf = hist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        img_eq = np.interp(img_np.flatten(), bins[:-1], cdf)
        img_eq = img_eq.reshape(img_np.shape).astype(np.uint8)
        return Image.fromarray(img_eq)


class GaussianNoise:
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        mu = sample.mean()
        snr = np.random.randint(low=4, high=8)
        sigma = mu / snr
        noise = torch.normal(
            mean=0.0, std=sigma, size=sample.shape, device=sample.device
        )
        return sample + noise


# ------------------------------------------------------------
# 2)  SimCLR LightningModule  **(加入 warm-up)**
# ------------------------------------------------------------
class SimCLRModel(pl.LightningModule):
    def __init__(self, backbone, hidden_dim: int, lr: float, warmup_epochs: int):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
        self.criterion = NTXentLoss(gather_distributed=True)
        self.warmup_epochs = warmup_epochs

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def shared_step(self, batch, stage: str):
        (x0, x1), _, _ = batch
        z0, z1 = self(x0), self(x1)
        loss = self.criterion(z0, z1)
        self.log(f"{stage}_loss_ssl", loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4
        )

        total_epochs = self.trainer.max_epochs
        warm_ep = min(self.warmup_epochs, total_epochs - 1)

        # 線性 warm-up → cosine decay
        scheduler_warm = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-4, end_factor=1.0, total_iters=warm_ep
        )
        scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_epochs - warm_ep
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[scheduler_warm, scheduler_cos], milestones=[warm_ep]
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ------------------------------------------------------------
# 3)  Backbone: ResNet-50
# ------------------------------------------------------------
def get_torchvision_backbone():
    backbone = torchvision.models.resnet50(weights=None)
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    return backbone, 2048


# ------------------------------------------------------------
# 4)  只抓部分影像
# ------------------------------------------------------------
def get_subset_filenames(folder: str, ratio: float):
    img_files = sorted(
        [p.name for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    return img_files[: int(len(img_files) * ratio)]


# ------------------------------------------------------------
# 5)  預訓練流程
# ------------------------------------------------------------
def ss_pretrain(
    experiment_name: str,
    replace: bool,
    batch_size: int,
    epochs: int,
    lr: float,
    workers: int,
    labeled_dataset_percent: float,
    devices: str | int,
    warmup_epochs: int,
):
    pl.seed_everything(1)

    # a. Transform
    from torchvision import transforms as T

    base_transform = T.Compose(
        [
            HistogramNormalize(),
            T.Grayscale(num_output_channels=3),
        ]
    )

    # b. Dataset / Dataloader
    train_dir = "../data/train"
    val_dir = "../data/test"

    subset_files = get_subset_filenames(train_dir, labeled_dataset_percent)
    print(f"Pretrain subset: {len(subset_files)}/{len(os.listdir(train_dir))} images")

    ds_train = LightlyDataset(train_dir, filenames=subset_files, transform=base_transform)
    ds_val = LightlyDataset(val_dir, transform=base_transform)

    collate_train = SimCLRCollateFunction(
        input_size=512,
        gaussian_blur=0.5,
        cj_prob=0.8,
        cj_strength=0.5,
        rr_prob=1.0,
        vf_prob=0.5,
        hf_prob=0.5,
    )
    collate_val = SimCLRCollateFunction(input_size=512)

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_train,
        num_workers=workers,
        pin_memory=True,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_val,
        num_workers=workers,
        pin_memory=True,
    )

    # c. Model
    backbone, out_dim = get_torchvision_backbone()
    model = SimCLRModel(backbone, hidden_dim=out_dim, lr=lr, warmup_epochs=warmup_epochs)

    # d. I/O
    save_root = Path(f"../../vinbig_output/{experiment_name}/pretrain")
    if save_root.exists():
        if replace:
            shutil.rmtree(save_root)
        else:
            raise RuntimeError(f"{save_root} exists!  --replace 以覆寫")
    save_root.mkdir(parents=True, exist_ok=True)
    logger = TensorBoardLogger(save_root, name="tb_logs")

    # e. Trainer
    if devices == "auto":
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = devices
    accelerator = "gpu" if num_gpus and torch.cuda.is_available() else "cpu"
    strategy = (
        DDPStrategy(find_unused_parameters=False)
        if accelerator == "gpu" and num_gpus > 1
        else "auto"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=num_gpus if accelerator == "gpu" else 1,
        strategy=strategy,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
        deterministic=True,
    )

    # f. Train
    trainer.fit(model, dl_train, dl_val)

    # g. Save backbone
    torch.save(model.backbone.state_dict(), save_root / "pretrained-backbone.pth")
    print(f"Backbone checkpoint saved to {save_root/'pretrained-backbone.pth'}")


# ------------------------------------------------------------
# 6)  CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("SimCLR unsupervised pre-training (multi-GPU)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--labeled-dataset-percent", type=float, default=1.0)
    p.add_argument(
        "--experiment-name",
        type=str,
        default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    p.add_argument("--replace", action="store_true", help="覆寫同名實驗資料夾")
    p.add_argument(
        "--devices",
        default="auto",
        help="'auto'（預設，全用）或給定整數，例如 2 表示使用 2 張 GPU",
    )
    p.add_argument("--warmup-epochs", type=int, default=10, help="線性 warm-up epoch 數")
    return p.parse_args()


def main():
    args = parse_args()
    args.devices = "auto" if args.devices == "auto" else int(args.devices)
    ss_pretrain(**vars(args))


if __name__ == "__main__":
    main()
