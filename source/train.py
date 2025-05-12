#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================
# Multi-GPU (DDP) fine-tuning for VinBigData Faster R-CNN
#   * Repeat-Factor oversampling
#   * Albumentations online aug
#   * 能載入 SimCLR 預訓練 backbone (.pth)
# ===========================================================

import os, json, math, random, argparse
import numpy as np, cv2, albumentations as A
import torch, torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision
from torchvision import transforms as tvT

# reproducibility
SEED = 1
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ───────────────────── 1.  Augmentations ────────────────────
def get_train_augs():
    return A.Compose(
        [
            A.ShiftScaleRotate(0.0625, 0.10, 10,
                               border_mode=cv2.BORDER_CONSTANT, value=0, p=0.4),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.25),
            A.RandomGamma(gamma_limit=(85, 115), p=0.25),
            A.GaussNoise(var_limit=(5, 15), p=0.2),
            A.MotionBlur(blur_limit=3, p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=16,
            min_visibility=0.3)
    )

def get_val_augs():     # fine-tune階段，valid/test 不做增強
    return None

# ───────────────────── 2.  Dataset  ─────────────────────────
class MyCocoDataset(Dataset):
    """COCO 格式，bbox=xyxy (pascal_voc)"""
    def __init__(self, img_dir, ann_file, transforms=None,
                 subset_percent=1.0, is_train=True):
        self.img_dir, self.transforms, self.is_train = img_dir, transforms, is_train
        with open(ann_file) as f: coco = json.load(f)

        images = coco["images"]; anns = coco["annotations"]
        cats   = {c["id"]: c["name"] for c in coco["categories"]}

        # img_id → anns
        img2anns = {}
        for a in anns:
            img2anns.setdefault(a["image_id"], []).append(a)

        images = [im for im in images if im["id"] in img2anns]
        n_use  = int(len(images) * subset_percent)
        self.images   = images[:n_use]
        self.img2anns = img2anns
        self.cats     = cats
        print(f"[INFO] use {n_use}/{len(images)} images ({subset_percent*100:.1f} %)")

        self.classes = [
            "Aortic_enlargement","Atelectasis","Calcification","Cardiomegaly",
            "Consolidation","ILD","Infiltration","Lung_Opacity","Nodule/Mass",
            "Other_lesion","Pleural_effusion","Pleural_thickening",
            "Pneumothorax","Pulmonary_fibrosis"
        ]
        self.cls2idx = {c: i+1 for i,c in enumerate(self.classes)}
        self.toTensor = tvT.ToTensor()

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img_path = os.path.join(self.img_dir, os.path.basename(info["file_name"]))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        boxes, labels = [], []
        for a in self.img2anns[info["id"]]:
            x,y,w,h = a["bbox"]
            boxes.append([x, y, x+w, y+h])
            labels.append(self.cls2idx[self.cats[a["category_id"]]])

        if self.is_train and self.transforms:
            for _ in range(10):
                aug = self.transforms(image=img, bboxes=boxes, labels=labels)
                if len(aug["bboxes"])>0:
                    img, boxes, labels = aug["image"], aug["bboxes"], aug["labels"]; break

        return self.toTensor(img), {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

# ───────────────────── 3.  Repeat-Factor ────────────────────
class RepeatFactorDataset(Dataset):
    def __init__(self, base_ds, tau=0.4):
        self.base_ds = base_ds
        total = len(base_ds)

        from collections import defaultdict
        cnt = defaultdict(int)
        for _, tgt in base_ds:
            for c in set(tgt["labels"].tolist()):
                cnt[c] += 1

        rf_cls = {c: math.sqrt(tau/(n/total)) if (n/total)<tau else 1.0
                  for c,n in cnt.items()}

        self.rep = []
        for idx in range(total):
            _, tgt = base_ds[idx]
            rf_i = max(rf_cls[c.item()] for c in tgt["labels"].unique())
            self.rep.extend([idx]*math.ceil(rf_i))

        print(f"[RepeatFactor] {total} → {len(self.rep)} (×{len(self.rep)/total:.2f})")

    def __len__(self): return len(self.rep)
    def __getitem__(self,i): return self.base_ds[self.rep[i]]

# ───────────────────── 4.  Model  ───────────────────────────
def build_model(num_classes=15):
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None, num_classes=num_classes)

def load_backbone(model, path, rank=0):
    if not os.path.isfile(path):
        if rank==0: print("[WARN] backbone not found:", path)
        return
    sd = torch.load(path, map_location="cpu")
    target = model.backbone.body.state_dict()
    model.backbone.body.load_state_dict({k: sd[k] for k in target if k in sd},
                                        strict=False)
    if rank==0: print("[INFO] SimCLR backbone loaded")

# ───────────────────── 5.  DDP helpers  ─────────────────────
def reduce_loss(loss):
    with torch.no_grad():
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= dist.get_world_size()
    return loss

def collate_fn(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)

# ───────────────────── 6.  Train loop  ──────────────────────
def train_loop(rank, world_size, args):
    # ─ 6-1. DDP 初始化
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # ─ 6-2. Dataset / Sampler / DataLoader
    base_train = MyCocoDataset("data/train", "data/train_annotations.json",
                               get_train_augs(), args.labeled_dataset_percent, True)
    train_ds   = RepeatFactorDataset(base_train, tau=0.4)
    sampler    = DistributedSampler(train_ds, shuffle=True)
    train_dl   = DataLoader(train_ds, batch_size=args.batch_size,
                            sampler=sampler, drop_last=True,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # ─ 6-3. Model & Optim
    model = build_model().to(device)
    load_backbone(model,
        f"../vinbig_output/{args.experiment_name}/pretrain/pretrained-backbone.pth",
        rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=1e-4)

    # ─ 6-4. Epoch loop
    for ep in range(1, args.epochs+1):
        sampler.set_epoch(ep)
        model.train(); running = 0.0
        for imgs, tgts in train_dl:
            imgs = [i.to(device) for i in imgs]
            tgts = [{k: v.to(device) for k,v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())
            optim.zero_grad(); loss.backward(); optim.step()

            loss_detached = reduce_loss(loss.detach())
            running += loss_detached.item()

        if rank==0:
            print(f"[Epoch {ep}/{args.epochs}] "
                  f"mean_loss={running/len(train_dl):.4f}")

    # ─ 6-5. 儲存
    if rank==0:
        out_dir = f"../vinbig_output/{args.experiment_name}"
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.module.state_dict(), f"{out_dir}/fasterrcnn_final.pth")
        print("[INFO] model saved to", f"{out_dir}/fasterrcnn_final.pth")

    dist.destroy_process_group()

# ───────────────────── 7.  CLI  ─────────────────────────────
def parse():
    p = argparse.ArgumentParser("DDP fine-tune")
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--labeled-dataset-percent", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)   # per-GPU
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.005)
    return p.parse_args()

# ───────────────────── 8.  Entry  ──────────────────────────
if __name__ == "__main__":
    args = parse()

    # world size 由 torchrun 自動設定
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 直接在 main process 開多 subprocess（已由 torchrun 處理）
    # 如果用 Python 單檔測試 (world_size==1) 也能跑 .
    train_loop(local_rank, world_size, args)
