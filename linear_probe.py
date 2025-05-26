#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear probe for SimCLR backbone (VinBigData, 14 labels)
"""

import json, torch, torchvision, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image

# ---------- 0. 專案路徑 ----------
ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "vinbig_output/simclr_60pct/pretrain/pretrained-backbone.pth"

IMG_ROOT   = ROOT / "source/data"                 # ← 只到 data 根目錄
TRAIN_JSON = ROOT / "source/data/train_annotations.json"
TEST_JSON  = ROOT / "source/data/test_annotations.json"

for p in (CKPT, IMG_ROOT, TRAIN_JSON, TEST_JSON):
    assert p.exists(), f"❌ 路徑不存在：{p}"

# ---------- 1. 讀 COCO → multi-hot label dict ----------
def load_labels(json_path):
    with open(json_path) as f:
        coco = json.load(f)
    num_cls = len(coco["categories"])             # =14
    id2fname = {im["id"]: im["file_name"] for im in coco["images"]}
    labels = {fname: np.zeros(num_cls, np.float32) for fname in id2fname.values()}
    for ann in coco["annotations"]:
        labels[id2fname[ann["image_id"]]][ann["category_id"]] = 1.0
    return labels

train_labels = load_labels(TRAIN_JSON)
test_labels  = load_labels(TEST_JSON)

# ---------- 2. Dataset ----------
class CocoMultiLabelDataset(Dataset):
    def __init__(self, img_root: Path, label_dict: dict, tf):
        self.img_root, self.labels, self.tf = img_root, label_dict, tf
        self.fnames = sorted(label_dict.keys())

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = (self.img_root / fname).resolve()
        if not img_path.exists():
            raise FileNotFoundError(f"❌ 找不到影像檔：{img_path}")
        img = Image.open(img_path).convert("RGB")
        return self.tf(img), torch.from_numpy(self.labels[fname])

tf = T.Compose([T.Grayscale(3), T.Resize(512), T.ToTensor()])
ds_train = CocoMultiLabelDataset(IMG_ROOT, train_labels, tf)
ds_test  = CocoMultiLabelDataset(IMG_ROOT, test_labels,  tf)

# ---------- 3. 凍結 backbone ----------
backbone = torchvision.models.resnet50(weights=None)
backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
backbone.load_state_dict(torch.load(CKPT, map_location="cpu"))
backbone.eval(); backbone.requires_grad_(False)
print("✅ Backbone loaded.")

# ---------- 4. 抽特徵 ----------
def extract(dataset):
    dl = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)
    feats, labs = [], []
    with torch.inference_mode():
        for x, y in dl:
            feats.append(backbone(x).flatten(1).numpy())
            labs.append(y.numpy())
    return np.vstack(feats), np.vstack(labs)

print("⧗ Extracting train features …")
X_train, y_train = extract(ds_train)
print("⧗ Extracting test  features …")
X_test,  y_test  = extract(ds_test)

# ---------- 5. 線性分類 ----------
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=8, solver="lbfgs"))
clf.fit(X_train, y_train)

# ---------- 6. 評估 ----------
y_score = clf.predict_proba(X_test)                  # [N, 14]
auroc = roc_auc_score(y_test, y_score, average="macro")
print(f"\n🎯 Linear-probe macro-AUROC: {auroc:.4f}")
