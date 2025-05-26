# Self-supervised Chest X-ray Detection with SimCLR + Faster R-CNN

> 本專案以 **SimCLR** 自監督預訓練 ResNet-50 backbone  
> 再搭配 **Faster R-CNN** 進行胸腔 X 光多類別病灶偵測  
> 透過 **Repeat-Factor oversampling** 與 **Albumentations online augmentation**  
> 大幅減少標註需求並提升 mAP

---

## Table of Contents
- [Motivation](#motivation)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Environment Setup](#environment-setup)
- [Pre-training - SimCLR](#pre-training---simclr)
- [Linear Probe Evaluation](#linear-probe-evaluation)
- [Fine-tuning - Faster R-CNN](#fine-tuning---faster-r-cnn)
- [Evaluation](#evaluation)
- [Results](#results)




---

## Motivation
- 原論文：**“Self-supervised Learning as a Means to Reduce the Need for Labeled Data in Medical Image Analysis”**  
- 改動重點  
  1. 將原先 MMDetection pipeline **全面改寫為 TorchVision**，降低相依性  
  2. 以 **VinBigData chest X-ray (512 × 512)** 作為資料來源；60 % 影像不含標註，供 SimCLR 自監督預訓練  
  3. 在 fine-tune 階段加入 **Albumentations online augmentation + Repeat-Factor oversampling**，解決類別不均  
  4. 全流程支援 **單機多 GPU（PyTorch DDP / torchrun）**  

---

## Project Structure
```
SELF_SUPERVISED_PRETRAINING/                 <!-- 專案根目錄 -->
├─ source/                                   <!-- 主要原始碼 -->
│  ├─ data/                                  <!-- 轉成 COCO 的影像與標註 -->
│  │  ├─ train/        # 80 % 訓練影像 
│  │  ├─ test/         # 20 % 測試影像  
│  │  ├─ train_annotations.json  
│  │  └─ test_annotations.json  
│  ├─ detection/        # 推論 ‧ 視覺化工具（暫空位示意）  
│  ├─ pretraining/      
│  │  └─ pretrain.py    # SimCLR Lightning Module  
│  ├─ convert_to_coco.py # 資料清理 + WBF + COCO 轉換  
│  ├─ train.py           # Faster R-CNN fine-tune (DDP)  
│  └─ test.py            # mAP 評估 & 可視化        
├─ environment.yaml      # conda / mamba 環境描述  
├─ linear_probe.py       # 線性 Probe 評估 simclr 效能  
└─ readme.md               

```
---

## Dataset Preparation
1. **下載影像**  
   - Kaggle: <https://www.kaggle.com/awsaf49/vinbigdata-512-image-dataset>  
2. **資料清理與 COCO 轉換** (`convert_to_coco.py`)  
   - 移除 `No finding` 類別影像  
   - 以 **Weighted Box Fusion (WBF)** 融合多位醫師的重疊框  
   - 依 80 / 20 split 產出  
     ```
     data/train/*.png
     data/test/*.png
     data/train_annotations.json
     data/test_annotations.json
     ```
3. **類別定義**（共 14 類，不含背景）
```
Aortic_enlargement, Atelectasis, Calcification, Cardiomegaly,
Consolidation, ILD, Infiltration, Lung_Opacity, Nodule/Mass,
Other_lesion, Pleural_effusion, Pleural_thickening,
Pneumothorax, Pulmonary_fibrosis
```
---

## Environment Setup

先下載我的environment.yaml
```
conda env create -f environment.yaml -n my_env_name
conda activate my_env_name
```
---

## Pre-training - SimCLR
```
cd source/pretraining
(ex)指令輸入:
python pretrain.py --experiment-name simclr_60pct --batch-size 8 --epochs 100 --lr 0.05 --workers 8 --labeled-dataset-percent 0.6 --warmup-epochs 10 --devices auto --replace//把train的百分之60資料做自監督學習
Backbone：ResNet-50 (去除 fc)
Augmentation：直方圖均衡, Gaussian Noise, Color Jitter, RandRotation, HF/VF, Gaussian Blur
訓練結束自動輸出:
vinbig_output/simclr_60pct/pretrain/pretrained-backbone.pth
```
---

## Linear Probe Evaluation
```
cd self_supervised_pretraining (也就是最初的目錄)
(ex)指令輸入:
python linear_probe.py
使用凍結的 SimCLR backbone 抽取影像特徵，
再以 LogisticRegression (One-vs-Rest) 訓練輕量級線性分類器，
透過 macro-AUROC 作為無監督預訓練品質的 proxy
```
---

## Fine-tuning - Faster R-CNN
```
cd self_supervised_pretraining (也就是最初的目錄)
(ex)指令輸入:
torchrun --nproc_per_node=8 train.py --experiment-name simclr_60pct --labeled-dataset-percent 0.4 --epochs 100 --batch-size 8 --lr 0.01
載入 SimCLR backbone
Repeat-Factor oversampling
Online Augmentation (Albumentations)：ShiftScaleRotate, Flip, Brightness/Contrast, Gamma, Noise, MotionBlur
最終模型輸出至:
vinbig_output/simclr_60pct/fasterrcnn_final.pth
```
---

## Evaluation
```
cd source
(ex)指令輸入:
python test.py --mode map --checkpoint fasterrcnn_final.pth --experiment-name simclr_60pct --test-data-dir data/test --test-ann-file data/test_annotations.json --device cuda:0//計算map
```
---

## Results

適當的使用online oversampling和online data augmentation能夠提升模型的泛化能力

