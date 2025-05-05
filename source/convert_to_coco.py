# based on https://www.kaggle.com/sreevishnudamodaran/vinbigdata-fusing-bboxes-coco-dataset#Building-COCO-DATASET
# ------------------------------------------------------------
# 產生 80 % train / 20 % test 之 COCO 格式 + WBF 融合
# ------------------------------------------------------------

import os
from pathlib import Path
from datetime import datetime
import shutil
from collections import Counter
import warnings
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- 視覺化輔助 ----------
def plot_img(img, size=(18, 18), cmap='gray', title=""):
    plt.figure(figsize=size)
    plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()

def plot_imgs(imgs, cols=2, size=10, cmap='gray', img_size=None):
    rows = len(imgs) // cols + 1
    fig = plt.figure(figsize=(cols * size, rows * size))
    for i, img in enumerate(imgs):
        if img_size is not None:
            img = cv.resize(img, img_size)
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle("")

def draw_bbox(image, box, label, color, thickness=3):
    alpha, alpha_box = 0.1, 0.4
    overlay_bbox, overlay_text, output = image.copy(), image.copy(), image.copy()

    text_width, text_height = cv.getTextSize(label.upper(),
                                             cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv.rectangle(overlay_bbox, (box[0], box[1]), (box[2], box[3]), color, -1)
    cv.addWeighted(overlay_bbox, alpha, output, 1 - alpha, 0, output)
    cv.rectangle(overlay_text,
                 (box[0], box[1] - 7 - text_height),
                 (box[0] + text_width + 2, box[1]),
                 (0, 0, 0), -1)
    cv.addWeighted(overlay_text, alpha_box, output, 1 - alpha_box, 0, output)
    cv.rectangle(output, (box[0], box[1]), (box[2], box[3]), color, thickness)
    cv.putText(output, label.upper(), (box[0], box[1] - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return output

def normalize_bboxes(df):
    df['x_min'] = df.x_min / df.width * 512
    df['y_min'] = df.y_min / df.height * 512
    df['x_max'] = df.x_max / df.width * 512
    df['y_max'] = df.y_max / df.height * 512
    df['w'] = df.x_max - df.x_min
    df['h'] = df.y_max - df.y_min
    df['area'] = df.w * df.h
    return df

# ---------- 標籤 ----------
labels = [
    "__ignore__", "Aortic_enlargement", "Atelectasis", "Calcification",
    "Cardiomegaly", "Consolidation", "ILD", "Infiltration", "Lung_Opacity",
    "Nodule/Mass", "Other_lesion", "Pleural_effusion", "Pleural_thickening",
    "Pneumothorax", "Pulmonary_fibrosis"
]
viz_labels = labels[1:]
label2color = [
    [59, 238, 119], [222, 21, 229], [94, 49, 164], [206, 221, 133],
    [117, 75, 3], [210, 224, 119], [211, 176, 166], [63, 7, 197],
    [102, 65, 77], [194, 134, 175], [209, 219, 50], [255, 44, 47],
    [89, 125, 149], [110, 27, 100]
]

# ---------- COCO 樣板 ----------
def make_coco_template():
    now = datetime.now()
    tmpl = dict(
        info=dict(
            description=None, url=None, version=None,
            year=now.year, contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(url=None, id=0, name=None)],
        images=[],
        annotations=[],
        categories=[],
        type='instances'
    )
    for i, name in enumerate(labels):
        if i == 0:   # __ignore__
            continue
        tmpl['categories'].append(dict(id=i - 1, name=name, supercategory=None))
    return tmpl

data_train = make_coco_template()
data_test  = make_coco_template()

# ---------- 輸出目錄 ----------
train_out_dir = 'data/train'
test_out_dir  = 'data/test'
for d in [train_out_dir, test_out_dir]:
    if Path(d).exists():
        shutil.rmtree(d)
    os.makedirs(d)

train_out_file = 'data/train_annotations.json'
test_out_file  = 'data/test_annotations.json'

# ---------- 讀取標註並正規化 ----------
all_images_folder = '../vinbigdata/train'
ann_csv           = '../vinbigdata/train.csv'

all_annotations = pd.read_csv(ann_csv)
all_annotations = all_annotations[all_annotations.class_id != 14]         # 去掉 no-finding
all_annotations['image_path'] = all_annotations['image_id'].map(
    lambda x: os.path.join(all_images_folder, f'{x}.png'))
all_annotations = normalize_bboxes(all_annotations)

all_image_paths = all_annotations['image_path'].unique()

# ---------- 切分 8 / 2 ----------
np.random.seed(1)
perm   = np.random.permutation(len(all_image_paths))
split  = int(0.8 * len(perm))        # 80 %
train_paths = all_image_paths[perm[:split]]
test_paths  = all_image_paths[perm[split:]]

print(f'train: {len(train_paths)}, test: {len(test_paths)}')

folders     = [train_out_dir,           test_out_dir]
paths_list  = [train_paths,             test_paths]
data_dicts  = [data_train,              data_test]
out_files   = [train_out_file,          test_out_file]

# ---------- WBF 參數 ----------
iou_thr, skip_box_thr = 0.2, 0.0001

# ------------------------------------------------------------
# 主迴圈：複製圖片、做 WBF、寫 COCO json
# ------------------------------------------------------------
for folder, paths, coco, out_file in zip(folders, paths_list, data_dicts, out_files):
    print(f'Saving to {folder} ...')
    viz_images = []

    for img_id, img_path in tqdm(list(enumerate(paths))):
        img = cv.imread(img_path)
        stem = Path(img_path).stem
        shutil.copy2(img_path, folder)  # 複製圖片到輸出資料夾

        # -- COCO image entry
        coco['images'].append(dict(
            id=img_id,
            license=0,
            file_name=os.path.join(Path(folder).name, f'{stem}.png'),
            height=img.shape[0],
            width=img.shape[1],
            date_captured=None,
            url=None
        ))

        ann_this = all_annotations[all_annotations.image_id == stem]
        boxes_viz   = ann_this[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
        labels_viz  = ann_this['class_id'].to_numpy()

        # -- 視覺化 (每 500 張)
        if img_id % 500 == 0:
            img_before = img.copy()
            for box, lab in zip(boxes_viz, labels_viz):
                img_before = draw_bbox(img_before, box.astype(int).tolist(),
                                       viz_labels[lab], label2color[lab])
            viz_images.append(img_before)

        # -- 組合同 label 的 box 以便 WBF
        boxes_list, scores_list, labels_list, weights = [], [], [], []
        boxes_single, labels_single = [], []
        cnt = Counter(ann_this['class_id'])

        for cid in cnt:
            cls_boxes = ann_this[ann_this.class_id == cid][
                ['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
            if cnt[cid] == 1:
                labels_single.append(cid)
                boxes_single.append(cls_boxes.squeeze().tolist())
            else:
                # WBF 前需做 0~1 正規化
                norm = cls_boxes / (img.shape[1], img.shape[0],
                                    img.shape[1], img.shape[0])
                boxes_list.append(norm.tolist())
                scores_list.append(np.ones(len(norm)).tolist())
                labels_list.append([cid] * len(norm))
                weights.append(1)

        # -- 執行 WBF
        boxes, scores, box_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

        # -- 轉回絕對座標 & 加入單框
        boxes = (boxes * (img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0])).round(1).tolist()
        box_labels = box_labels.astype(int).tolist()
        boxes.extend(boxes_single)
        box_labels.extend(labels_single)

        # -- 加入 COCO annotation
        for b, lab in zip(boxes, box_labels):
            x_min, y_min, x_max, y_max = b
            w, h = x_max - x_min, y_max - y_min
            coco['annotations'].append(dict(
                id=len(coco['annotations']),
                image_id=img_id,
                category_id=int(lab),
                bbox=[round(x_min, 1), round(y_min, 1),
                      round(w, 1), round(h, 1)],
                area=round(w * h, 1),
                iscrowd=0
            ))

        # -- 視覺化 (每 500 張)
        if img_id % 500 == 0:
            img_after = img.copy()
            for b, lab in zip(boxes, box_labels):
                img_after = draw_bbox(img_after, list(map(int, b)),
                                      viz_labels[lab], label2color[lab])
            viz_images.append(img_after)

    # -- 顯示對照圖
    if viz_images:
        plot_imgs(viz_images, cmap=None, size=40)
        plt.figtext(0.28, 0.9, "Original", size=15)
        plt.figtext(0.71, 0.9, "WBF",      size=15)
        plt.show()

    # -- 輸出 json
    with open(out_file, 'w') as f:
        json.dump(coco, f, indent=4)

print("Done.")
