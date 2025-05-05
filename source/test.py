import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import cv2 as cv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# 如果有 references/detection/coco_eval.py / coco_utils.py，需要加路徑
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
REFERENCES_DIR = os.path.abspath(os.path.join(FILE_DIR, '..', 'references'))
if REFERENCES_DIR not in sys.path:
    sys.path.append(REFERENCES_DIR)

from detection import utils
from detection.coco_eval import CocoEvaluator
from detection.coco_utils import get_coco_api_from_dataset

from pycocotools.coco import COCO

###############################################################################
# 1) 測試資料集 (COCO Format)
###############################################################################
class MyCocoTestDataset(Dataset):
    """
    - 讀取 convert_to_coco 後的 test_annotations.json，其 category_id=0..13 (14疾病)
    - 但模型訓練時使用了 15 類 (0=背景, 1..14=疾病)。
      => 所以 這裡要把 annotation 中的 category_id (0..13) +1 => label=1..14。

    - boxes: [x, y, w, h] (COCO 格式)
      TorchVision 在 forward 時會自動轉 xyxy，用於推論/計算 loss。
    """
    def __init__(self, image_dir, ann_file, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.ann_file = ann_file
        self.transform = transform

        # 讀取 COCO JSON
        with open(self.ann_file, 'r') as f:
            self.coco = json.load(f)

        self.images = self.coco['images']        # list of {id, file_name, ...}
        self.annotations = self.coco['annotations']  # list of {id, image_id, category_id, bbox, ...}

        # 建立 img_id => [ann, ...]
        valid_image_ids = set(img['id'] for img in self.images)
        self.img_id_to_anns = {}
        for ann in self.annotations:
            cat_id = ann['category_id']
            # 預期 cat_id 在 0..13
            if (cat_id < 0) or (cat_id > 13):
                continue

            img_id = ann['image_id']
            if img_id not in valid_image_ids:
                print(f"[Warning] annotation with image_id={img_id} not in images[]")
                continue

            self.img_id_to_anns.setdefault(img_id, []).append(ann)

        # 14 個疾病 (id=0..13)，順序要與 convert_to_coco 裡 categories 一致
        # 但因為你訓練的模型是 (0=背景, 1..14=疾病)，
        # 所以對應 classes[i] => i=0..13 =>  "Aortic_enlargement" ...
        # 只是拿來可視化/對應 label-1
        self.classes = [
            "Aortic_enlargement",   # 0
            "Atelectasis",          # 1
            "Calcification",        # 2
            "Cardiomegaly",         # 3
            "Consolidation",        # 4
            "ILD",                  # 5
            "Infiltration",         # 6
            "Lung_Opacity",         # 7
            "Nodule/Mass",          # 8
            "Other_lesion",         # 9
            "Pleural_effusion",     # 10
            "Pleural_thickening",   # 11
            "Pneumothorax",         # 12
            "Pulmonary_fibrosis"    # 13
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id   = img_info['id']
        file_n   = os.path.basename(img_info['file_name'])  # e.g. test/xxx.png
        path     = os.path.join(self.image_dir, file_n)

        # 讀圖
        img = Image.open(path).convert('RGB')

        ann_list = self.img_id_to_anns.get(img_id, [])

        boxes = []
        labels= []
        area  = []
        iscrowd = []

        for ann in ann_list:
            x, y, w, h = ann['bbox']  # COCO => [x, y, w, h]
            boxes.append([x, y, w, h])
            cat_id = ann['category_id']  # 0..13
            # shift => label = cat_id + 1 (1..14)
            label  = (cat_id + 1)
            labels.append(label)
            area.append(w*h)
            iscrowd.append(0)

        # 組成 target
        target = {
            "boxes":    torch.as_tensor(boxes,  dtype=torch.float32).reshape(-1,4),
            "labels":   torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor([img_id]),
            "area":     torch.as_tensor(area,   dtype=torch.float32),
            "iscrowd":  torch.as_tensor(iscrowd,dtype=torch.int64),
        }

        if self.transform:
            img = self.transform(img)
        return img, target


###############################################################################
# 2) 載入 Faster R-CNN 權重
###############################################################################
def load_fasterrcnn_model(checkpoint_path, num_classes=15, device='cuda'):
    """
    - 如果你訓練時是 (0=背景 + 14個疾病 => 15) ，這裡就用 15。
    - 嚴格對應: final classification layer => out_channels = num_classes
    """
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    sd = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


###############################################################################
# 3) 繪圖: draw_predictions_and_gt
###############################################################################
def draw_predictions_and_gt(
    img_cv,
    pred_boxes, pred_labels, pred_scores,
    gt_boxes_xyxy=None, gt_labels=None,
    classes=None,
    score_thr=0.3
):
    """
    - 模型輸出 pred_labels=1..14 => 需要 -1 才能對應 classes[0..13]
    - GT 也是 1..14 => 同樣 -1
    """
    # 畫預測
    for i in range(len(pred_boxes)):
        score = pred_scores[i]
        if score < score_thr:
            continue
        x1, y1, x2, y2 = pred_boxes[i].astype(int)
        lbl = pred_labels[i]  # 1..14
        lbl_idx = lbl - 1     # => 0..13

        if classes and (0 <= lbl_idx < len(classes)):
            label_str = f"{classes[lbl_idx]}:{score:.2f}"
        else:
            label_str = f"{lbl}:{score:.2f}"

        cv.rectangle(img_cv, (x1,y1),(x2,y2),(0,255,0),2)   # Green
        cv.putText(img_cv, label_str, (x1,y1-5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # 畫 GT
    if gt_boxes_xyxy is not None and gt_labels is not None:
        for j in range(len(gt_boxes_xyxy)):
            x1,y1,x2,y2 = gt_boxes_xyxy[j].astype(int)
            glb = gt_labels[j]   # 1..14
            glb_idx = glb - 1    # => 0..13
            if classes and (0 <= glb_idx < len(classes)):
                glabel = classes[glb_idx]
            else:
                glabel = f"G{glb}"
            cv.rectangle(img_cv, (x1,y1),(x2,y2),(255,0,0),1) # Blue
            cv.putText(img_cv, glabel, (x1,y1-5),
                       cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)

    return img_cv


###############################################################################
# 3-bis) visualize: 走訪 dataset, 每張圖跑推論, 畫結果
###############################################################################
def visualize(model, dataset, device, classes,
              out_dir='vis_results', score_thr=0.3):
    os.makedirs(out_dir, exist_ok=True)
    from torchvision import transforms as T

    for i in range(len(dataset)):
        img_tensor, tgt = dataset[i]  # (C,H,W), {boxes, labels(1..14), ...}
        img_gpu = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(img_gpu)[0]  # => {'boxes','labels','scores'}

        p_boxes  = preds['boxes'].cpu().numpy()
        p_labels = preds['labels'].cpu().numpy()   # 1..14
        p_scores = preds['scores'].cpu().numpy()

        # GT 是 xywh => 轉 xyxy
        g_boxes_wh = tgt['boxes'].numpy()  # shape [N,4], xywh
        g_boxes_xyxy = g_boxes_wh.copy()
        g_boxes_xyxy[:,2] = g_boxes_xyxy[:,0] + g_boxes_xyxy[:,2]
        g_boxes_xyxy[:,3] = g_boxes_xyxy[:,1] + g_boxes_xyxy[:,3]
        g_labels = tgt['labels'].numpy()   # 1..14

        # to cv2 BGR
        img_cv = np.array(T.ToPILImage()(img_tensor))[:,:,::-1].copy()

        drawn = draw_predictions_and_gt(
            img_cv,
            p_boxes, p_labels, p_scores,
            g_boxes_xyxy, g_labels,
            classes=classes,
            score_thr=score_thr
        )
        out_path = os.path.join(out_dir, f"{i:04d}.png")
        cv.imwrite(out_path, drawn)
        print(f"[{i+1}/{len(dataset)}] => {out_path}")


###############################################################################
# 4) dataset-check: label hist
###############################################################################
def dataset_check(dataset, classes):
    """
    - 統計 test dataset 中的 label 分布, 這裡 label=1..14 => (label-1)=0..13
    """
    from collections import Counter
    cc = Counter()
    for i in range(len(dataset)):
        _, tgt = dataset[i]
        labs = tgt['labels'].tolist()  # 1..14
        cc.update(labs)

    x_idx = np.arange(len(classes))  # 0..13
    freq = [cc[x+1] for x in x_idx]  # x+1 => label=1..14
    plt.bar(x_idx, freq)
    plt.xticks(x_idx, classes, rotation=40)
    plt.tight_layout()
    plt.title("Test dataset label distribution (label=1..14 => minus1 => classes[0..13])")
    plt.show()


###############################################################################
# 5) evaluate_map: 直接用 CocoEvaluator
###############################################################################
def evaluate_map(model, dataset, device, iou_type='bbox'):
    """
    - 要注意: dataset.ann_file 內 category_id=0..13
    - 但我們餵給 model 時 label=1..14
    - CocoEvaluator 預設會把 annotation 的 category_id=0..13 帶入
      => evaluate 時, 需要把 model 輸出 label=1..14 => 轉回 0..13
        (否則 coco_eval 會對不上)
    """
    dl = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda b: list(zip(*b)))

    coco_gt = COCO(dataset.ann_file)
    evaluator = CocoEvaluator(coco_gt, [iou_type])
    valid_ids = set(coco_gt.getImgIds())

    model.eval()
    with torch.no_grad():
        for imgs, tgts in dl:
            imgs = [img.to(device) for img in imgs]
            final_tgts = []
            for t in tgts:
                d = {k:v.to(device) if hasattr(v,'to') else v for k,v in t.items()}
                final_tgts.append(d)

            outs = model(imgs)  # list of dict
            formatted_outputs = {}
            for out, tgt in zip(outs, final_tgts):
                out_cpu = {k: v.cpu() for k,v in out.items()}
                # out_cpu['boxes'], out_cpu['labels']=1..14, out_cpu['scores']
                img_id_tensor = tgt["image_id"]
                img_id = int(img_id_tensor.item()) if img_id_tensor.numel()==1 else int(img_id_tensor[0].item())

                if img_id not in valid_ids:
                    print(f"[Warning] skip image_id {img_id} not in GT")
                    continue

                # 把輸出 label=1..14 => category_id=0..13
                out_cpu["labels"] = out_cpu["labels"] - 1  # => 0..13

                out_cpu["image_id"] = img_id
                formatted_outputs[img_id] = out_cpu

            if formatted_outputs:
                evaluator.update(formatted_outputs)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()


###############################################################################
# main
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        "TorchVision testing code (map/visualize/dataset-check)"
    )
    parser.add_argument('--mode', choices=['map','visualize','dataset-check'], default='map',
                        help="map=評估mAP, visualize=畫框, dataset-check=統計每類標註數量")
    parser.add_argument('--checkpoint', default='fasterrcnn_final.pth',
                        help="你的 FasterRCNN 權重檔名 (.pth)")
    parser.add_argument('--experiment-name', default='simclr_60pct',
                        help="對應 ../vinbig_output/{experiment-name}/{checkpoint}")
    parser.add_argument('--test-data-dir', default='../data/test',
                        help="測試影像資料夾 (convert_to_coco 產生)")
    parser.add_argument('--test-ann-file', default='../data/test_annotations.json',
                        help="測試的 COCO JSON")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help="視覺化時的預測分數閾值")
    parser.add_argument('--output-dir', default='vis_results')
    return parser.parse_args()


def main():
    args = parse_args()

    # 假設你訓練時用了 15 類 (0=BG, 1..14=疾病)
    # => 這裡也要 num_classes=15
    ckpt_path = os.path.join('../vinbig_output', args.experiment_name, args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] {ckpt_path} not found")
        return

    model = load_fasterrcnn_model(ckpt_path, num_classes=15, device=args.device)

    # 建立 test dataset (將 annotation['category_id']=0..13 => label=1..14)
    tfm = transforms.ToTensor()
    dataset = MyCocoTestDataset(args.test_data_dir, args.test_ann_file, transform=tfm)

    # 可查看有多少張圖
    coco_gt = COCO(args.test_ann_file)
    print(f"[INFO] test_annotations => total images = {len(coco_gt.getImgIds())}")

    # 供可視化 / label對應
    classes = [
        "Aortic_enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
        "Consolidation", "ILD", "Infiltration", "Lung_Opacity",
        "Nodule/Mass", "Other_lesion", "Pleural_effusion", "Pleural_thickening",
        "Pneumothorax", "Pulmonary_fibrosis"
    ]

    if args.mode == 'map':
        evaluate_map(model, dataset, args.device, iou_type='bbox')

    elif args.mode == 'visualize':
        visualize(model, dataset, args.device,
                  classes=classes,
                  out_dir=args.output_dir,
                  score_thr=args.score_thr)

    elif args.mode == 'dataset-check':
        dataset_check(dataset, classes)

if __name__=="__main__":
    main()