import torch
from torch.utils.data import Dataset


class YoloDataset(Dataset):
    def __init__(self, scene_dataset):
        self.scene_dataset = scene_dataset

    def __len__(self):
        return len(self.scene_dataset)
    
    def clamp_to_01(self, value):
        if value < 0:
            return 0
        if value > 1:
            return 1
        return value
    

    def convert_bb_yolo(self, bb, W, H):
        x, y, w, h = bb
        xc = x + w / 2
        yc = y + h / 2
        xc /= W
        yc /= H
        w  /= W
        h  /= H
        xc = self.clamp_to_01(xc)
        yc = self.clamp_to_01(yc)
        w = self.clamp_to_01(w)
        h = self.clamp_to_01(h)
        return torch.tensor([xc, yc, w, h], dtype=torch.float32)

    def __getitem__(self, idx):
        sample = self.scene_dataset[idx]
        W, H = sample["size"]

        boxes = []
        labels = []

        for obj in sample["objects"]:
            x, y, w, h = obj["bbox"]
            boxes.append(self.convert_bb_yolo([x, y, w, h], W, H))
            labels.append(obj["label"])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.stack(boxes)
            labels = torch.tensor(labels, dtype=torch.long)

        return {
            "rgb": sample["rgb"],
            "boxes": boxes,
            "labels": labels,
        }


def yolo_to_pixel(bb, W, H):
    xc, yc, w, h = bb
    w *= W
    h *= H
    x = xc * W - w / 2
    y = yc * H - h / 2
    return [x, y, w, h]


def iou(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0

def match_bbox_to_gt(det, gt_objects, iou_thr):
    best_iou = 0.0
    best_gt = None

    for gt in gt_objects:
        if det["label"] != gt["label"]:
            continue

        val = iou(det["bbox"], gt["bbox"])
        if val > best_iou:
            best_iou = val
            best_gt = gt

    if best_iou >= iou_thr:
        return best_gt
    return None



class YoloDetections:
    def __init__(self, yolo_model, scene_dataset, iou_thr=0.5):
        self.yolo = yolo_model
        self.scene_dataset = scene_dataset
        self.iou_thr = iou_thr

    def __call__(self, idx):
        scene = self.scene_dataset[idx]
        gt_objects = scene["objects"]

        img = scene["rgb"].permute(1, 2, 0).cpu().numpy()

        results = self.yolo(
            img,
            imgsz=640,
            conf=0.25,
            verbose=False
        )

        detections = []

        r = results[0]
        if r.boxes is None:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy().astype(int)

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            det_bbox = [x1, y1, x2 - x1, y2 - y1]

            det = {
                "bbox": det_bbox,
                "label": label,
            }

            gt = match_bbox_to_gt(det, gt_objects, self.iou_thr)
            if gt is None:
                continue

            detections.append({
                "bbox": det_bbox,
                "label": label,
                "rotation": gt["rotation"],
                "translation": gt["translation"],  
            })

        return detections

