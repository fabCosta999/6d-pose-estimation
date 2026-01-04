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

        box = self.convert_bb_yolo(sample["bbox"], W, H).unsqueeze(0)
        label = torch.tensor([sample["label"]], dtype=torch.long)


        return {
            "rgb": sample["rgb"],
            "box": box,
            "label": label,
        }



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


class YoloDetections:
    def __init__(self, yolo_model, scene_dataset, iou_thr=0.5):
        self.yolo = yolo_model
        self.scene_dataset = scene_dataset
        self.iou_thr = iou_thr

    def __call__(self, idx):
        scene = self.scene_dataset[idx]

        gt_bbox = scene["bbox"]
        gt_label = scene["label"]

        img = scene["rgb"].permute(1, 2, 0).cpu().numpy()

        results = self.yolo(
            img,
            imgsz=640,
            conf=0.25,
            verbose=False
        )

        r = results[0]
        if r.boxes is None:
            return []

        boxes = r.boxes.xyxy.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy().astype(int)

        best_iou = 0.0
        best_det = None

        for box, label in zip(boxes, labels):
            if label != gt_label:
                continue

            x1, y1, x2, y2 = box
            det_bbox = [x1, y1, x2 - x1, y2 - y1]

            val = iou(det_bbox, gt_bbox)
            if val > best_iou:
                best_iou = val
                best_det = det_bbox

        if best_iou < self.iou_thr or best_det is None:
            return None

        return {
            "bbox": best_det,
            "label": gt_label,
            "rotation": scene["rotation"],
            "translation": scene["translation"],
        }


