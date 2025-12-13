import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml
import torchvision.transforms as transforms
from PIL import Image

class LinemodDataset(Dataset):
    CLASSES = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    OBJ_ID_TO_CLASS = {obj_id: i for i, obj_id in enumerate(CLASSES)}

    def __init__(self, dataset_root, split="train"):
        self.dataset_root = Path(dataset_root)
        self.split = split

        self.samples = []
        self.gt_data = {}

        for obj_id in self.CLASSES:
            obj_dir = self.dataset_root / "data" / f"{obj_id:02d}"

            split_file = obj_dir / f"{split}.txt"
            with open(split_file) as f:
                for line in f:
                    self.samples.append((obj_id, int(line.strip())))

            with open(obj_dir / "gt.yml") as f:
                self.gt_data[obj_id] = yaml.safe_load(f)

        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

    def convert_bb_yolo(self, bb, W, H):
        x, y, w, h = bb
        xc = x + w / 2
        yc = y + h / 2
        xc /= W
        yc /= H
        w  /= W
        h  /= H
        cx = cx.clamp(0, 1)
        cy = cy.clamp(0, 1)
        w  = w.clamp(0, 1)
        h  = h.clamp(0, 1)
        return torch.tensor([xc, yc, w, h], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj_id, img_id = self.samples[idx]

        img_path = (
            self.dataset_root / "data" / f"{obj_id:02d}" /
            "rgb" / f"{img_id:04d}.png"
        )

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        img = self.transform(img)

        entries = self.gt_data[obj_id][img_id]

        boxes = []
        labels = []

        for entry in entries:
            obj_id_gt = entry["obj_id"]

            # skip oggetti che non usi
            if obj_id_gt not in self.OBJ_ID_TO_CLASS:
                continue

            boxes.append(self.convert_bb_yolo(entry["obj_bb"], W, H))
            labels.append(self.OBJ_ID_TO_CLASS[obj_id_gt])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.stack(boxes)
            labels = torch.tensor(labels, dtype=torch.long)

        return {
            "rgb": img,
            "boxes": boxes,
            "labels": labels,
        }


