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


    def rotation_matrix_to_quaternion(self, R):
        """
        R: tensor (3, 3)
        ritorna: tensor (4,) [qw, qx, qy, qz]
        """
        trace = R.trace()

        if trace > 0:
            s = torch.sqrt(trace + 1.0) * 2
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

        q = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)
        return q / torch.norm(q)  



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
        rotations = []

        for entry in entries:
            obj_id_gt = entry["obj_id"]

            
            if obj_id_gt not in self.OBJ_ID_TO_CLASS:
                continue

            boxes.append(self.convert_bb_yolo(entry["obj_bb"], W, H))
            labels.append(self.OBJ_ID_TO_CLASS[obj_id_gt])
            R_flat = entry["cam_R_m2c"]
            R = torch.tensor(R_flat, dtype=torch.float32).view(3, 3)
            q = self.rotation_matrix_to_quaternion(R)
            rotations.append(q)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            rotations = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.stack(boxes)
            labels = torch.tensor(labels, dtype=torch.long)
            rotations = torch.stack(rotations)

        return {
            "rgb": img,
            "boxes": boxes,
            "labels": labels,
            "rotations": rotations,
        }


