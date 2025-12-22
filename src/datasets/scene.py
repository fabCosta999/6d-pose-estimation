import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml
import torchvision.transforms as transforms
from PIL import Image
from src.utils.quaternions import rotation_matrix_to_quaternion

class LinemodSceneDataset(Dataset):
    CLASSES = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    OBJ_ID_TO_CLASS = {obj_id: i for i, obj_id in enumerate(CLASSES)}

    def __init__(self, dataset_root, split="train", img_size=640):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.img_size = img_size

        self.samples = []
        self.gt_data = {}

        for obj_id in self.CLASSES:
            obj_dir = self.dataset_root / "data" / f"{obj_id:02d}"

            with open(obj_dir / f"{split}.txt") as f:
                for line in f:
                    self.samples.append((obj_id, int(line.strip())))

            with open(obj_dir / "gt.yml") as f:
                self.gt_data[obj_id] = yaml.safe_load(f)

        self.rgb_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),  # depth rimane 1xHxW
        ])


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        obj_id, img_id = self.samples[idx]

        base_dir = self.dataset_root / "data" / f"{obj_id:02d}"

        img_path = base_dir / "rgb" / f"{img_id:04d}.png"
        depth_path = base_dir / "depth" / f"{img_id:04d}.png"

        img = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path)

        W, H = img.size

        rgb = self.rgb_transform(img)
        depth = self.depth_transform(depth)

        objects = []

        for entry in self.gt_data[obj_id][img_id]:
            obj_id_gt = entry["obj_id"]
            if obj_id_gt not in self.OBJ_ID_TO_CLASS:
                continue

            R = torch.tensor(entry["cam_R_m2c"], dtype=torch.float32).view(3, 3)
            q = rotation_matrix_to_quaternion(R)

            objects.append({
                "bbox": entry["obj_bb"],          # pixel
                "label": self.OBJ_ID_TO_CLASS[obj_id_gt],
                "rotation": q,
            })

        return {
            "img_path": img_path,
            "rgb": rgb,
            "depth": depth,
            "objects": objects,
            "size": (W, H),
        }



class GTDetections:
    def __init__(self, scene_dataset):
        self.scene_dataset = scene_dataset

    def __call__(self, idx):
        # ritorna direttamente gli oggetti GT
        return self.scene_dataset[idx]["objects"]
