import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml
import torchvision.transforms as transforms
from PIL import Image
from src.utils.quaternions import rotation_matrix_to_quaternion
import numpy as np

class LinemodSceneDataset(Dataset):
    CLASSES = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    OBJ_ID_TO_CLASS = {obj_id: i for i, obj_id in enumerate(CLASSES)}

    def __init__(self, dataset_root, split="train", split_ratio=0.8, seed=42):
        np.random.seed(seed) 
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.samples = []
        self.gt_data = {}

        for obj_id in self.CLASSES:
            obj_dir = self.dataset_root / "data" / f"{obj_id:02d}"

            rgb_dir = obj_dir / "rgb"
            num_images = len(list(rgb_dir.glob("*.png")))


            indexes = np.arange(num_images)
            np.random.shuffle(indexes)
            
            split_point = int(split_ratio * num_images)
            if split == "train":
                img_ids = indexes[:split_point]
            else:
                img_ids = indexes[split_point:]

            for img_id in img_ids:
                self.samples.append((obj_id, img_id))

            with open(obj_dir / "gt.yml") as f:
                self.gt_data[obj_id] = yaml.safe_load(f)
        

        any_obj = self.CLASSES[0]
        info_path = self.dataset_root / "data" / f"{any_obj:02d}" / "info.yml"

        with open(info_path) as f:
            info = yaml.safe_load(f)

        cam_info = next(iter(info.values()))

        self.K = torch.tensor(cam_info["cam_K"], dtype=torch.float32).view(3, 3)
        self.depth_scale = cam_info.get("depth_scale", 1.0) / 1000 # converting it to meters

        self.rgb_transform = transforms.ToTensor()


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        obj_id, img_id = self.samples[idx]

        base_dir = self.dataset_root / "data" / f"{obj_id:02d}"

        img_path = base_dir / "rgb" / f"{img_id:04d}.png"
        depth_path = base_dir / "depth" / f"{img_id:04d}.png"

        img = Image.open(img_path).convert("RGB")

        W, H = img.size

        rgb = self.rgb_transform(img)

        object = None
        for entry in self.gt_data[obj_id][img_id]:
            if int(entry["obj_id"]) == obj_id:
                object = entry
                break                

        if object is None:
            raise RuntimeError(
                f"Object {obj_id} not found in image {img_id}"
            )

        R = torch.tensor(object["cam_R_m2c"], dtype=torch.float32).view(3, 3)
        q = rotation_matrix_to_quaternion(R)
        t = torch.tensor(object["cam_t_m2c"], dtype=torch.float32).view(3)

        return {
            "img_path": img_path,
            "depth_path": depth_path, 
            "cam_intrinsics": self.K,
            "rgb": rgb,
            "bbox": object["obj_bb"],
            "label": self.OBJ_ID_TO_CLASS[obj_id],
            "rotation": q,
            "translation":t, 
            "size": (W, H),
        }
    



class GTDetections:
    def __init__(self, scene_dataset):
        self.scene_dataset = scene_dataset

    def __call__(self, idx):
        # ritorna direttamente l'oggetto GT
        sample = self.scene_dataset[idx]
        return {
            "rgb": sample["rgb"],
            "bbox": sample["bbox"],
            "label": sample["label"],
            "rotation": sample["rotation"],
            "translation":sample["translation"],   
        }
