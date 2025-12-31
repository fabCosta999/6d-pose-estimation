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

    def __init__(self, dataset_root, split="train"):
        self.dataset_root = Path(dataset_root)
        self.split = split

        self.samples = []
        self.gt_data = {}

        for obj_id in self.CLASSES:
            obj_dir = self.dataset_root / "data" / f"{obj_id:02d}"

            rgb_dir = obj_dir / "rgb"
            num_images = len(list(rgb_dir.glob("*.png")))
            split_point = int(0.8 * num_images)
            if split == "train":
                img_ids = range(0, split_point)
            else:
                img_ids = range(split_point, num_images)

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
        self.depth_scale = cam_info.get("depth_scale", 1.0)

        self.rgb_transform = transforms.ToTensor()
        self.depth_transform = transforms.ToTensor()


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
            t = torch.tensor(entry["cam_t_m2c"], dtype=torch.float32).view(3)

            objects.append({
                "bbox": entry["obj_bb"],          # pixel
                "label": self.OBJ_ID_TO_CLASS[obj_id_gt],
                "rotation": q,
                "translation":t, 
            })

        return {
            "img_path": img_path,
            "depth": depth * self.depth_scale,
            "cam_intrinsics": self.K,
            "rgb": rgb,
            "objects": objects,
            "size": (W, H),
        }



class GTDetections:
    def __init__(self, scene_dataset):
        self.scene_dataset = scene_dataset

    def __call__(self, idx):
        # ritorna direttamente gli oggetti GT
        return self.scene_dataset[idx]["objects"]
