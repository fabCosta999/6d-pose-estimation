from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class RGBDDataset(Dataset):
    def __init__(
        self,
        scene_dataset,
        detection_provider,
        img_size=224,
        padding=0,
    ):
        self.scene_dataset = scene_dataset
        self.detection_provider = detection_provider
        self.img_size = img_size
        self.padding = padding
        self.depth_mean = 990.7
        self.depth_std  = 311.8

        self.rgb_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),

            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # (1, H, W)
        ])

    def __len__(self):
        return len(self.scene_dataset)

    def crop_with_padding(self, img, bbox):
        x, y, w, h = bbox
        if w <= 1 or h <= 1:
            return None

        pad_w = w * self.padding
        pad_h = h * self.padding
        cx = x + w / 2
        cy = y + h / 2

        w2 = w + 2 * pad_w
        h2 = h + 2 * pad_h

        x1 = int(max(0, cx - w2 / 2))
        y1 = int(max(0, cy - h2 / 2))
        x2 = int(min(img.width,  cx + w2 / 2))
        y2 = int(min(img.height, cy + h2 / 2))

        if x2 <= x1 or y2 <= y1:
            return None

        return img.crop((x1, y1, x2, y2))

    def __getitem__(self, idx):
        scene = self.scene_dataset[idx]
        det = self.detection_provider(idx)

        if det is None:
            return self.__getitem__((idx + 1) % len(self))

        # RGB
        rgb_img = Image.open(scene["img_path"]).convert("RGB")
        rgb_crop = self.crop_with_padding(rgb_img, det["bbox"])
        if rgb_crop is None:
            return self.__getitem__((idx + 1) % len(self))
        rgb_crop = self.rgb_transform(rgb_crop)

        # DEPTH: ricarichiamo come PIL
        depth_path = scene["depth_path"]
        depth_img = Image.open(depth_path)
        depth_crop = self.crop_with_padding(depth_img, det["bbox"])
        if depth_crop is None:
            return self.__getitem__((idx + 1) % len(self))
        depth_crop = self.depth_transform(depth_crop)
        depth_crop = depth_crop * self.scene_dataset.depth_scale
        depth_crop = (depth_crop - self.depth_mean) / self.depth_std

        return {
            "rgb": rgb_crop,            # (3, H, W)
            "depth": depth_crop,        # (1, H, W)
            "label": det["label"],
            "rotation": det["rotation"],
            "translation": det["translation"],
            "bbox": det["bbox"],
        }
