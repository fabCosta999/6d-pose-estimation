from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class ResNetDataset(Dataset):
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

        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
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
            # print("BAD CROP:", bbox, "img size:", img.size)
            return None

        return img.crop((x1, y1, x2, y2))

    def __getitem__(self, idx):
        scene = self.scene_dataset[idx]
        det = self.detection_provider(idx)
        if det is None:
            # print(f"missing detection {idx}")
            return self.__getitem__((idx + 1) % len(self))
        
        img = Image.open(scene["img_path"]).convert("RGB")
        crop = self.crop_with_padding(img, det["bbox"])
        if crop is None:
            # salta sample rotto
            return self.__getitem__((idx + 1) % len(self))
        crop = self.transform(crop)

        return {
            "rgb": crop,
            "label": det["label"],
            "rotation": det["rotation"],
        }
