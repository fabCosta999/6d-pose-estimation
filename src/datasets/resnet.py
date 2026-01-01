from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class ResNetDataset(Dataset):
    def __init__(
        self,
        scene_dataset,
        detection_provider,
        img_size=224,
        padding=0.1,
        transform=True,
    ):
        self.scene_dataset = scene_dataset
        self.detection_provider = detection_provider
        self.img_size = img_size
        self.padding = padding

        if (transform):
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.3,
                    hue=0.05,
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transform.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])


        # flat index: (scene_idx, det_idx)
        self.index = []
        for scene_idx in range(len(scene_dataset)): 
            dets = detection_provider(scene_idx) 
            for det_idx in range(len(dets)): 
                self.index.append((scene_idx, det_idx))

    def __len__(self):
        return len(self.index)

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
        scene_idx, det_idx = self.index[idx]

        scene = self.scene_dataset[scene_idx]
        dets = self.detection_provider(scene_idx)
        obj = dets[det_idx]

        img = Image.open(scene["img_path"]).convert("RGB")
        crop = self.crop_with_padding(img, obj["bbox"])
        if crop is None:
            # salta sample rotto
            return self.__getitem__((idx + 1) % len(self))
        crop = self.transform(crop)

        return {
            "rgb": crop,
            "label": obj["label"],
            "rotation": obj["rotation"],
        }
