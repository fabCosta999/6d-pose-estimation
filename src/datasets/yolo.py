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




