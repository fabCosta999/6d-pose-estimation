from pathlib import Path
import random
from src.datasets.yolo import YoloDataset
from src.utils.export_yolo import export_split
from src.utils.export_yolo import create_data_yaml

def main():
    out_dir = Path("data/dataset_yolo")
    train_ds = YoloDataset("data/Linemod_preprocessed", split="train")
    test_ds  = YoloDataset("data/Linemod_preprocessed", split="test")
    indices = list(range(len(train_ds)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.9 * len(indices))
    train_idx = indices[:split]
    valid_idx = indices[split:]
    print(f"Train images: {len(train_idx)}")
    print(f"Valid images: {len(valid_idx)}")
    print(f"Test images : {len(test_ds)}")
    export_split(train_ds, out_dir, "train", train_idx)
    export_split(train_ds, out_dir, "valid", valid_idx)
    export_split(test_ds,  out_dir, "test", list(range(len(test_ds))))
    create_data_yaml(out_dir)


if __name__ == "__main__":
    main()