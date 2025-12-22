from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from src.datasets.yolo import YoloDataset
from src.utils.visualization import show_image_with_bbox


def yolo_collate_fn(batch):
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
    }


def main():
    print("[INFO] constructing dataset...")
    dataset = YoloDataset("data/Linemod_preprocessed", split="train")
    print("[INFO] dataset ready")

    print("[INFO] preparing dataloader...")
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=yolo_collate_fn
    )
    print("[INFO] dataloader ready")

    batch = next(iter(loader))

    imgs = batch["rgb"]
    bboxes = batch["boxes"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(4):
        show_image_with_bbox(imgs[i], bboxes[i], ax=axes[i])

    plt.tight_layout()
    plt.savefig("data/sample_visualization.png")
    print("[INFO] Saved visualization to data/sample_visualization.png")


if __name__ == "__main__":
    main()
