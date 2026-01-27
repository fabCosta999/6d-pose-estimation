import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from src.datasets.scene import LinemodSceneDataset
from src.datasets.yolo import YoloDataset
from src.utils.visualization import show_image_with_bbox


def yolo_collate_fn(batch):
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
    }


def main(args):
    print("[INFO] constructing dataset...")
    dataset_scene = LinemodSceneDataset(args.data_root, split="train")
    dataset = YoloDataset(dataset_scene)
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
    plt.savefig(f"{args.out_dir}/sample_visualization.png")
    print(f"[INFO] Saved visualization to {args.out_dir}/sample_visualization.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/Linemod_preprocessed")
    parser.add_argument("--out_dir", type=str, default="visualize_data")
    args = parser.parse_args()
    main(args)
