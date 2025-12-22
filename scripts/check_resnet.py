import torch
from torch.utils.data import DataLoader

from src.datasets.scene import LinemodSceneDataset, GTDetections
from src.datasets.resnet import ResNetDataset


def main():
    dataset_root = "data/Linemod_preprocessed"

    # Scene dataset
    scene_ds = LinemodSceneDataset(
        dataset_root=dataset_root,
        split="train",
        img_size=640,
    )

    # GT detections (no YOLO)
    gt_detections = GTDetections(scene_ds)

    # ResNet dataset
    resnet_ds = ResNetDataset(
        scene_dataset=scene_ds,
        detection_provider=gt_detections,
        img_size=224,
        padding=0.2,
    )

    print("Numero scene:", len(scene_ds))
    print("Numero oggetti (flat):", len(resnet_ds))

    # DataLoader
    loader = DataLoader(
        resnet_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # per debug
    )

    batch = next(iter(loader))

    print("\n=== BATCH CHECK ===")
    print("rgb:", batch["rgb"].shape)          # [B, 3, 224, 224]
    print("label:", batch["label"].shape)      # [B]
    print("rotation:", batch["rotation"].shape)  # [B, 4]

    print("\n=== SAMPLE VALUES ===")
    print("labels:", batch["label"])
    print("rotation[0]:", batch["rotation"][0])
    print("||q||:", torch.norm(batch["rotation"][0]).item())


if __name__ == "__main__":
    main()
