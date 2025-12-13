from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.datasets.linemod import LinemodDataset
from src.utils.visualization import show_image_with_bbox


def main():
    dataset = LinemodDataset("data/Linemod_preprocessed", split="train")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))

    imgs = batch["rgb"]
    bboxes = batch["bbox"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(4):
        show_image_with_bbox(imgs[i], bboxes[i], ax=axes[i])

    plt.show()


if __name__ == "__main__":
    main()
