import os
import torch
import torchvision.transforms as T
from PIL import ImageDraw, ImageFont
from src.datasets.scene import LinemodSceneDataset, GTDetections
from src.datasets.resnet import ResNetDataset  


def tensor_to_pil(img_tensor):
    """Assume img is (3,H,W) normalized in [0,1] or ImageNet stats."""
    img = img_tensor.clone().detach()

    # if ImageNet-normalized
    if img.min() < 0:
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        img = img * std + mean

    img = img.clamp(0, 1)
    img = (img * 255).byte()
    return T.ToPILImage()(img)


def draw_label(img, text):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw.rectangle((0, 0, img.width, 40), fill=(0, 0, 0))
    draw.text((10, 5), text, fill=(255, 255, 255), font=font)
    return img


def save_one_image_per_class(dataset, out_dir="crops"):
    os.makedirs(out_dir, exist_ok=True)

    saved = set()

    for item in dataset:
        class_id = item["label"]
        img = item["rgb"]

        if class_id in saved:
            continue

        pil_img = tensor_to_pil(img)
        label_text = f"Class {class_id}"
        pil_img = draw_label(pil_img, label_text)

        out_path = os.path.join(out_dir, f"class_{class_id:02d}.png")
        pil_img.save(out_path)

        saved.add(class_id)

        if len(saved) == 13:
            break

    print(f"Saved {len(saved)} images in '{out_dir}'")


if __name__ == "__main__":
    scene_ds = LinemodSceneDataset("data/Linemod_preprocessed", split="test")
    gt = GTDetections(scene_ds)
    ds = ResNetDataset(scene_ds, gt)
    save_one_image_per_class(ds)
