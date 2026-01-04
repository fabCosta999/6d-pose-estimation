from PIL import Image
from tqdm import tqdm
from pathlib import Path
import yaml


def export_split(dataset, out_dir, split_name, indices):
    img_out = out_dir / split_name / "images"
    lbl_out = out_dir / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for new_id, ds_idx in enumerate(tqdm(indices, desc=f"Exporting {split_name}")):
        sample = dataset[ds_idx]
        img = sample["rgb"]              
        box = sample["box"].squeeze(0)      
        label = int(sample["label"].item()) 
        fname = f"{new_id:06d}"
        img_pil = Image.fromarray(
            (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
        )
        img_pil.save(img_out / f"{fname}.png")
        xc, yc, w, h = box.tolist()
        with open(lbl_out / f"{fname}.txt", "w") as f:
            f.write(f"{label} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                

def create_data_yaml(dataset_dir):
    data_yaml = {
        "path": "/content/6d-pose-estimation/data/dataset_yolo",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 13,
        "names": [
            "obj_01","obj_02","obj_04","obj_05","obj_06",
            "obj_08","obj_09","obj_10","obj_11","obj_12",
            "obj_13","obj_14","obj_15",
        ],
    }

    with open(dataset_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print("[INFO] data.yaml created")


