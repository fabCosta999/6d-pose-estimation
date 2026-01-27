import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def compute_depth_mean_std(
    dataset_root,
    classes,
    depth_scale=1.0,   
    max_depth=None      
):
    dataset_root = Path(dataset_root)

    sum_depth = 0.0
    sum_sq_depth = 0.0
    num_pixels = 0

    for obj_id in classes:
        depth_dir = dataset_root / "data" / f"{obj_id:02d}" / "depth"
        depth_files = sorted(depth_dir.glob("*.png"))

        for depth_path in tqdm(depth_files, desc=f"Obj {obj_id:02d}"):
            depth = np.array(Image.open(depth_path), dtype=np.float32)

            #(LINEMOD: 0 = invalid)
            valid_mask = depth > 0
            if not np.any(valid_mask):
                continue
            depth = depth[valid_mask] / depth_scale  
            if max_depth is not None:
                depth = depth[depth <= max_depth]
            sum_depth += depth.sum()
            sum_sq_depth += (depth ** 2).sum()
            num_pixels += depth.size

    mean = sum_depth / num_pixels
    std = np.sqrt(sum_sq_depth / num_pixels - mean ** 2)

    return mean, std


def main(args):
    CLASSES = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]

    mean_depth, std_depth = compute_depth_mean_std(
        dataset_root=args.data_root,
        classes=CLASSES,
        depth_scale=1.0,   
        max_depth=2000.0      
    )

    print(f"Depth mean: {mean_depth:.4f} m")
    print(f"Depth std : {std_depth:.4f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/Linemod_preprocessed")

    args = parser.parse_args()
    main(args)

