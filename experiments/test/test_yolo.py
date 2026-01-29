import argparse
from ultralytics import YOLO

def main(args):
    model = YOLO(args.yolo_weights)

    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=640,
        batch=16,
        project=args.out_dir,
        name="val",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_weights", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/dataset_yolo/data.yaml")
    parser.add_argument("--out_dir", type=str, default="test_yolo")
    args = parser.parse_args()
    main(args)
