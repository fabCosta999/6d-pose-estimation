import argparse
from ultralytics import YOLO

def main(args):
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        imgsz=640,
        epochs=args.epochs,
        batch=args.batch_size,
        freeze=10,
        workers=4,
        project=args.out_dir,
        name="train"
    )
    print(results)
    print(results.save_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11s.pt")
    parser.add_argument("--data", type=str, default="data/dataset_yolo/data.yaml")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="train_yolo")
    args = parser.parse_args()
    main(args)