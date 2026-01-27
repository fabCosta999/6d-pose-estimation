#!/usr/bin/env python3
import argparse
import zipfile
from pathlib import Path
import gdown


DEFAULT_URL = (
    "https://drive.google.com/file/d/"
    "18Zm-LKs6Kw0odfqQwA5Xhjv3AzrAMKsa/view?usp=drive_link"
    )
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "Linemod_preprocessed.zip"


def download_dataset(url: str):
    if ZIP_PATH.exists():
        print(f"[INFO] Zip already exists: {ZIP_PATH}")
        return

    print("[INFO] Downloading LINEMOD dataset...")
    gdown.download(
        url=url,
        output=str(ZIP_PATH),
        quiet=False,
        fuzzy=True,
    )


def extract_dataset():
    print("[INFO] Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)


def main(args):
    download_dataset(args.url)
    extract_dataset()
    print("[INFO] Dataset ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="Google Drive URL for the LINEMOD zip file",
    )
    args = parser.parse_args()
    main(args)
