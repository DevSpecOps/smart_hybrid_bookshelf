import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

KAGGLE_DATASET = "zygmunt/goodbooks-10k"
REQUIRED_FILES = ["books.csv", "ratings.csv", "tags.csv", "book_tags.csv", "to_read.csv"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="archive")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def already_present(out, files):
    missing = [f for f in files if not (out / f).exists()]
    return len(missing) == 0, missing


def download_kaggle(out):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("FAIL: kaggle package not installed", file=sys.stderr)
        print("  pip install kaggle", file=sys.stderr)
        sys.exit(1)

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print("FAIL: Kaggle authentication failed", file=sys.stderr)
        print("  Ensure ~/.kaggle/kaggle.json exists (or KAGGLE_API_TOKEN env var)", file=sys.stderr)
        print("  " + str(e), file=sys.stderr)
        sys.exit(1)

    staging = out / "_kaggle_download"
    staging.mkdir(parents=True, exist_ok=True)
    print("Downloading " + KAGGLE_DATASET + " ...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(staging), unzip=False)

    zips = list(staging.glob("*.zip"))
    if not zips:
        print("FAIL: no zip downloaded", file=sys.stderr)
        sys.exit(1)
    zpath = zips[0]
    print("Extracting " + zpath.name + " ...")
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(out)
    shutil.rmtree(staging)


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if not args.force:
        ok, missing = already_present(out, REQUIRED_FILES)
        if ok:
            print("OK: all CSVs already present in " + str(out))
            for f in REQUIRED_FILES:
                p = out / f
                print("  " + f + "  " + str(round(p.stat().st_size / 1e6, 2)) + " MB")
            return
        print("Missing: " + ", ".join(missing))

    download_kaggle(out)

    ok, missing = already_present(out, REQUIRED_FILES)
    if not ok:
        print("FAIL: missing after download: " + ", ".join(missing), file=sys.stderr)
        sys.exit(1)
    print("OK: downloaded")
    for f in REQUIRED_FILES:
        p = out / f
        print("  " + f + "  " + str(round(p.stat().st_size / 1e6, 2)) + " MB")


if __name__ == "__main__":
    main()
