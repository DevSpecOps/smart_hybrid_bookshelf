import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests

# Официальные релизы датасета goodbooks-10k на GitHub
RELEASES = {
    "books.csv": "https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/books.zip",
    "ratings.csv": "https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/ratings.zip",
    "tags.csv": "https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/tags.zip",
    "book_tags.csv": "https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/book_tags.zip",
    "to_read.csv": "https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/to_read.zip",
}
REQUIRED_FILES = list(RELEASES.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="archive")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def already_present(out, files):
    missing = [f for f in files if not (out / f).exists()]
    return len(missing) == 0, missing


def download_and_extract(url, out, target_file):
    print(f"Downloading {target_file} ...")
    staging = out / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    zip_path = staging / (target_file + ".zip")

    with requests.get(url, stream=True) as resp:
        if resp.status_code != 200:
            print(f"FAIL: HTTP {resp.status_code} for {url}", file=sys.stderr)
            sys.exit(1)
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"  extracting {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB) ...")
    with zipfile.ZipFile(zip_path, "r") as z:
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

    for f in REQUIRED_FILES:
        p = out / f
        if p.exists():
            p.unlink()

    for target_file, url in RELEASES.items():
        download_and_extract(url, out, target_file)

    ok, missing = already_present(out, REQUIRED_FILES)
    if not ok:
        print("FAIL: missing after download: " + ", ".join(missing), file=sys.stderr)
        sys.exit(1)
    print("OK: downloaded all files from GitHub releases")
    for f in REQUIRED_FILES:
        p = out / f
        print("  " + f + "  " + str(round(p.stat().st_size / 1e6, 2)) + " MB")


if __name__ == "__main__":
    main()