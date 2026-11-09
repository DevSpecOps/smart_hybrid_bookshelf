import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--max-users", type=int, default=None)
    return parser.parse_args()


def run(cmd):
    print()
    print(">>> " + " ".join(str(c) for c in cmd))
    rc = subprocess.call([str(c) for c in cmd])
    if rc != 0:
        print("FAIL: " + " ".join(str(c) for c in cmd))
        sys.exit(rc)


def file_exists(p):
    return Path(p).exists()


def check_archive():
    return all(file_exists("archive/" + f) for f in ["books.csv","ratings.csv","tags.csv","book_tags.csv","to_read.csv"])


def check_raw():
    return all(file_exists("data/raw/" + f) for f in ["ratings.csv","books.csv","users.csv","book_tags.csv","tags.csv"])


def check_features():
    return all(file_exists("data/processed/" + f) for f in ["user_features.csv","item_features.csv"])


def check_checkpoint():
    return file_exists("data/processed/two_tower.pt")


def main():
    args = parse_args()

    if args.force or not check_archive():
        run([sys.executable, "scripts/download_data.py"])
    else:
        print("SKIP download_data (archive exists)")

    if args.force or not check_raw():
        run([sys.executable, "-m", "src.data.load_real", "--archive", "archive", "--output", "data/raw", "--config", args.config])
    else:
        print("SKIP load_real (data/raw exists)")

    if args.force or not check_features():
        run([sys.executable, "-m", "src.data.build_features", "--raw", "data/raw", "--output", "data/processed"])
    else:
        print("SKIP build_features (data/processed exists)")

    if not args.skip_train:
        if args.force or not check_checkpoint():
            cmd = [sys.executable, "scripts/train_two_tower.py", "--config", args.config]
            if args.full:
                cmd.append("--full")
            run(cmd)
        else:
            print("SKIP train_two_tower (checkpoint exists; use --force to retrain)")
    else:
        print("SKIP train_two_tower (--skip-train)")

    cmd = [sys.executable, "-m", "src.evaluate_baseline", "--config", args.config, "--output", "reports/metrics_final.json"]
    if args.max_users:
        cmd.extend(["--max-users", str(args.max_users)])
    run(cmd)

    print()
    print("PIPELINE COMPLETE: reports/metrics_final.json")


if __name__ == "__main__":
    main()
