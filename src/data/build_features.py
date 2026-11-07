import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def build_user_features(ratings):
    g = ratings.groupby("user_id")["rating"]
    feats = pd.DataFrame({
        "user_avg_rating": g.mean(),
        "user_n_ratings": g.size(),
        "user_std_rating": g.std().fillna(0.0),
    }).reset_index()
    return feats


def build_item_features(ratings):
    g = ratings.groupby("book_id")["rating"]
    feats = pd.DataFrame({
        "item_avg_rating": g.mean(),
        "item_n_ratings": g.size(),
        "item_std_rating": g.std().fillna(0.0),
    }).reset_index()
    feats["item_popularity_rank"] = (
        feats["item_n_ratings"].rank(ascending=False, method="min").astype(int)
    )
    return feats


def tag_entropy(tag_string):
    if pd.isna(tag_string):
        return 0.0
    tags = [t for t in str(tag_string).split("|") if t]
    if not tags:
        return 0.0
    _, counts = np.unique(tags, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def build_item_metadata(books):
    books = books.copy()
    books["item_tag_diversity"] = books["tags"].apply(tag_entropy)
    cols = ["book_id", "title", "tags", "author", "year", "item_tag_diversity"]
    return books[cols]


def main():
    args = parse_args()
    raw = Path(args.raw)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(raw / "ratings.csv")
    books = pd.read_csv(raw / "books.csv")

    user_feats = build_user_features(ratings)
    item_feats = build_item_features(ratings)
    item_meta = build_item_metadata(books)
    item_feats = item_feats.merge(item_meta, on="book_id", how="left")

    user_feats.to_csv(out / "user_features.csv", index=False)
    item_feats.to_csv(out / "item_features.csv", index=False)

    print("user_features: " + str(user_feats.shape))
    print("item_features: " + str(item_feats.shape))


if __name__ == "__main__":
    main()
