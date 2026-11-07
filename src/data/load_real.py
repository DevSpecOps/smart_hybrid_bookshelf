import argparse
from pathlib import Path

import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tag_strings(book_tags, tags, top_k=20):
    merged = book_tags.merge(tags, on="tag_id", how="left")
    merged = merged.sort_values(
        ["goodreads_book_id", "count"], ascending=[True, False]
    )
    top = merged.groupby("goodreads_book_id").head(top_k)
    agg = top.groupby("goodreads_book_id")["tag_name"].apply(
        lambda s: "|".join(s.dropna().astype(str).unique())
    )
    return agg.to_dict()


def build_books(books, book_tags, tags):
    tag_map = build_tag_strings(book_tags, tags, top_k=20)
    out = pd.DataFrame({
        "book_id": books["book_id"].astype(int),
        "goodreads_book_id": books["goodreads_book_id"].astype(int),
        "title": books["title"].astype(str),
        "original_title": books.get("original_title", books["title"]).fillna("").astype(str),
        "authors": books.get("authors", "").fillna("").astype(str),
    })
    out["author"] = out["authors"].str.split(",").str[0].str.strip()
    year_col = "original_publication_year" if "original_publication_year" in books.columns else None
    if year_col:
        out["year"] = pd.to_numeric(books[year_col], errors="coerce").fillna(0).astype(int)
    else:
        out["year"] = 0
    out["tags"] = out["goodreads_book_id"].map(tag_map).fillna("")
    return out


def filter_ratings(ratings, min_user, min_book):
    user_counts = ratings.groupby("user_id").size()
    keep_users = user_counts[user_counts >= min_user].index
    df = ratings[ratings["user_id"].isin(keep_users)].copy()
    book_counts = df.groupby("book_id").size()
    keep_books = book_counts[book_counts >= min_book].index
    df = df[df["book_id"].isin(keep_books)].copy()
    return df.reset_index(drop=True)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    arc = Path(args.archive)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    books_raw = pd.read_csv(arc / "books.csv")
    ratings_raw = pd.read_csv(arc / "ratings.csv")
    tags_raw = pd.read_csv(arc / "tags.csv")
    book_tags_raw = pd.read_csv(arc / "book_tags.csv")

    min_user = cfg["data"]["min_user_ratings"]
    min_book = cfg["data"]["min_book_ratings"]

    ratings = filter_ratings(ratings_raw, min_user, min_book)
    books = build_books(books_raw, book_tags_raw, tags_raw)

    keep_book_ids = set(ratings["book_id"].unique())
    books = books[books["book_id"].isin(keep_book_ids)].reset_index(drop=True)

    keep_goodreads = set(books["goodreads_book_id"].unique())
    book_tags = book_tags_raw[book_tags_raw["goodreads_book_id"].isin(keep_goodreads)].reset_index(drop=True)
    used_tag_ids = set(book_tags["tag_id"].unique())
    tags = tags_raw[tags_raw["tag_id"].isin(used_tag_ids)].reset_index(drop=True)

    ratings.to_csv(out / "ratings.csv", index=False)
    books.to_csv(out / "books.csv", index=False)
    book_tags.to_csv(out / "book_tags.csv", index=False)
    tags.to_csv(out / "tags.csv", index=False)

    users = pd.DataFrame({"user_id": sorted(ratings["user_id"].unique())})
    users.to_csv(out / "users.csv", index=False)

    print("books: " + str(len(books)))
    print("users: " + str(len(users)))
    print("ratings: " + str(len(ratings)))
    print("book_tags: " + str(len(book_tags)))
    print("tags: " + str(len(tags)))
    print("density: " + str(round(len(ratings) / (len(users) * len(books)), 8)))


if __name__ == "__main__":
    main()
