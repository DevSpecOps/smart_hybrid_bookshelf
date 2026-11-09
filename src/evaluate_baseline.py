import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.baseline import (
    PopularityRecommender,
    ContentBasedRecommender,
    ItemBasedCF,
    SVDRecommender,
)
from src.data.split import random_leave_one_out_split
from src.evaluation.harness import EvaluationHarness
from src.evaluation.report import save_metrics
from src.models.adapters import ModelAdapter
from src.models.segmented_hybrid import SegmentedHybrid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-users", type=int, default=None)
    return parser.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_models(raw_dir, cfg):
    models_cfg = cfg.get("models", {})
    pop_cfg = models_cfg.get("popularity", {})
    knn_cfg = models_cfg.get("item_knn", {})
    svd_cfg = models_cfg.get("svd", {})
    tt_cfg = models_cfg.get("two_tower", {})

    books = pd.read_csv(raw_dir / "books.csv")
    book_tags = pd.read_csv(raw_dir / "book_tags.csv")
    tags = pd.read_csv(raw_dir / "tags.csv")

    pop_min = pop_cfg.get("min_ratings", 10)
    pop_score = pop_cfg.get("score", "bayesian")
    knn_neighbors = knn_cfg.get("k_neighbors", 100)
    knn_predict = knn_cfg.get("k_predict", 20)
    svd_components = svd_cfg.get("n_components", 32)
    svd_iter = svd_cfg.get("n_iter", 10)

    candidates = [
        ("popularity", lambda: PopularityRecommender(min_ratings=pop_min, score=pop_score), None),
        ("content", lambda: ContentBasedRecommender(), (books, book_tags, tags)),
        ("itemknn", lambda: ItemBasedCF(n_neighbors=knn_neighbors, k_predict=knn_predict), None),
    ]
    if SVDRecommender is not None:
        candidates.append((
            "svd",
            lambda: SVDRecommender(n_factors=svd_components, n_epochs=svd_iter),
            None,
        ))

    base = {}
    for name, factory, extra in candidates:
        try:
            base[name] = ModelAdapter(name, factory(), fit_extra=extra)
        except Exception as e:
            print("WARN " + name + " build failed: " + type(e).__name__ + ": " + str(e))

    checkpoint = Path(tt_cfg.get("checkpoint", "data/processed/two_tower.pt"))
    if checkpoint.exists():
        try:
            from src.models.two_tower import TwoTowerRecommender
            tt = TwoTowerRecommender.load(checkpoint)
            base["two_tower"] = ModelAdapter("two_tower", tt)
            print("loaded two_tower from " + str(checkpoint))
        except Exception as e:
            print("WARN two_tower load failed: " + type(e).__name__ + ": " + str(e))
    else:
        print("WARN two_tower checkpoint missing (run: python scripts/train_two_tower.py)")

    hybrid_cfg = cfg.get("hybrid", {})
    hybrid = SegmentedHybrid(
        models=base,
        weights=hybrid_cfg.get("weights", {}),
        cold_threshold=hybrid_cfg.get("cold_threshold", 50),
        k_rrf=hybrid_cfg.get("k_rrf", 60),
        candidates_per_model=hybrid_cfg.get("candidates_per_model", 50),
    )

    models = dict(base)
    models["hybrid"] = hybrid
    return models


def print_summary(results):
    for name, per_k in results.items():
        for k, m in per_k.items():
            g = m["global"]
            line = "{0:12s} {1:6s} P={2:.4f} R={3:.4f} nDCG={4:.4f} cov={5:.3f} n={6}"
            print(line.format(
                name, k,
                g["precision"], g["recall"], g["ndcg"],
                m["coverage"], g["n_users"],
            ))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    raw = pd.read_csv("data/raw/ratings.csv")
    seed = cfg["project"]["seed"]
    min_r = cfg["data"]["min_user_ratings"]
    train, test = random_leave_one_out_split(raw, min_ratings=min_r, seed=seed)

    if args.max_users:
        keep = test["user_id"].drop_duplicates().head(args.max_users).tolist()
        test = test[test["user_id"].isin(keep)].reset_index(drop=True)

    print("train: " + str(train.shape) + "  test: " + str(test.shape))

    models = build_models(Path("data/raw"), cfg)
    print("built models: " + str(list(models.keys())))
    if not models:
        raise RuntimeError("no models built")

    harness = EvaluationHarness(
        models=models,
        k_values=cfg["evaluation"]["k_values"],
        cold_threshold=cfg["hybrid"]["cold_threshold"],
    )
    results = harness.evaluate(train, test)
    out = save_metrics(results, args.output)
    print("metrics saved to " + str(out))
    print_summary(results)


if __name__ == "__main__":
    main()