import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.baseline import PopularityRecommender, ContentBasedRecommender, ItemBasedCF, SVDRecommender
from src.data.split import random_leave_one_out_split
from src.evaluation.harness import EvaluationHarness
from src.models.adapters import ModelAdapter
from src.models.segmented_hybrid import SegmentedHybrid

cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
raw = pd.read_csv("data/raw/ratings.csv")
train, test = random_leave_one_out_split(raw, min_ratings=5, seed=42)
keep = test["user_id"].drop_duplicates().head(500).tolist()
test = test[test["user_id"].isin(keep)].reset_index(drop=True)

books = pd.read_csv("data/raw/books.csv")
book_tags = pd.read_csv("data/raw/book_tags.csv")
tags = pd.read_csv("data/raw/tags.csv")

base = {
    "popularity": ModelAdapter("popularity", PopularityRecommender(min_ratings=10, score="bayesian")),
    "content": ModelAdapter("content", ContentBasedRecommender(), fit_extra=(books, book_tags, tags)),
    "itemknn": ModelAdapter("itemknn", ItemBasedCF(n_neighbors=100, k_predict=20)),
    "svd": ModelAdapter("svd", SVDRecommender(n_factors=32, n_epochs=10)),
}

print("Fitting base models once...")
for name, adapter in base.items():
    adapter.fit(train)
print("Base models fitted.")

configs = [
    {"name": "A_current",   "threshold": 50,  "weights": {"cold_start": {"popularity": 0.6, "content": 0.4}, "active": {"content": 0.50, "itemknn": 0.30, "popularity": 0.15, "svd": 0.05}}},
    {"name": "B_knn_heavy", "threshold": 50,  "weights": {"cold_start": {"popularity": 0.6, "content": 0.4}, "active": {"content": 0.35, "itemknn": 0.45, "popularity": 0.15, "svd": 0.05}}},
    {"name": "C_content_heavy", "threshold": 50, "weights": {"cold_start": {"popularity": 0.6, "content": 0.4}, "active": {"content": 0.60, "itemknn": 0.25, "popularity": 0.10, "svd": 0.05}}},
    {"name": "D_no_svd",    "threshold": 50,  "weights": {"cold_start": {"popularity": 0.6, "content": 0.4}, "active": {"content": 0.45, "itemknn": 0.35, "popularity": 0.20}}},
    {"name": "E_th150",     "threshold": 150, "weights": {"cold_start": {"popularity": 0.6, "content": 0.4}, "active": {"content": 0.45, "itemknn": 0.35, "popularity": 0.15, "svd": 0.05}}},
    {"name": "F_balanced",  "threshold": 100, "weights": {"cold_start": {"popularity": 0.5, "content": 0.5}, "active": {"content": 0.40, "itemknn": 0.40, "popularity": 0.15, "svd": 0.05}}},
]

results = {}
for c in configs:
    hyb = SegmentedHybrid(
        models=base,
        weights=c["weights"],
        cold_threshold=c["threshold"],
        k_rrf=60,
        candidates_per_model=50,
    )
    h = EvaluationHarness(
        models={**base, "hybrid": hyb},
        k_values=[10, 20],
        cold_threshold=c["threshold"],
    )
    h.models["hybrid"].fit(train)
    recs_by_user = {}
    test_relevant = test.groupby("user_id")["book_id"].apply(set).to_dict()
    for uid in test_relevant:
        try:
            recs_by_user[uid] = hyb.recommend(uid, n=20)
        except Exception:
            recs_by_user[uid] = []
    train_counts = train.groupby("user_id").size().to_dict()
    segments = {uid: ("cold_start" if train_counts.get(uid, 0) < c["threshold"] else "active") for uid in test_relevant}
    catalog = set(train["book_id"].unique()) | set(test["book_id"].unique())
    k10 = h._aggregate(recs_by_user, test_relevant, segments, 10, len(catalog))
    k20 = h._aggregate(recs_by_user, test_relevant, segments, 20, len(catalog))
    entry = {
        "threshold": c["threshold"],
        "weights": c["weights"],
        "hybrid_R@10": k10["global"]["recall"],
        "hybrid_nDCG@10": k10["global"]["ndcg"],
        "hybrid_cov@10": k10["coverage"],
        "hybrid_R@20": k20["global"]["recall"],
        "cold_n": k10["cold_start"]["n_users"],
        "active_n": k10["active"]["n_users"],
    }
    results[c["name"]] = entry
    print('{0:18s} th={1:3d} R@10={2:.4f} nDCG@10={3:.4f} cov@10={4:.3f} R@20={5:.4f} cold(n)={6} active(n)={7}'.format(
        c["name"], c["threshold"], entry["hybrid_R@10"], entry["hybrid_nDCG@10"],
        entry["hybrid_cov@10"], entry["hybrid_R@20"], entry["cold_n"], entry["active_n"]))

Path("reports").mkdir(exist_ok=True)
Path("reports/sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print()
best = max(results.items(), key=lambda kv: kv[1]["hybrid_R@10"])
print('BEST by R@10: ' + best[0] + '  R@10=' + str(round(best[1]["hybrid_R@10"], 4)))