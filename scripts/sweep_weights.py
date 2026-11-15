import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import yaml
from src.baseline import PopularityRecommender, ContentBasedRecommender, ItemBasedCF, SVDRecommender
from src.data.split import random_leave_one_out_split, split_users_for_sweep_eval
from src.evaluation.harness import EvaluationHarness
from src.models.adapters import ModelAdapter
from src.models.segmented_hybrid import SegmentedHybrid
from src.models.two_tower import TwoTowerRecommender

cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
raw = pd.read_csv("data/raw/ratings.csv")
seed = cfg["project"]["seed"]
min_r = cfg["data"]["min_user_ratings"]
train, test = random_leave_one_out_split(raw, min_ratings=min_r, seed=seed)
relevant = test[test["rating"] >= 4]
eligible = sorted(relevant["user_id"].unique().tolist())
eval_users, sweep_users = split_users_for_sweep_eval(eligible, seed=seed)
sweep_users = sweep_users[:500]
test = test[test["user_id"].isin(sweep_users)].reset_index(drop=True)
print("sweep users (disjoint from eval " + str(len(eval_users)) + "): " + str(len(sweep_users)))

books = pd.read_csv("data/raw/books.csv")
book_tags = pd.read_csv("data/raw/book_tags.csv")
tags = pd.read_csv("data/raw/tags.csv")

base = {
    "popularity": ModelAdapter("popularity", PopularityRecommender(min_ratings=10, score="bayesian")),
    "content": ModelAdapter("content", ContentBasedRecommender(), fit_extra=(books, book_tags, tags)),
    "itemknn": ModelAdapter("itemknn", ItemBasedCF(n_neighbors=100, k_predict=20)),
    "svd": ModelAdapter("svd", SVDRecommender(n_factors=32, n_epochs=10)),
}
base["two_tower"] = ModelAdapter("two_tower", TwoTowerRecommender.load("data/processed/two_tower.pt"))
print("Fitting base models on train split (including Two-Tower seen)...")
for n, a in base.items():
    try:
        a.fit(train)
    except Exception as e:
        print("WARN", n, type(e).__name__, e)
print("done")

current_w = cfg["hybrid"]["weights"]
configs = [
    {"name": "current_config", "threshold": cfg["hybrid"]["cold_threshold"], "weights": current_w},
    {"name": "A_current", "threshold": 50, "weights": {"cold_start": {"popularity": 0.6, "content": 0.4}, "active": {"content": 0.50, "itemknn": 0.30, "popularity": 0.15, "svd": 0.05}}},
    {"name": "B_knn_heavy", "threshold": 50, "weights": {"cold_start": {"popularity": 0.6, "content": 0.4}, "active": {"content": 0.35, "itemknn": 0.45, "popularity": 0.15, "svd": 0.05}}},
    {"name": "C_content_heavy", "threshold": 50, "weights": {"cold_start": {"popularity": 0.6, "content": 0.4}, "active": {"content": 0.60, "itemknn": 0.25, "popularity": 0.10, "svd": 0.05}}},
    {"name": "D_no_svd", "threshold": 50, "weights": {"cold_start": {"popularity": 0.6, "content": 0.4}, "active": {"content": 0.45, "itemknn": 0.35, "popularity": 0.20}}},
    {"name": "F_balanced", "threshold": 100, "weights": {"cold_start": {"popularity": 0.5, "content": 0.5}, "active": {"content": 0.40, "itemknn": 0.40, "popularity": 0.15, "svd": 0.05}}},
    {"name": "G_tt_20", "threshold": 20, "weights": {"cold_start": {"content": 0.85, "popularity": 0.15}, "active": {"itemknn": 0.45, "content": 0.30, "two_tower": 0.20, "popularity": 0.05}}},
]

results = {}
for c in configs:
    hyb = SegmentedHybrid(models=base, weights=c["weights"], cold_threshold=c["threshold"], k_rrf=60, candidates_per_model=50)
    h = EvaluationHarness(models={**base, "hybrid": hyb}, k_values=[10, 20], cold_threshold=c["threshold"])
    hyb.fit(train)
    trel = test[test["rating"] >= 4].groupby("user_id")["book_id"].apply(set).to_dict()
    recs = {}
    for uid in trel:
        try: recs[uid] = hyb.recommend(uid, n=20)
        except Exception: recs[uid] = []
    cnt = train.groupby("user_id").size().to_dict()
    seg = {uid: ("cold_start" if cnt.get(uid, 0) < c["threshold"] else "active") for uid in trel}
    cat = set(train["book_id"].unique()) | set(test["book_id"].unique())
    k10 = h._aggregate(recs, trel, seg, 10, len(cat))
    k20 = h._aggregate(recs, trel, seg, 20, len(cat))
    results[c["name"]] = {"threshold": c["threshold"], "weights": c["weights"], "R@10": k10["global"]["recall"], "nDCG@10": k10["global"]["ndcg"], "cov@10": k10["coverage"], "R@20": k20["global"]["recall"], "cold_n": k10["cold_start"]["n_users"], "active_n": k10["active"]["n_users"]}
    print("{:20s} th={:3d} R@10={:.4f} nDCG={:.4f} cov={:.3f} cold_n={:3d}".format(c["name"], c["threshold"], k10["global"]["recall"], k10["global"]["ndcg"], k10["coverage"], k10["cold_start"]["n_users"]))

Path("reports/sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
best = max(results.items(), key=lambda kv: kv[1]["R@10"])
print()
print("BEST:", best[0], "R@10=", round(best[1]["R@10"], 4))
