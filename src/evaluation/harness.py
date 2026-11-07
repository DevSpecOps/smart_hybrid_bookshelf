import numpy as np

from src.baseline.metrics import precision_at_k, recall_at_k, ndcg_at_k


class EvaluationHarness:
    def __init__(self, models, k_values, cold_threshold):
        self.models = models
        self.k_values = k_values
        self.cold_threshold = cold_threshold

    def evaluate(self, train_df, test_df):
        train_counts = train_df.groupby("user_id").size().to_dict()
        test_relevant = (
            test_df.groupby("user_id")["book_id"].apply(set).to_dict()
        )
        catalog = set(train_df["book_id"].unique()) | set(test_df["book_id"].unique())
        total_items = len(catalog)

        segments = {}
        for uid in test_relevant:
            n = train_counts.get(uid, 0)
            segments[uid] = "cold_start" if n < self.cold_threshold else "active"

        max_k = max(self.k_values)
        results = {}
        for name, adapter in self.models.items():
            try:
                adapter.fit(train_df)
            except Exception as e:
                print("WARN " + name + " fit failed: " + type(e).__name__ + ": " + str(e))
                continue
            recs_by_user = {}
            for uid in test_relevant:
                try:
                    recs_by_user[uid] = adapter.recommend(uid, n=max_k)
                except Exception as e:
                    recs_by_user[uid] = []
            model_result = {}
            for k in self.k_values:
                model_result["k_" + str(k)] = self._aggregate(
                    recs_by_user, test_relevant, segments, k, total_items
                )
            results[name] = model_result
        return results

    def _aggregate(self, recs_by_user, test_relevant, segments, k, total_items):
        buckets = {"global": [], "cold_start": [], "active": []}
        recommended_items = set()
        for uid, recs in recs_by_user.items():
            rel = test_relevant[uid]
            top = recs[:k]
            entry = {
                "p": precision_at_k(recs, rel, k),
                "r": recall_at_k(recs, rel, k),
                "n": ndcg_at_k(recs, rel, k),
            }
            buckets["global"].append(entry)
            buckets[segments[uid]].append(entry)
            for i in top:
                recommended_items.add(int(i))
        coverage = len(recommended_items) / total_items if total_items else 0.0
        out = {"coverage": float(coverage)}
        for bucket, entries in buckets.items():
            if not entries:
                out[bucket] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "ndcg": 0.0,
                    "n_users": 0,
                }
                continue
            out[bucket] = {
                "precision": float(np.mean([e["p"] for e in entries])),
                "recall": float(np.mean([e["r"] for e in entries])),
                "ndcg": float(np.mean([e["n"] for e in entries])),
                "n_users": len(entries),
            }
        return out
