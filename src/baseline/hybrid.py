import numpy as np


class HybridRecommender:
    def __init__(self, content_model, cf_model, svd_model, popularity_model,
                 train_counts=None, warm_threshold=50, k_rrf=60,
                 weights=None):
        self.content = content_model
        self.cf = cf_model
        self.svd = svd_model
        self.popularity = popularity_model
        self.train_counts = train_counts or {}
        self.warm_threshold = warm_threshold
        self.k_rrf = k_rrf
        self.weights = weights or {"content": 1.0, "cf": 1.0, "svd": 0.2}
        self.seen = {}

    def fit(self, ratings):
        self.seen = ratings.groupby("user_id")["book_id"].apply(set).to_dict()
        counts = ratings.groupby("book_id").size().to_dict()
        self.train_counts = counts
        return self

    def _rank_map(self, recs):
        return {int(b): i for i, b in enumerate(recs)}

    def recommend(self, user_id, n=10):
        seen = self.seen.get(user_id, set())

        if not seen:
            return self.popularity.recommend_cold(n)

        cand_content = self.content.recommend(user_id, max(n * 5, 50))
        cand_cf = self.cf.recommend(user_id, max(n * 5, 50))
        cand_svd = self.svd.get_recommendations(user_id, N=max(n * 5, 50))

        rank_content = self._rank_map(cand_content)
        rank_cf = self._rank_map(cand_cf)
        rank_svd = self._rank_map(cand_svd)

        candidates = set(cand_content) | set(cand_cf) | set(cand_svd)
        candidates -= seen

        scores = {}
        for b in candidates:
            is_warm = self.train_counts.get(b, 0) >= self.warm_threshold
            rrf = 0.0
            if b in rank_content:
                rrf += self.weights["content"] / (self.k_rrf + rank_content[b] + 1)
            if b in rank_cf:
                rrf += self.weights["cf"] / (self.k_rrf + rank_cf[b] + 1)
            if b in rank_svd and is_warm:
                rrf += self.weights["svd"] / (self.k_rrf + rank_svd[b] + 1)
            scores[b] = rrf

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [int(b) for b, _ in ranked[:n]]

    def coverage(self, n=10, max_users=500):
        exposed = set()
        sample = list(self.seen.keys())[:max_users]
        for uid in sample:
            exposed.update(self.recommend(int(uid), n))
        return len(exposed) / 10000.0
