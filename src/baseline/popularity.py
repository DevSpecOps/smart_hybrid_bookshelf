import pandas as pd


class PopularityRecommender:
    def __init__(self, min_ratings=50, score="bayesian"):
        self.min_ratings = min_ratings
        self.score = score
        self.ranking = None
        self.stats = None
        self.seen = {}

    def fit(self, ratings):
        agg = ratings.groupby("book_id")["rating"].agg(["mean", "count"])
        agg = agg[agg["count"] >= self.min_ratings].copy()

        if self.score == "bayesian":
            C = agg["count"].mean()
            m = agg["mean"].mean()
            agg["score"] = (agg["count"] * agg["mean"] + C * m) / (agg["count"] + C)
        elif self.score == "mean":
            agg["score"] = agg["mean"]
        elif self.score == "count":
            agg["score"] = agg["count"]
        else:
            raise ValueError(f"unknown score: {self.score}")

        self.stats = agg
        self.ranking = agg.sort_values("score", ascending=False).index.to_numpy()
        self.seen = ratings.groupby("user_id")["book_id"].apply(set).to_dict()
        return self

    def recommend(self, user_id, n=10):
        return [b for b, _ in self.recommend_with_scores(user_id, n)]

    def recommend_with_scores(self, user_id, n=10):
        seen = self.seen.get(user_id, set())
        scores = self.stats["score"]
        out = []
        for b in self.ranking:
            if b not in seen:
                out.append((int(b), float(scores.loc[b])))
                if len(out) >= n:
                    break
        return out

    def recommend_cold(self, n=10):
        return [int(b) for b in self.ranking[:n]]

    def coverage(self, n=10):
        catalog = set(self.ranking.tolist())
        if not catalog:
            return 0.0
        exposed = set()
        for user_id in self.seen:
            exposed.update(self.recommend(user_id, n))
        return len(exposed) / len(catalog)
