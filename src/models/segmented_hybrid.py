class SegmentedHybrid:
    is_orchestrator = True

    def __init__(self, models, weights, cold_threshold, k_rrf=60, candidates_per_model=50):
        self.models = models
        self.weights = weights
        self.cold_threshold = cold_threshold
        self.k_rrf = k_rrf
        self.candidates_per_model = candidates_per_model
        self.seen = {}

    def fit(self, ratings):
        self.seen = ratings.groupby("user_id")["book_id"].apply(set).to_dict()
        return self

    def recommend(self, user_id, n=10):
        history = self.seen.get(user_id, set())
        segment = "cold_start" if len(history) < self.cold_threshold else "active"
        weights = self.weights.get(segment, {})
        active = {name: w for name, w in weights.items() if w > 0 and name in self.models}
        if not active:
            return []

        rank_maps = {}
        for name in active:
            try:
                recs = self.models[name].recommend(user_id, n=self.candidates_per_model)
            except Exception:
                recs = []
            rank_maps[name] = {int(b): r for r, b in enumerate(recs)}

        scores = {}
        for name, rm in rank_maps.items():
            w = active[name]
            for item, rank in rm.items():
                if item in history:
                    continue
                contribution = w / (self.k_rrf + rank + 1)
                scores[item] = scores.get(item, 0.0) + contribution

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [int(b) for b, _ in ranked[:n]]
