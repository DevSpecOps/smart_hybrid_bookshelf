class ModelAdapter:
    def __init__(self, name, model, fit_extra=None):
        self.name = name
        self.model = model
        self.fit_extra = tuple(fit_extra) if fit_extra else ()

    def fit(self, ratings):
        if self.fit_extra:
            self.model.fit(*self.fit_extra, ratings)
        else:
            self.model.fit(ratings)
        return self

    def recommend(self, user_id, n=10):
        raw = self.model.recommend(user_id, n=n)
        out = []
        seen = set()
        for x in raw:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if i in seen:
                continue
            seen.add(i)
            out.append(i)
            if len(out) >= n:
                break
        return out
