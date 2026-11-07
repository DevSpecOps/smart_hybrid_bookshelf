import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCF:
    def __init__(self, n_neighbors=100, k_predict=20, min_ratings_per_item=10):
        self.n_neighbors = n_neighbors
        self.k_predict = k_predict
        self.min_ratings_per_item = min_ratings_per_item
        self.user_ids = None
        self.item_ids = None
        self.user_index = None
        self.item_index = None
        self.matrix = None
        self.item_means = None
        self.neighbors = None
        self.user_rated = None

    def fit(self, ratings):
        self.user_ids = np.sort(ratings["user_id"].unique())
        self.item_ids = np.sort(ratings["book_id"].unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {b: i for i, b in enumerate(self.item_ids)}
        rows = ratings["user_id"].map(self.user_index).to_numpy()
        cols = ratings["book_id"].map(self.item_index).to_numpy()
        vals = ratings["rating"].to_numpy(dtype=np.float64)
        self.matrix = csr_matrix((vals, (rows, cols)),
                                 shape=(len(self.user_ids), len(self.item_ids)))
        counts = np.asarray((self.matrix > 0).sum(axis=0)).ravel()
        sums = np.asarray(self.matrix.sum(axis=0)).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            self.item_means = np.where(counts > 0, sums / counts, 0.0)
        self.user_rated = [
            set(np.nonzero(self.matrix.getrow(i))[1].tolist())
            for i in range(self.matrix.shape[0])
        ]
        self.neighbors = self._compute_top_neighbors()
        return self

    def _compute_top_neighbors(self):
        item_counts = np.asarray((self.matrix > 0).sum(axis=0)).ravel()
        k = min(self.n_neighbors, self.matrix.shape[1] - 1)
        chunk = 500
        result = {}
        n_items = self.matrix.shape[1]
        for start in range(0, n_items, chunk):
            end = min(start + chunk, n_items)
            sims = cosine_similarity(self.matrix[:, start:end].T, self.matrix.T)
            for local_i in range(end - start):
                i = start + local_i
                if item_counts[i] < self.min_ratings_per_item:
                    result[i] = []
                    continue
                row = sims[local_i]
                row[i] = -np.inf
                top = np.argpartition(-row, k)[:k]
                top = top[np.argsort(-row[top])]
                result[i] = [(int(j), float(row[j])) for j in top
                             if row[j] > 0 and item_counts[j] >= self.min_ratings_per_item]
        return result

    def predict(self, user_id, book_id):
        if user_id not in self.user_index or book_id not in self.item_index:
            return float(self.item_means[self.item_index.get(book_id, 0)])
        ui = self.user_index[user_id]
        bi = self.item_index[book_id]
        rated = self.user_rated[ui]
        num, den = 0.0, 0.0
        for j, sim in self.neighbors.get(bi, [])[:self.k_predict]:
            if j in rated:
                num += sim * self.matrix[ui, j]
                den += abs(sim)
        return float(num / den) if den else float(self.item_means[bi])

    def recommend(self, user_id, n=10):
        if user_id not in self.user_index:
            return []
        ui = self.user_index[user_id]
        rated = self.user_rated[ui]
        num = {}
        den = {}
        for rated_bi in rated:
            rating = self.matrix[ui, rated_bi]
            for neigh_bi, sim in self.neighbors.get(rated_bi, [])[:self.k_predict]:
                if neigh_bi in rated:
                    continue
                num[neigh_bi] = num.get(neigh_bi, 0.0) + sim * rating
                den[neigh_bi] = den.get(neigh_bi, 0.0) + abs(sim)
        if not num:
            return []
        scores = {bi: num[bi] / den[bi] for bi in num if den[bi] > 0}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [int(self.item_ids[bi]) for bi, _ in ranked[:n]]

    def coverage(self, n=10, max_users=500):
        exposed = set()
        sample = self.user_ids[:max_users]
        for uid in sample:
            exposed.update(self.recommend(int(uid), n))
        return len(exposed) / len(self.item_ids)
