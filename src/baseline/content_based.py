import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:
    def __init__(self, top_tags_per_book=20, n_neighbors=100, min_df=2, max_df=0.9):
        self.top_tags_per_book = top_tags_per_book
        self.n_neighbors = n_neighbors
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.tfidf = None
        self.book_ids = None
        self.neighbors = None
        self.seen = {}

    def _build_profiles(self, books, book_tags, tags):
        indexed = books.set_index("book_id")
        title_by_local = indexed["original_title"].fillna(indexed["title"]).to_dict()
        local_to_global = dict(zip(books["book_id"], books["goodreads_book_id"]))

        merged = book_tags.merge(tags, on="tag_id")
        merged = merged.sort_values(["goodreads_book_id", "count"], ascending=[True, False])
        top_tags = merged.groupby("goodreads_book_id").head(self.top_tags_per_book)
        tag_text = top_tags.groupby("goodreads_book_id")["tag_name"].apply(
            lambda s: " ".join(s.astype(str))
        ).to_dict()

        profiles = {}
        for local_id, global_id in local_to_global.items():
            title = title_by_local.get(local_id, "") or ""
            tags_str = tag_text.get(global_id, "") or ""
            profiles[int(local_id)] = f"{title} {tags_str}".strip()
        return profiles

    def fit(self, books, book_tags, tags, ratings):
        profiles = self._build_profiles(books, book_tags, tags)
        self.book_ids = np.array(sorted(profiles.keys()), dtype=np.int64)
        docs = [profiles[int(i)] for i in self.book_ids]

        self.vectorizer = TfidfVectorizer(
            stop_words="english", min_df=self.min_df, max_df=self.max_df,
            lowercase=True, ngram_range=(1, 2),
        )
        self.tfidf = csr_matrix(self.vectorizer.fit_transform(docs))

        self.neighbors = self._compute_top_neighbors(self.n_neighbors)
        self.seen = ratings.groupby("user_id")["book_id"].apply(set).to_dict()
        return self

    def _compute_top_neighbors(self, k):
        n = self.tfidf.shape[0]
        k = min(k, n - 1)
        chunk = 1000
        result = {}
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sims = linear_kernel(self.tfidf[start:end], self.tfidf)
            for local_i in range(end - start):
                i = start + local_i
                row = sims[local_i]
                row[i] = -np.inf
                top = np.argpartition(-row, k)[:k]
                top = top[np.argsort(-row[top])]
                result[int(self.book_ids[i])] = [
                    (int(self.book_ids[j]), float(row[j])) for j in top if row[j] > 0
                ]
        return result

    def get_similar_books(self, book_id, N=5):
        return [bid for bid, _ in self.neighbors.get(int(book_id), [])[:N]]

    def recommend(self, user_id, n=10):
        seen = self.seen.get(user_id, set())
        if not seen:
            return []
        scores = {}
        for rated in seen:
            for neigh, sim in self.neighbors.get(int(rated), []):
                if neigh in seen:
                    continue
                scores[neigh] = scores.get(neigh, 0.0) + sim
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [bid for bid, _ in ranked[:n]]

    def coverage(self, n=10):
        catalog = set(int(b) for b in self.book_ids.tolist())
        if not catalog:
            return 0.0
        exposed = set()
        for user_id in self.seen:
            exposed.update(self.recommend(user_id, n))
        return len(exposed) / len(catalog)
