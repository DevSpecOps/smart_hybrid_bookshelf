import numpy as np


def _as_set(x):
    return set(x)


def precision_at_k(recommended, relevant, k):
    rec = list(recommended)[:k]
    if k <= 0 or not rec:
        return 0.0
    return len(_as_set(rec) & _as_set(relevant)) / k


def recall_at_k(recommended, relevant, k):
    rec = list(recommended)[:k]
    rel = _as_set(relevant)
    if not rel:
        return 0.0
    return len(_as_set(rec) & rel) / len(rel)


def dcg_at_k(recommended, relevant, k):
    rec = list(recommended)[:k]
    rel = _as_set(relevant)
    return sum(1.0 / np.log2(i + 2) for i, item in enumerate(rec) if item in rel)


def idcg_at_k(relevant, k):
    n = min(len(_as_set(relevant)), k)
    return sum(1.0 / np.log2(i + 2) for i in range(n))


def ndcg_at_k(recommended, relevant, k):
    ideal = idcg_at_k(relevant, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(recommended, relevant, k) / ideal
