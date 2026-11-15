import numpy as np
import pandas as pd

from src.baseline.metrics import precision_at_k, recall_at_k, ndcg_at_k
from src.data.split import random_leave_one_out_split, split_users_for_sweep_eval


def test_metrics_basic():
    recs = [1, 2, 3, 4, 5]
    rel = {1, 3, 5}
    assert abs(precision_at_k(recs, rel, 5) - 0.6) < 1e-9
    assert abs(recall_at_k(recs, rel, 5) - 1.0) < 1e-9
    assert 0.0 < ndcg_at_k(recs, rel, 5) <= 1.0


def test_metrics_edge_cases():
    assert precision_at_k([], {1, 2}, 5) == 0.0
    assert recall_at_k([], {1, 2}, 5) == 0.0
    assert ndcg_at_k([], {1, 2}, 5) == 0.0
    assert precision_at_k([1, 2, 3], set(), 5) == 0.0


def test_split_deterministic():
    df = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "book_id": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "rating": [5, 4, 3, 5, 4, 3, 3, 4, 5, 4],
    })
    tr1, te1 = random_leave_one_out_split(df, min_ratings=5, seed=42)
    tr2, te2 = random_leave_one_out_split(df, min_ratings=5, seed=42)
    assert (tr1 == tr2).all().all()
    assert (te1 == te2).all().all()
    assert te1.groupby("user_id").size().max() == 1
    assert set(tr1.user_id) == set(te1.user_id)


def test_split_sweep_eval_disjoint():
    users = list(range(1, 101))
    eval_users, sweep_users = split_users_for_sweep_eval(users, seed=42)
    assert set(eval_users).isdisjoint(set(sweep_users))
    assert set(eval_users) | set(sweep_users) == set(users)
    assert len(eval_users) + len(sweep_users) == 100


def test_split_sweep_eval_deterministic():
    users = list(range(1, 101))
    e1, s1 = split_users_for_sweep_eval(users, seed=42)
    e2, s2 = split_users_for_sweep_eval(users, seed=42)
    assert e1 == e2
    assert s1 == s2
