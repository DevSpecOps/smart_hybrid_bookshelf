import numpy as np
import pandas as pd


def random_leave_one_out_split(ratings, min_ratings=5, seed=42):
    counts = ratings.groupby("user_id").size()
    eligible = counts[counts >= min_ratings].index
    df = ratings[ratings["user_id"].isin(eligible)].copy()
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, group in df.groupby("user_id"):
        if len(group) < 2:
            continue
        idx = rng.choice(group.index.values, size=1, replace=False)
        test_idx.append(int(idx[0]))
    test = df.loc[test_idx].reset_index(drop=True)
    train = df.drop(test_idx).reset_index(drop=True)
    return train, test


def split_users_for_sweep_eval(eligible_users, seed=42):
    rng = np.random.default_rng(seed + 100)
    arr = np.array(sorted(eligible_users))
    rng.shuffle(arr)
    half = len(arr) // 2
    eval_users = arr[:half].tolist()
    sweep_users = arr[half:].tolist()
    return eval_users, sweep_users
