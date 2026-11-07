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
