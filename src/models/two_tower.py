import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    def __init__(self, num_entities, id_dim, feature_dim, hidden_dims, out_dim):
        super().__init__()
        self.id_embed = nn.Embedding(num_entities, id_dim)
        nn.init.normal_(self.id_embed.weight, std=0.05)
        input_dim = id_dim + feature_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, ids, features):
        id_emb = self.id_embed(ids)
        x = torch.cat([id_emb, features], dim=-1)
        x = self.mlp(x)
        return F.normalize(x, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim,
                 id_dim=32, embedding_dim=32, hidden_dims=(128, 64)):
        super().__init__()
        self.user_tower = Tower(num_users, id_dim, user_feature_dim, hidden_dims, embedding_dim)
        self.item_tower = Tower(num_items, id_dim, item_feature_dim, hidden_dims, embedding_dim)

    def forward(self, user_ids, user_features, item_ids, item_features):
        u = self.user_tower(user_ids, user_features)
        i = self.item_tower(item_ids, item_features)
        return u, i


class TwoTowerRecommender:
    def __init__(self, user_emb, item_emb, user_id_to_idx, item_id_to_idx, idx_to_item_id, seen):
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.user_id_to_idx = user_id_to_idx
        self.item_id_to_idx = item_id_to_idx
        self.idx_to_item_id = idx_to_item_id
        self.seen = seen

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            user_emb=data["user_emb"],
            item_emb=data["item_emb"],
            user_id_to_idx=data["user_id_to_idx"],
            item_id_to_idx=data["item_id_to_idx"],
            idx_to_item_id=data["idx_to_item_id"],
            seen=data["seen"],
        )

    def fit(self, ratings):
        self.seen = ratings.groupby("user_id")["book_id"].apply(set).to_dict()
        return self

    def recommend(self, user_id, n=10):
        uid = int(user_id)
        if uid not in self.user_id_to_idx:
            return []
        ui = self.user_id_to_idx[uid]
        scores = self.user_emb[ui] @ self.item_emb.T
        seen = self.seen.get(uid, set())
        seen_idx = [self.item_id_to_idx[int(b)] for b in seen if int(b) in self.item_id_to_idx]
        if seen_idx:
            scores[np.array(seen_idx, dtype=np.int64)] = -np.inf
        k = min(n, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [int(self.idx_to_item_id[int(i)]) for i in top]