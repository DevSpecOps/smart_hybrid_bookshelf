import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader

from src.data.split import random_leave_one_out_split
from src.models.two_tower import TwoTowerModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="data/processed/two_tower.pt")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_device(pref):
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    def __init__(self, ratings, user_id_to_idx, item_id_to_idx, user_features, item_features):
        uidx = ratings["user_id"].map(user_id_to_idx).values
        iidx = ratings["book_id"].map(item_id_to_idx).values
        self.user_ids = torch.tensor(uidx, dtype=torch.long)
        self.item_ids = torch.tensor(iidx, dtype=torch.long)
        self.user_features = torch.tensor(user_features, dtype=torch.float32)
        self.item_features = torch.tensor(item_features, dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        ui = int(self.user_ids[idx].item())
        ii = int(self.item_ids[idx].item())
        return self.user_ids[idx], self.user_features[ui], self.item_ids[idx], self.item_features[ii]


def build_features(users_df, items_df, user_id_to_idx, item_id_to_idx):
    user_cols = ["user_avg_rating", "user_n_ratings", "user_std_rating"]
    item_cols = ["item_avg_rating", "item_n_ratings", "item_std_rating", "item_tag_diversity", "item_popularity_rank"]
    users_idx = users_df.set_index("user_id").reindex(list(user_id_to_idx.keys()))
    items_idx = items_df.set_index("book_id").reindex(list(item_id_to_idx.keys()))
    uf = users_idx[user_cols].fillna(0.0).values.astype(np.float32)
    itf = items_idx[item_cols].fillna(0.0).values.astype(np.float32)
    uf = (uf - uf.mean(0)) / (uf.std(0) + 1e-8)
    itf = (itf - itf.mean(0)) / (itf.std(0) + 1e-8)
    return uf, itf


def train_loop(model, loader, optimizer, device, temperature, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for u_ids, u_feats, i_ids, i_feats in loader:
            u_ids = u_ids.to(device)
            u_feats = u_feats.to(device)
            i_ids = i_ids.to(device)
            i_feats = i_feats.to(device)
            optimizer.zero_grad()
            u_emb, i_emb = model(u_ids, u_feats, i_ids, i_feats)
            logits = (u_emb @ i_emb.T) / temperature
            labels = torch.arange(logits.size(0), device=device)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
            if n_batches % 500 == 0:
                print("  epoch " + str(epoch + 1) + " batch " + str(n_batches) + "/" + str(len(loader)) + " loss=" + str(round(float(loss.item()), 4)))
        print("epoch " + str(epoch + 1) + " done, avg loss=" + str(round(total_loss / max(n_batches, 1), 4)))


def precompute_embeddings(model, num_users, num_items, user_features, item_features, device, batch_size=1024):
    model.eval()
    with torch.no_grad():
        user_embs = []
        for start in range(0, num_users, batch_size):
            end = min(start + batch_size, num_users)
            ids = torch.arange(start, end, dtype=torch.long, device=device)
            feats = torch.tensor(user_features[start:end], dtype=torch.float32, device=device)
            user_embs.append(model.user_tower(ids, feats).cpu().numpy())
        user_embs = np.concatenate(user_embs, axis=0).astype(np.float32)
        item_embs = []
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            ids = torch.arange(start, end, dtype=torch.long, device=device)
            feats = torch.tensor(item_features[start:end], dtype=torch.float32, device=device)
            item_embs.append(model.item_tower(ids, feats).cpu().numpy())
        item_embs = np.concatenate(item_embs, axis=0).astype(np.float32)
    return user_embs, item_embs


def main():
    args = parse_args()
    if Path(args.checkpoint).exists() and not args.force_retrain:
        print("OK: checkpoint exists at " + args.checkpoint)
        print("  use --force-retrain to regenerate")
        return
    set_seeds(args.seed)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tt = cfg["models"]["two_tower"]
    epochs = args.epochs if args.epochs else tt.get("epochs", 3)
    device = get_device(args.device)
    print("device: " + str(device))

    ratings = pd.read_csv("data/raw/ratings.csv")
    users_df = pd.read_csv("data/processed/user_features.csv")
    items_df = pd.read_csv("data/processed/item_features.csv")

    min_r = cfg["data"]["min_user_ratings"]
    train_ratings, _ = random_leave_one_out_split(ratings, min_ratings=min_r, seed=args.seed)
    print("train ratings (test held out): " + str(len(train_ratings)))

    user_id_to_idx = {int(u): i for i, u in enumerate(sorted(ratings["user_id"].unique()))}
    item_id_to_idx = {int(b): i for i, b in enumerate(sorted(ratings["book_id"].unique()))}
    idx_to_item_id = {i: int(b) for b, i in item_id_to_idx.items()}
    print("num users: " + str(len(user_id_to_idx)))
    print("num items: " + str(len(item_id_to_idx)))

    threshold = tt.get("positives_threshold", 4)
    positives = train_ratings[train_ratings["rating"] >= threshold].copy()
    print("positives (rating >= " + str(threshold) + "): " + str(len(positives)))

    if not args.full:
        sample_size = min(int(tt.get("subsample_size", 2000000)), len(positives))
        positives = positives.sample(n=sample_size, random_state=args.seed)
        print("subsample: " + str(len(positives)) + " (use --full for all)")

    user_features, item_features = build_features(users_df, items_df, user_id_to_idx, item_id_to_idx)
    print("user_features: " + str(user_features.shape))
    print("item_features: " + str(item_features.shape))

    model = TwoTowerModel(
        num_users=len(user_id_to_idx),
        num_items=len(item_id_to_idx),
        user_feature_dim=user_features.shape[1],
        item_feature_dim=item_features.shape[1],
        id_dim=int(tt.get("id_dim", 32)),
        embedding_dim=int(tt.get("embedding_dim", 32)),
        hidden_dims=tuple(tt.get("hidden_dims", [128, 64])),
    ).to(device)
    print("model params: " + str(sum(p.numel() for p in model.parameters())))

    dataset = PairDataset(positives, user_id_to_idx, item_id_to_idx, user_features, item_features)
    loader = DataLoader(
        dataset,
        batch_size=int(tt.get("batch_size", 1024)),
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    print("batches per epoch: " + str(len(loader)))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(tt.get("lr", 0.001)))
    train_loop(model, loader, optimizer, device, float(tt.get("temperature", 0.07)), epochs)

    print("precomputing embeddings...")
    user_emb, item_emb = precompute_embeddings(
        model, len(user_id_to_idx), len(item_id_to_idx),
        user_features, item_features, device,
    )

    seen = ratings.groupby("user_id")["book_id"].apply(set).to_dict()

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "user_emb": user_emb,
        "item_emb": item_emb,
        "user_id_to_idx": user_id_to_idx,
        "item_id_to_idx": item_id_to_idx,
        "idx_to_item_id": idx_to_item_id,
        "seen": seen,
        "config": tt,
    }, args.checkpoint)
    print("saved checkpoint: " + args.checkpoint)
    print("user_emb: " + str(user_emb.shape))
    print("item_emb: " + str(item_emb.shape))


if __name__ == "__main__":
    main()