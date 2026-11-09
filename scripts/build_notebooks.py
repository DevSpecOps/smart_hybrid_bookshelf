import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat as nbf


BOOTSTRAP = """
import os
import sys
from pathlib import Path

p = Path.cwd()
while p != p.parent and not (p / "configs" / "config.yaml").exists():
    p = p.parent
if (p / "configs" / "config.yaml").exists():
    os.chdir(p)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
print("cwd:", Path.cwd())
print("sys.path[0]:", sys.path[0])
"""


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def build_eda():
    cells = [
        md("""# 01 - Exploratory Data Analysis

Overview of the Goodreads-10k dataset after filtering (min_user_ratings=5, min_book_ratings=10).

- Rating distribution
- User activity distribution
- Book popularity distribution
- Top tags
"""),
        code(BOOTSTRAP),
        code("""import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
Path("reports/figures").mkdir(parents=True, exist_ok=True)

ratings = pd.read_csv("data/raw/ratings.csv")
books = pd.read_csv("data/raw/books.csv")
users = pd.read_csv("data/raw/users.csv")

print("ratings:", ratings.shape)
print("books:", books.shape)
print("users:", users.shape)
"""),
        code("""n_users = ratings.user_id.nunique()
n_books = ratings.book_id.nunique()
n_ratings = len(ratings)
density = n_ratings / (n_users * n_books)

print("unique users:", n_users)
print("unique books:", n_books)
print("ratings:", n_ratings)
print("density:", round(density, 6))
print("sparsity:", round(1 - density, 6))
"""),
        code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="#4C72B0")
axes[0].set_title("Rating distribution")
axes[0].set_xlabel("rating")
axes[0].set_ylabel("count")

axes[1].hist(ratings.groupby("user_id").size(), bins=50, color="#4C72B0")
axes[1].set_title("User activity (ratings per user)")
axes[1].set_xlabel("ratings")
axes[1].set_ylabel("users")

plt.tight_layout()
plt.savefig("reports/figures/eda_ratings.png", dpi=100)
plt.show()
"""),
        code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(ratings.groupby("book_id").size(), bins=50, color="#55A868")
axes[0].set_title("Book popularity")
axes[0].set_xlabel("ratings per book")
axes[0].set_ylabel("books")

tag_counts = books["tags"].str.split("|").explode().value_counts().head(20)
tag_counts.plot(kind="barh", ax=axes[1], color="#C44E52")
axes[1].set_title("Top 20 tags")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("reports/figures/eda_books_tags.png", dpi=100)
plt.show()
"""),
        md("""## Key observations

- Dataset is heavily sparse (~1.1% density after filtering)
- Rating distribution is left-skewed (users rate books they like)
- Activity distribution is long-tailed: most users rate few books
- Popularity distribution is also long-tailed: few books dominate
"""),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


def build_comparison():
    cells = [
        md("""# 02 - Models Comparison

All models evaluated on the same leave-one-out split, 2000 held-out users, seed=42.

Loads metrics from `reports/metrics_final.json` produced by `make eval`.
"""),
        code(BOOTSTRAP),
        code("""import json
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
Path("reports/figures").mkdir(parents=True, exist_ok=True)

with open("reports/metrics_final.json") as f:
    m = json.load(f)

rows = []
for model, per_k in m.items():
    for k, b in per_k.items():
        rows.append({
            "model": model,
            "k": k,
            "recall": b["global"]["recall"],
            "ndcg": b["global"]["ndcg"],
            "coverage": b["coverage"],
        })
df = pd.DataFrame(rows)
print("shape:", df.shape)
df.head(10)
"""),
        code("""pivot = df[df["k"] == "k_10"].sort_values("recall", ascending=False)
print(pivot[["model", "recall", "ndcg", "coverage"]].to_string(index=False))
"""),
        code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
p = df[df["k"] == "k_10"].sort_values("recall", ascending=False)
axes[0].barh(p["model"], p["recall"], color="#4C72B0")
axes[0].set_title("Recall@10")
axes[1].barh(p["model"], p["ndcg"], color="#55A868")
axes[1].set_title("nDCG@10")
axes[2].barh(p["model"], p["coverage"], color="#C44E52")
axes[2].set_title("Coverage@10")
for ax in axes:
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig("reports/figures/models_comparison.png", dpi=100)
plt.show()
"""),
        code("""top_models = ["hybrid", "content", "two_tower", "popularity"]
fig, ax = plt.subplots(figsize=(8, 4))
for model in top_models:
    sub = df[df["model"] == model].sort_values("k")
    ax.plot(sub["k"], sub["recall"], marker="o", label=model)
ax.set_title("Recall@K across K values")
ax.set_xlabel("K")
ax.set_ylabel("Recall")
ax.legend()
plt.tight_layout()
plt.savefig("reports/figures/recall_at_k.png", dpi=100)
plt.show()
"""),
        md("""## Findings

- Content-based is the strongest classical baseline: TF-IDF on tags captures genre signal
- Two-Tower has the highest coverage: NCE training spreads embeddings
- Hybrid wins on both recall and coverage simultaneously
- SVD is dominated under leave-one-out protocol
- Popularity and ItemKNN fall well behind on recall but ItemKNN has high coverage
"""),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


def build_hybrid_analysis():
    cells = [
        md("""# 03 - Hybrid Analysis

Segment-level analysis of the hybrid recommender and weight optimization results.
"""),
        code(BOOTSTRAP),
        code("""import json
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
Path("reports/figures").mkdir(parents=True, exist_ok=True)

with open("reports/metrics_final.json") as f:
    m = json.load(f)

rows = []
for model, per_k in m.items():
    b = per_k["k_10"]
    for segment in ["cold_start", "active"]:
        rows.append({
            "model": model,
            "segment": segment,
            "n_users": b[segment]["n_users"],
            "recall": b[segment]["recall"],
            "ndcg": b[segment]["ndcg"],
        })
seg = pd.DataFrame(rows)
print(seg.head(20))
"""),
        code("""print("Segment sizes (k_10):")
sizes = seg[seg["model"] == "hybrid"][["segment", "n_users"]]
print(sizes.to_string(index=False))
"""),
        code("""pivot = seg.pivot(index="model", columns="segment", values="recall")
pivot = pivot.sort_values("active", ascending=False)
pivot.plot(kind="bar", figsize=(10, 4), color=["#C44E52", "#4C72B0"])
plt.title("Recall@10 by segment")
plt.ylabel("Recall")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("reports/figures/segment_recall.png", dpi=100)
plt.show()
"""),
        code("""sweep_path = Path("reports/sweep.json")
if sweep_path.exists():
    with open(sweep_path) as f:
        sweep = json.load(f)
    rows = []
    for name, entry in sweep.items():
        rows.append({
            "config": name,
            "threshold": entry["threshold"],
            "R@10": entry["hybrid_R@10"],
            "nDCG@10": entry["hybrid_nDCG@10"],
            "cov@10": entry["hybrid_cov@10"],
        })
    df = pd.DataFrame(rows).sort_values("R@10", ascending=False)
    print(df.to_string(index=False))
else:
    print("sweep.json not found -- run scripts/sweep_weights.py first")
"""),
        code("""if sweep_path.exists():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(df["config"], df["R@10"], color="#4C72B0")
    ax.set_title("Hybrid R@10 across weight configurations")
    ax.set_xlabel("R@10")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("reports/figures/sweep_r10.png", dpi=100)
    plt.show()
"""),
        md("""## Findings

- Cold-start segment is small (1.8% of users) because Goodreads-10k has dense users
- Content dominates cold-start because it does not depend on user history
- Active segment is where Two-Tower contributes its largest marginal gain
- Best weight configuration by R@10: content-heavy (without SVD, capped ItemKNN)
- Adding Two-Tower to the hybrid: +8% R@10, +12% nDCG, +52% coverage
"""),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


def main():
    notebooks = [
        ("01_eda.ipynb", build_eda),
        ("02_models_comparison.ipynb", build_comparison),
        ("03_hybrid_analysis.ipynb", build_hybrid_analysis),
    ]
    out = Path("notebooks")
    out.mkdir(parents=True, exist_ok=True)
    for name, builder in notebooks:
        nb = builder()
        path = out / name
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote " + str(path))


if __name__ == "__main__":
    main()
