# Smart Hybrid Bookshelf

A hybrid book recommendation system that combines classical approaches (popularity, content-based, item-based collaborative filtering, matrix factorization) with a neural Two-Tower retrieval model. The project extends the homework baseline `smart_bookshelf` with segment-aware hybridization and an end-to-end reproducible pipeline.

## Table of Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Reproducing the Results](#reproducing-the-results)
- [Models](#models)
- [Hybrid Strategy](#hybrid-strategy)
- [Evaluation](#evaluation)
- [Results](#results)
- [Configuration](#configuration)
- [License](#license)

## Motivation

The baseline `smart_bookshelf` demonstrated that a hybrid of content-based, collaborative, and SVD signals improves ranking quality over any single model. However, it relied on a random per-user split and a single fusion strategy. This project addresses three limitations:

- **Cold-start users** are not handled explicitly; the hybrid fuses all models uniformly, which can amplify noise for users with very few interactions.
- **No neural component** is used, so side features (genres, authors) cannot be combined with collaborative signals in a learned embedding space.
- **No end-to-end pipeline** exists to reproduce the full flow from raw Goodreads data to final ranked recommendations.

We introduce segment-aware weighting, a Two-Tower model trained with negative sampling, and a reproducible `make reproduce` target, then evaluate all models under a leave-one-out protocol.

## Dataset

The project uses **Goodreads-10k** (the `zygmuntz/goodbooks-10k` release), the same dataset used in the homework baseline `smart_bookshelf`. It contains roughly 6 million ratings across 10,000 books and 53,000 users.

The downloader (`scripts/download_data.py`) pulls the official CSVs directly from the dataset's GitHub releases, so no Kaggle account or API token is required. Files are placed in `archive/` which is gitignored.

| File | Rows | Columns |
|---|---|---|
| `books.csv` | 10,000 | `book_id`, `goodreads_book_id`, `title`, `original_title`, ... |
| `ratings.csv` | ~6,000,000 | `user_id`, `book_id`, `rating` |
| `tags.csv` | ~34,000 | `tag_id`, `tag_name` |
| `book_tags.csv` | ~1,000,000 | `goodreads_book_id`, `tag_id`, `count` |
| `to_read.csv` | (not used) | `user_id`, `book_id` |

The canonical flat schema in `data/raw/` is produced by `src.data.load_real` and preserves the source schema so that vendored baseline models can be used without transformation.

## Architecture

~~~
Goodreads CSV (archive/) -> load_real -> data/raw/ -> build_features -> data/processed/
                                                                |
                                                                v
                    Candidate generators (popularity, content, itemknn, svd, two-tower)
                                                                |
                                                                v
                                        Segment-aware weighted RRF hybrid ranker
                                                                |
                                                                v
                                                     Ranked recommendations
~~~

Pipeline stages:

1. **Dataset acquisition** - `scripts/download_data.py`
2. **Canonical loading** - `src/data/load_real.py`
3. **Feature engineering** - `src/data/build_features.py`
4. **Model training** - classical (`src/baseline/`) and neural (`src/models/two_tower.py`)
5. **Candidate generation** - top-N per model
6. **Hybrid ranking** - segment-aware weighted Reciprocal Rank Fusion (`src/models/segmented_hybrid.py`)
7. **Evaluation** - leave-one-out protocol with Precision@K / Recall@K / nDCG@K (`src/evaluation/`)
8. **Reporting** - JSON metrics and comparative plots

## Project Structure

~~~
.
|-- archive/                 # Raw Goodreads CSVs (gitignored, downloaded)
|-- configs/                 # YAML experiment configs
|-- data/
|   |-- raw/                 # Canonical schema (same as source)
|   |-- interim/
|   `-- processed/           # Feature tables
|-- notebooks/               # Exploratory and reporting notebooks
|-- reports/                 # Auto-generated metrics and figures
|-- scripts/
|   `-- download_data.py     # Goodreads-10k downloader (GitHub releases)
|-- src/
|   |-- baseline/            # HW baseline models (vendored from smart_bookshelf)
|   |-- data/                # Canonical loader, features, splits
|   |-- models/              # Adapters, hybrid, neural
|   |-- evaluation/          # Metrics and evaluation harness
|   `-- pipeline.py          # End-to-end orchestration
|-- tests/                   # Smoke tests
|-- Makefile
|-- requirements.txt
`-- README.md
~~~

## Getting Started

### Prerequisites

- Python 3.10+
- (optional) CUDA-capable GPU for neural training
- Internet access for the initial dataset download

### Installation

~~~bash
git clone https://github.com/DevSpecOps/smart_hybrid_bookshelf.git
cd smart_hybrid_bookshelf
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
~~~

## Reproducing the Results

~~~bash
python scripts/download_data.py
python -m src.data.load_real --archive archive --output data/raw --config configs/config.yaml
python -m src.data.build_features --raw data/raw --output data/processed
python -m src.evaluate_baseline --config configs/config.yaml --output reports/metrics_baseline.json
~~~

On Unix, `make reproduce` runs all steps. On Windows, run the Python commands directly or use the Makefile under WSL.

## Models

| Model | Family | Strength | Weakness |
|---|---|---|---|
| Popularity (Bayesian) | Non-personalized | Robust to cold start | No personalization |
| Content-based (TF-IDF) | Item-item | Works with no interactions | Limited by metadata |
| ItemKNN | Collaborative | Strong intra-category signal | Popularity bias |
| SVD | Latent factor | Captures co-occurrence | No side features |
| Two-Tower (neural) | Retrieval | Side-feature aware, scalable | Needs enough interactions |

## Hybrid Strategy

Users are segmented by interaction count:

- **Cold start** (< 5 ratings): weighted Popularity + Content.
- **Active** (>= 5 ratings): weighted SVD + ItemKNN + Two-Tower + Popularity.

Scores are fused using **weighted Reciprocal Rank Fusion (RRF)**, which is rank-based and therefore scale-invariant across models. Already-rated items are filtered after ranking. The RRF constant `k_rrf` and all segment weights are configurable in `configs/config.yaml`.

## Evaluation

The protocol is **leave-one-out per user**: for each user with at least 5 ratings, one interaction is held out at random (seeded). We report Precision@K, Recall@K, and nDCG@K for K in {5, 10, 20}, aggregated globally and per user segment, plus catalog coverage@K.

## Results

Run `make reproduce` to populate this section. The baseline `smart_bookshelf` reported hybrid RRF recall@10 = 0.1667 on the same dataset, versus 0.1432 for content-based and 0.0739 for ItemKNN. This project aims to improve further with neural signals for active users.

## Baseline Provenance

The vendored baseline models in `src/baseline/` are copied verbatim from the homework repository `smart_bookshelf` at commit `
d09071a2dfbfd674d78d8d0178771bf09d9b6f2a
`. See `src/baseline/VENDORED.md` for file-level SHA-256 verification.

## Configuration

All hyperparameters live in `configs/config.yaml`. Weights, model sizes, RRF constant, and evaluation settings can be changed without touching code.

## License

MIT
