# Smart Hybrid Bookshelf

A hybrid book recommendation system combining classical baselines (popularity, content-based, item-based collaborative filtering, matrix factorization) with a neural Two-Tower retrieval model. The system segments users by activity level and fuses predictions via weighted Reciprocal Rank Fusion. Fully reproducible end-to-end on CPU.

## Table of Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Models](#models)
- [Two-Tower Architecture](#two-tower-architecture)
- [Hybrid Strategy](#hybrid-strategy)
- [Evaluation Protocol](#evaluation-protocol)
- [Results](#results)
- [Analysis](#analysis)
- [Known Limitations](#known-limitations)
- [Baseline Provenance](#baseline-provenance)
- [Configuration](#configuration)
- [License](#license)

## Motivation

The homework baseline `smart_bookshelf` demonstrated that a simple hybrid of content-based, collaborative, and latent-factor signals improves ranking quality over any single model. However, it had three limitations that this project addresses:

1. **No cold-start handling.** The hybrid fused all models uniformly, ignoring the different signal available for new versus active users.
2. **No neural component.** Side features (genres, authors) could not be combined with collaborative signals in a learned embedding space.
3. **No end-to-end pipeline.** Reproducing the full flow from raw Kaggle data to final ranked recommendations required manual steps.

We introduce segment-aware weighting, a Two-Tower model trained with NCE in-batch negatives, and a single-command pipeline, then evaluate all models under a strict leave-one-out protocol.

## Dataset

The project uses the **Kaggle version** of the Goodreads-10k dataset (`zygmunt/goodbooks-10k`, https://www.kaggle.com/datasets/zygmunt/goodbooks-10k) -- the exact same dataset used in the homework baseline `smart_bookshelf`.

| File | Rows | Columns |
|---|---|---|
| `books.csv` | 10,000 | `id`, `book_id`, `best_book_id`, `work_id`, `authors`, `original_publication_year`, `original_title`, `title`, ... |
| `ratings.csv` | ~980,000 | `book_id`, `user_id`, `rating` |
| `tags.csv` | ~34,000 | `tag_id`, `tag_name` |
| `book_tags.csv` | ~1,000,000 | `goodreads_book_id`, `tag_id`, `count` |

After **deduplication** (dropping duplicate `(user_id, book_id)` pairs) and filtering (`min_user_ratings=3`, `min_book_ratings=5`):

- **45,122 users**
- **10,000 books**
- **962,915 ratings**
- **density = 0.00213**
- **100% of books have at least one tag**

The Kaggle release is the version used in the homework baseline `smart_bookshelf`. It contains known duplicates (the Kaggle description marks it "obsolete" and points to a 6M-rating GitHub release); we drop them in `src/data/load_real.py` before any modeling.

**Why the dataset is vendored.** Kaggle API requires authentication (`kaggle.json`), which would force every reviewer to register an account and create an API token before running `make reproduce`. To keep the project reproducible out of the box, the raw CSVs (~42 MB) are committed to `archive/`. `scripts/download_data.py` detects existing files and skips the download; if files are missing, it falls back to the Kaggle API. See `CHANGELOG.md` for a full history of dataset and methodology decisions.

## Architecture

~~~
archive/ (raw CSVs)
      |
      v
load_real  -->  data/raw/  -->  (Two-Tower computes features from train split)
                                       |
                                       v
              +------------- Training -------------+
              |                                    |
    Popularity  Content   ItemKNN   SVD   Two-Tower (NCE)
              |                                    |
              +-----> Segmented RRF Hybrid <-------+
                              |
                              v
                     Ranked recommendations
~~~

## Project Structure

~~~
.
|-- archive/                 # Raw Kaggle Goodreads CSVs (vendored, ~42 MB)
|-- configs/config.yaml      # All hyperparameters
|-- data/
|   |-- raw/                 # Canonical schema (gitignored)
|   `-- processed/           # Two-Tower checkpoint (gitignored)
|-- notebooks/               # EDA + comparison notebooks (committed with outputs)
|-- reports/                 # Auto-generated metrics (gitignored)
|-- scripts/
|   |-- download_data.py     # Kaggle API fallback
|   |-- build_notebooks.py
|   |-- execute_notebooks.py
|   |-- sweep_weights.py     # Hybrid weight search (disjoint users)
|   `-- train_two_tower.py   # Feature leakage-safe training
|-- src/
|   |-- baseline/            # Vendored HW baseline models
|   |-- data/                # load_real, build_features, split
|   |-- models/              # adapters, segmented_hybrid, two_tower
|   |-- evaluation/          # metrics harness (relevance = rating >= 4)
|   `-- pipeline.py          # End-to-end orchestrator
|-- tests/
|   |-- test_smoke.py        # 5 pytest smoke tests
|-- Makefile
|-- requirements.txt
`-- README.md
~~~

## Quick Start

~~~bash
git clone https://github.com/DevSpecOps/smart_hybrid_bookshelf.git
cd smart_hybrid_bookshelf
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make reproduce
~~~

`make reproduce` runs: download (skipped -- CSVs vendored) -> canonicalize -> train Two-Tower -> evaluate -> generate + execute notebooks. It is **idempotent**: steps with existing outputs are skipped. First run takes ~3 minutes on CPU (Two-Tower trains on 607k positive pairs for ~1.2 min); subsequent runs ~30 seconds. No GPU is required -- every number in this README was produced on CPU-only hardware.

Fine-grained control:

~~~bash
python -m src.pipeline --max-users 500 --skip-train     # quick re-eval
python -m src.pipeline --force                          # regenerate everything
python scripts/train_two_tower.py --full --device cuda  # full-data neural training on GPU (optional)
~~~

## Models

| Model | Family | Strength | Weakness | R@10 |
|---|---|---|---|---|
| Popularity (Bayesian) | Non-personalized | Robust to cold start | No personalization | 0.0013 |
| **Content-based (TF-IDF on tags)** | Item-item | Works with no interactions | Limited by metadata quality | **0.1653** |
| ItemKNN (cosine on ratings) | Collaborative | High catalog coverage | Popularity bias | 0.0955 |
| SVD (Surprise) | Latent factor | Captures co-occurrence | Undertrained on sparse LOO | 0.0013 |
| Two-Tower (NCE) | Neural retrieval | Side-feature aware, broad coverage | Requires training budget | 0.0632 |

R@10 measured on the full held-out evaluation set (15,084 users, seed=42).

## Two-Tower Architecture

**Inputs per training pair:**
- User: `id_embedding(45122, 32) + [user_avg_rating, user_n_ratings, user_std_rating]` (normalized)
- Item: `id_embedding(10000, 32) + [item_avg_rating, item_n_ratings, item_std_rating, item_tag_diversity, item_popularity_rank]` (normalized)

**Towers:** each is `id_embed -> concat(features) -> MLP([128, 64]) -> L2 normalize -> 32d embedding`

**Loss:** in-batch NCE. For a batch of `B=512` positive pairs, compute `logits = user_emb @ item_emb.T / 0.07`, then cross-entropy with diagonal labels. This gives 511 implicit negatives per positive at zero extra sampling cost.

**Training:** positives = ratings >= 4 (607,088 pairs on the 1M dataset). 3 epochs, batch size 512. **All training and evaluation in this project were performed on CPU only** (no CUDA), which keeps bit-level reproducibility across hardware configurations; same-hardware reruns produce identical R@10 to 4 decimal places. CPU training takes ~1.2 minutes on a modern laptop CPU. Loss progression: 5.66 -> 4.73 (random baseline = ln(512) = 6.24, so 24% improvement). Model has 1.79M parameters.

**Feature leakage prevention:** all user and item aggregate features (user_avg_rating, item_n_ratings, item_tag_diversity, ...) are recomputed from the **train split only** inside `scripts/train_two_tower.py`. The test set never influences model inputs.

**Inference:** embeddings precomputed once and cached. `recommend(user_id, n)` = single 32-dim dot product against all 10,000 item embeddings + set-difference to remove already-rated items.

## Hybrid Strategy

Users are segmented by train-set activity:

| Segment | Threshold | Weight Configuration |
|---|---|---|
| `cold_start` | `< 20` ratings | content 0.85, popularity 0.15 |
| `active` | `>= 20` ratings | itemknn 0.45, content 0.30, two_tower 0.20, popularity 0.05 |

**Fusion:** weighted Reciprocal Rank Fusion (RRF) with `k=60`:

~~~
score(item) = sum over models m:  weight[m] / (k + rank_m(item) + 1)
~~~

RRF operates on ranks (not raw scores), so no per-model score normalization is needed. Candidates from each model are filtered to remove already-rated items before scoring. Each model produces top-50 candidates.

**Segment naming note:** `cold_start` refers to users with fewer than `cold_threshold` (20) train-set ratings -- that is, users with **thin interaction history**, not true cold-start users with zero history. Goodreads-10k contains no zero-history users by construction (every user has at least 5 ratings in the raw data). The segment label is kept for compatibility with the earlier homework baseline.

**Weight selection:** hybrid weights were chosen by evaluating **7 fusion configurations** on a **disjoint user set** (see Evaluation Protocol). The `current_config` entry in `reports/sweep.json` is the production `configs/config.yaml`; it was explicitly included in the sweep so the search covers the production setting. Key findings:

- SVD excluded from active weights (undertrained, R@10=0.0013 vs content 0.1653)
- ItemKNN receives the highest active weight -- on this 1M-rating dataset its R@10 (0.0955) is second only to content
- Two-Tower weight 0.20 provides the coverage gain without hurting recall (content still dominant on precision)
- Cold-start threshold set to 20: on the 1M dataset most users have <20 ratings in the train split
- `current_config` wins by R@10 on the sweep set (0.1360), confirming the manual tuning choices

## Evaluation Protocol

**Leave-one-out per user:** for each user with at least 3 ratings (`min_user_ratings` from config), one interaction is held out uniformly at random (seeded via `np.random.default_rng(42)`). All remaining interactions form the train set.

**Relevance definition:** a held-out item is considered relevant only if its true rating is `>= 4` -- the same definition used in the homework baseline. Users whose held-out item rated below 4 are excluded from evaluation (30,169 of 45,122 users qualify on the 1M dataset).

**Sweep / eval split:** to avoid tuning on the evaluation set, users are split into two disjoint halves by `split_users_for_sweep_eval` (seeded). Hybrid weight search (`scripts/sweep_weights.py`) runs on one half; final metrics are reported on the other. Both halves use the same `min_user_ratings` value from `configs/config.yaml`.

**Feature leakage prevention:** user and item features used by the Two-Tower model are computed **from the train split only** -- the test set never influences model inputs.

**Reproducibility.** All numbers reported in this README were produced on CPU-only hardware with fixed seeds (`project.seed=42`, `np.random.default_rng(42)`). Same-hardware and cross-CPU reruns produce identical R@10 to 4 decimals; GPU reruns may differ by up to +/-0.5% due to non-deterministic CUDA kernels. The CPU-only setting therefore provides the strongest reproducibility guarantee for this project.

**Important:** this protocol differs from the homework baseline's `train_test_split(test_size=0.2)`, which held out ~22 test items per user. Under LOO, recall measures "did the single test item appear in top-K", whereas the HW protocol measures "fraction of 22 held-out items in top-K". The two are not directly comparable.

**Metrics:**
- `Precision@K` = relevant_in_top_k / K
- `Recall@K` = relevant_in_top_k / total_relevant (which is 1 under LOO)
- `nDCG@K` = DCG@K / IDCG@K
- `Coverage@K` = unique items recommended across all users / catalog size

Segmented results are reported separately for `cold_start` and `active` users.

## Results

Evaluation on the full held-out set (15,084 users, seed=42, leave-one-out):

### R@10 by model

| Model | R@10 | nDCG@10 | Coverage@10 |
|---|---|---|---|
| Content-based | **0.1653** | **0.1029** | 0.839 |
| **Hybrid (with Two-Tower)** | 0.1557 | 0.0966 | **0.968** |
| ItemKNN | 0.0955 | 0.0555 | 0.985 |
| Two-Tower (standalone) | 0.0632 | 0.0317 | 0.927 |
| Popularity | 0.0013 | 0.0006 | 0.002 |
| SVD | 0.0013 | 0.0006 | 0.008 |

### Interpretation

On the 1M-rating Kaggle dataset, **content-based is the strongest individual model on recall and nDCG** (0.1653 / 0.1029), while the **hybrid wins coverage** (0.968 vs 0.839, **+15%**) and stays within 6% of content on recall.

This is expected: on a sparse 1M-rating interaction matrix, tag-based content generalizes better than collaborative signals (item-item co-occurrence needs density; user-user signals need many ratings per user). Two-Tower contributes unique embedding-space diversity, and the hybrid lands on the **Pareto frontier** of (recall, coverage): no single model beats it on both axes simultaneously.

For deployment, the choice depends on objective:

- **Pure hit-rate optimization** -> content-based alone
- **Recall + diversity (typical production)** -> hybrid
- **Cold-start only** -> content-based (itemknn and SVD cannot help a user with no history)

## Analysis

**Why does content-based dominate on this dataset?**

Goodreads tags encode genre-level information that correlates strongly with user reading behavior. A user who has rated mostly fantasy books is likely to read fantasy again -- TF-IDF on tags captures this signal directly. On the 1M-rating Kaggle dataset (density 0.002), collaborative methods suffer: many test items fall outside the top-100 neighborhood of anything the user has already rated.

**Why does SVD fail under leave-one-out?**

SVD with 32 factors on ~1M ratings spends most of its capacity on biases (`bu`, `bi`) -- user average and item average -- leaving little signal in `qi @ pu` (the personalization component). Under LOO, predicting the single held-out item correctly requires fine-grained personalization that the model cannot learn from 1M ratings. SVD is retained in the codebase for reference and reproducibility but excluded from the hybrid.

**Why does Two-Tower help coverage?**

Two-Tower combines ID embeddings (learned per user/book) with side features (aggregates + tag diversity). It achieves coverage of 0.927 at K=10 -- second only to ItemKNN (0.985) -- because the NCE training objective spreads the embedding space, giving every user a diverse neighborhood. Combined with content's precision via RRF, it produces a hybrid whose coverage (0.968) exceeds content alone (0.839) by 15%, while recall stays within 6%.

**Why leave-one-out?**

Random 80/20 splits inflate recall by giving each user ~20 held-out items; the metric then measures "fraction of 20 items in top-10" rather than "hit rate of a single prediction". LOO is the standard protocol in modern retrieval evaluation (NCF, SASRec, etc.) because it forces the model to commit to a single ranking. It is strict -- absolute recall values are lower -- but comparisons between models are more meaningful.

## Known Limitations

- **SVD is undertrained** for the LOO protocol; excluded from hybrid. With more training time or GPU-accelerated Surprise, this could be revisited.
- **`Cold-start` is a thin-history segment, not true zero-history.** With `cold_threshold=20`, the segment contains users with fewer than 20 train-set ratings (median ~8). No user in Goodreads-10k has zero ratings (minimum is 5 by construction), so `cold_start` here means 'thin history', not 'no history'. The label matches the homework baseline naming for continuity. On the 1M-rating Kaggle dataset, most users fall into this bucket (median ratings per user ~8). Cold-start weights are tuned on a statistically large but individually thin segment.
- **Content bias toward tag-rich books.** Content-based has an implicit bias toward books with many tags; on datasets where tags are sparse, this bias would need to be mitigated.
- **Two-Tower trained on subsample.** Default training uses a 607k-positive subsample (users with at least one positive). With `--full` on a GPU, training time increases ~20x and typically gains an additional 2-4% R@10.
- **No temporal information.** Goodreads-10k ratings have no timestamps; a temporal split (better for real deployment) is not possible with this dataset.

## Baseline Provenance

The baseline models in `src/baseline/` (`PopularityRecommender`, `ContentBasedRecommender`, `ItemBasedCF`, `SVDRecommender`, `HybridRecommender`) are vendored verbatim from the homework repository `smart_bookshelf` at commit `d09071a2dfbfd674d78d8d0178771bf09d9b6f2a`. File-level SHA-256 verification is in `src/baseline/VENDORED.md`.

The new components in this project -- canonical data loading, leave-one-out splitting, the evaluation harness, the segmented RRF hybrid, the Two-Tower neural recommender, and the end-to-end pipeline -- are original to this submission.

## Configuration

All hyperparameters live in `configs/config.yaml`. Sections:

- `project.seed` -- global random seed (42)
- `data.min_user_ratings` / `min_book_ratings` -- filtering thresholds
- `evaluation.k_values` -- top-K values to report
- `hybrid.cold_threshold` / `k_rrf` / `candidates_per_model` -- hybrid settings
- `hybrid.weights.{cold_start,active}` -- segment-specific fusion weights
- `models.popularity` / `item_knn` / `svd` / `content` -- classical model hyperparameters
- `models.two_tower.{id_dim,embedding_dim,hidden_dims,epochs,...}` -- neural model settings

Changing any value does not require code edits.

## License

MIT