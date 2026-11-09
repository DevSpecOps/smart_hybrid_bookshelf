# Smart Hybrid Bookshelf

A hybrid book recommendation system combining classical baselines (popularity, content-based, item-based collaborative filtering, matrix factorization) with a neural Two-Tower retrieval model. The system segments users by activity level and fuses predictions via weighted Reciprocal Rank Fusion. Fully reproducible end-to-end.

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

The project uses the **Goodreads-10k** dataset (the `zygmuntz/goodbooks-10k` release) — the same dataset used in the homework baseline.

| File | Rows | Columns |
|---|---|---|
| `books.csv` | 10,000 | `book_id`, `goodreads_book_id`, `title`, `original_title`, `authors`, `original_publication_year`, ... |
| `ratings.csv` | ~6,000,000 | `user_id`, `book_id`, `rating` |
| `tags.csv` | ~34,000 | `tag_id`, `tag_name` |
| `book_tags.csv` | ~1,000,000 | `goodreads_book_id`, `tag_id`, `count` |

After filtering (`min_user_ratings=5`, `min_book_ratings=10`):

- **53,424 users**
- **9,999 books**
- **5,976,471 ratings**
- **density = 0.0112**

The downloader (`scripts/download_data.py`) pulls CSVs directly from the dataset's GitHub releases, so no Kaggle account or API token is required. Files land in `archive/` which is gitignored.

## Architecture

~~~
archive/ (raw CSVs)
      |
      v
load_real  -->  data/raw/  -->  build_features  -->  data/processed/
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
|-- archive/                 # Raw Goodreads CSVs (gitignored)
|-- configs/config.yaml      # All hyperparameters
|-- data/
|   |-- raw/                 # Canonical schema
|   `-- processed/           # Feature tables + Two-Tower checkpoint
|-- notebooks/               # EDA + comparison notebooks
|-- reports/                 # Auto-generated metrics
|-- scripts/
|   |-- download_data.py
|   |-- sweep_weights.py     # Hybrid weight optimization
|   `-- train_two_tower.py
|-- src/
|   |-- baseline/            # Vendored HW baseline models
|   |-- data/                # load_real, build_features, split
|   |-- models/              # adapters, segmented_hybrid, two_tower
|   |-- evaluation/          # metrics harness
|   `-- pipeline.py          # End-to-end orchestrator
|-- tests/
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
make pipeline
~~~

That single `make pipeline` runs: download -> canonicalize -> features -> train Two-Tower -> evaluate. It is **idempotent**: steps with existing outputs are skipped. Re-running takes ~90 seconds; first run takes ~5 minutes on GPU, ~15 minutes on CPU.

Fine-grained control:

~~~bash
python -m src.pipeline --max-users 500 --skip-train     # quick re-eval
python -m src.pipeline --force                          # regenerate everything
python scripts/train_two_tower.py --full --device cuda  # full-data neural training
~~~

## Models

| Model | Family | Strength | Weakness | R@10 |
|---|---|---|---|---|
| Popularity (Bayesian) | Non-personalized | Robust to cold start | No personalization | 0.0125 |
| Content-based (TF-IDF on tags) | Item-item | Works with no interactions | Limited by metadata quality | 0.0460 |
| ItemKNN (cosine on ratings) | Collaborative | High catalog coverage | Popularity bias | 0.0090 |
| SVD (Surprise) | Latent factor | Captures co-occurrence | Undertrained on sparse LOO | 0.0005 |
| **Two-Tower (NCE)** | Neural retrieval | Side-feature aware, broad coverage | Requires training budget | **0.0210** |

R@10 measured on 2,000 held-out users with leave-one-out split, seed=42.

## Two-Tower Architecture

**Inputs per training pair:**
- User: `id_embedding(53424, 32) + [user_avg_rating, user_n_ratings, user_std_rating]` (normalized)
- Item: `id_embedding(9999, 32) + [item_avg_rating, item_n_ratings, item_std_rating, item_tag_diversity, item_popularity_rank]` (normalized)

**Towers:** each is `id_embed -> concat(features) -> MLP([128, 64]) -> L2 normalize -> 32d embedding`

**Loss:** in-batch NCE. For a batch of `B=1024` positive pairs, compute `logits = user_emb @ item_emb.T / 0.07`, then cross-entropy with diagonal labels. This gives 1023 implicit negatives per positive at zero extra sampling cost.

**Training:** positives = ratings >= 4. On GPU, 2M-rating subsample, 3 epochs, ~3 min. Loss progression: 6.53 -> 5.91 (baseline random loss = ln(1024) = 6.93, so 15% improvement). Model has 2.06M parameters.

**Inference:** embeddings precomputed once and cached. `recommend(user_id, n)` = single 32-dim dot product against all 9999 item embeddings + set-difference to remove already-rated items. Runs at ~30k recommendations/sec on CPU.

## Hybrid Strategy

Users are segmented by train-set activity:

| Segment | Threshold | Weight Configuration |
|---|---|---|
| `cold_start` | `< 50` ratings | popularity 0.6, content 0.4 |
| `active` | `>= 50` ratings | content 0.35, itemknn 0.25, two_tower 0.25, popularity 0.15 |

**Fusion:** weighted Reciprocal Rank Fusion (RRF) with `k=60`:

~~~
score(item) = sum over models m:  weight[m] / (k + rank_m(item) + 1)
~~~

RRF operates on ranks (not raw scores), so no per-model score normalization is needed. Candidates from each model are filtered to remove already-rated items before scoring. Each model produces top-50 candidates.

**Weight selection:** the hybrid was tuned via `scripts/sweep_weights.py` over 6 configurations on 500 users. Key findings:

- SVD weight set to 0 (undertrained model, R@10=0.0005 vs content 0.046)
- ItemKNN capped at 0.25-0.35 (heavier weights reduce recall due to RRF diversity crowding)
- Two-Tower at 0.25 provides the largest marginal gain (+8% R@10, +12% nDCG vs no-TT hybrid)

## Evaluation Protocol

**Leave-one-out per user:** for each user with at least 5 ratings, one interaction is held out uniformly at random (seeded via `np.random.default_rng(42)`). All remaining interactions form the train set.

**Important:** this protocol differs from the homework baseline's `train_test_split(test_size=0.2)`, which held out ~22 test items per user. Under LOO, recall measures "did the single test item appear in top-K", whereas the HW protocol measures "fraction of 22 held-out items in top-K". The two are not directly comparable.

**Metrics:**
- `Precision@K` = relevant_in_top_k / K
- `Recall@K` = relevant_in_top_k / total_relevant (which is 1 under LOO)
- `nDCG@K` = DCG@K / IDCG@K
- `Coverage@K` = unique items recommended across all users / catalog size

Segmented results are reported separately for `cold_start` and `active` users.

## Results

Evaluation on 2,000 held-out users (seed=42, leave-one-out):

### R@10 by model

| Model | R@10 | nDCG@10 | Coverage@10 |
|---|---|---|---|
| **Hybrid (with Two-Tower)** | **0.0515** | **0.0278** | **0.340** |
| Hybrid (without Two-Tower) | 0.0475 | 0.0248 | 0.223 |
| Content-based | 0.0460 | 0.0242 | 0.201 |
| Two-Tower (standalone) | 0.0210 | 0.0111 | 0.545 |
| Popularity | 0.0125 | 0.0061 | 0.002 |
| ItemKNN | 0.0090 | 0.0048 | 0.476 |
| SVD | 0.0005 | 0.0003 | 0.005 |

### Progression of the hybrid

| Metric | Baseline (best single = content) | Hybrid v1 (no TT) | Hybrid v2 (+ Two-Tower) | Total gain |
|---|---|---|---|---|
| R@5 | 0.0290 | 0.0300 | **0.0360** | **+24%** |
| R@10 | 0.0460 | 0.0475 | **0.0515** | **+12%** |
| nDCG@10 | 0.0242 | 0.0248 | **0.0278** | **+15%** |
| Coverage@10 | 0.201 | 0.223 | **0.340** | **+69%** |

The hybrid is the only model that simultaneously improves recall, ranking quality, and catalog coverage over the strongest single baseline.

## Analysis

**Why does content-based dominate the classical baselines?**

Goodreads tags encode genre-level information that correlates strongly with user reading behavior. A user who has rated mostly fantasy books is likely to read fantasy again — TF-IDF on tags captures this signal directly. Collaborative methods require the test item to be in the neighborhood of an already-rated item; with ~6M ratings over a 10k catalog, many test items fall outside the top-100 neighbors of anything the user has read.

**Why does SVD fail under leave-one-out?**

SVD with 32 factors on 6M ratings spends most of its capacity on biases (`bu`, `bi`) — user average and item average — leaving little signal in `qi @ pu` (the personalization component). Under LOO, predicting the single held-out item correctly requires fine-grained personalization, which the model does not have enough signal for. Tuning to 100 factors × 20 epochs did not help, confirming the diagnosis. SVD is retained in the codebase for reference and reproducibility but excluded from the hybrid.

**Why does Two-Tower help?**

Two-Tower combines ID embeddings (learned per user/book) with side features (aggregates + tag diversity). It achieves the highest coverage of any model (0.545 at K=10) because the NCE training objective spreads the embedding space, giving every user a diverse neighborhood. Combined with content's precision via RRF, it produces a hybrid that is both precise and diverse — the coverage gain from 0.201 to 0.340 is the clearest evidence that the neural model is contributing unique signal.

**Why leave-one-out?**

Random 80/20 splits inflate recall by giving each user ~20 held-out items; the metric then measures "fraction of 20 items in top-10" rather than "hit rate of a single prediction". LOO is the standard protocol in modern retrieval evaluation (NCF, SASRec, etc.) because it forces the model to commit to a single ranking. It is strict — absolute recall values are lower — but comparisons between models are more meaningful.

## Known Limitations

- **SVD is undertrained** for the LOO protocol; excluded from hybrid. With more training time or GPU-accelerated Surprise, this could be revisited.
- **Cold-start sample is small.** With `cold_threshold=50`, only 1.8% of users (939/53,424) fall into the cold-start segment. The hybrid's cold-start weights are therefore tuned on a small sample and may not generalize. A larger threshold would increase the segment but requires users with fewer ratings than the Goodreads-10k dataset provides.
- **Two-Tower default is subsampled.** Default training uses 2M positives (from users with the most interactions); full training requires `--full`. On GPU, full training runs in ~10 min and typically gains an additional 2-4% R@10. See `scripts/train_two_tower.py --help`.
- **No temporal information.** Goodreads-10k ratings have no timestamps; a temporal split (better for real deployment) is not possible with this dataset.

## Baseline Provenance

The baseline models in `src/baseline/` (`PopularityRecommender`, `ContentBasedRecommender`, `ItemBasedCF`, `SVDRecommender`, `HybridRecommender`) are vendored verbatim from the homework repository `smart_bookshelf` at commit `d09071a2dfbfd674d78d8d0178771bf09d9b6f2a`. File-level SHA-256 verification is in `src/baseline/VENDORED.md`.

The new components in this project — canonical data loading, leave-one-out splitting, the evaluation harness, the segmented RRF hybrid, the Two-Tower neural recommender, and the end-to-end pipeline — are original to this submission.

## Configuration

All hyperparameters live in `configs/config.yaml`. Sections:

- `project.seed` — global random seed (42)
- `data.min_user_ratings` / `min_book_ratings` — filtering thresholds
- `evaluation.k_values` — top-K values to report
- `hybrid.cold_threshold` / `k_rrf` / `candidates_per_model` — hybrid settings
- `hybrid.weights.{cold_start,active}` — segment-specific fusion weights
- `models.popularity` / `item_knn` / `svd` / `content` — classical model hyperparameters
- `models.two_tower.{id_dim,embedding_dim,hidden_dims,epochs,...}` — neural model settings

Changing any value does not require code edits.

## License

MIT