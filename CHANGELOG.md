# Changelog

All notable changes to this project are documented in this file. The format follows a simple commit-by-commit history aligned with the git log.

## commit nine (latest)

### Fixed after second review round
- **Sweep/eval protocol consistency:** `scripts/sweep_weights.py` now calls `adapter.fit(train)` for all models, including Two-Tower. Previously the Two-Tower `seen` set was loaded from the checkpoint (built on full `ratings`, including held-out items), which could silently exclude test items during sweep-only ranking. Fix aligns sweep and eval on the same train-only interaction history. After re-running sweep, `current_config` still wins by R@10 (0.1360 on sweep set).
- **README: `min_user_ratings`.** Evaluation Protocol now states "at least 3 ratings" (matching `configs/config.yaml`), not "5 ratings".
- **README: `cold_start` definition.** Clarified that the label refers to users with thin interaction history (< 20 train-set ratings), not true zero-history users. Goodreads-10k has no zero-rating users by construction.
## commit eight

### Dataset alignment
- **Aligned dataset with HW spec:** switched to the Kaggle release `zygmunt/goodbooks-10k` (1M ratings, deduplicated).
- Earlier iterations used the GitHub successor release (6M ratings) -- same dataset family, different version. The Kaggle release is the canonical one referenced by the homework baseline `smart_bookshelf`.
- Raw CSVs (~42 MB) are now vendored in `archive/` so that `make reproduce` works without a Kaggle API token.

### Code review fixes
- **Data leakage:** all user and item features for the Two-Tower model are recomputed from the train split only; the test set never influences model inputs.
- **Relevance definition:** a held-out item is relevant only if its true rating >= 4 (matches the homework baseline semantics).
- **Sweep / eval disjoint users:** hybrid weights are tuned on one user half and evaluated on the other. Both halves use the same `min_user_ratings` value from config.
- **`current_config` in sweep:** the production weight configuration is explicitly included in the sweep, so `reports/sweep.json` covers the deployed setting and `current_config` wins by R@10.
- **Smoke tests:** `tests/test_smoke.py` (5 pytest tests) covering metrics, splits, and disjoint user handling.

### Reproducibility
- **CPU-only:** every number in `README.md` was produced on CPU-only hardware with fixed seeds.
- **Idempotent pipeline:** `make reproduce` skips completed stages and generates all artifacts (data, checkpoint, metrics, notebooks).
- **Vendored dataset:** no external API token required -- `archive/*.csv` in git.

## commit seven fix
- Aligned notebook markdown with eval data; strengthened README known-limitations section.

## commit seven
- Added EDA, model comparison, and hybrid analysis notebooks with inline outputs.

## commit six
- End-to-end pipeline (`src/pipeline.py`) and README with results.

## commit five
- Two-Tower neural recommender with device-agnostic training (CPU/GPU auto-detect).

## commit four
- Segment-aware weighted RRF hybrid with tuned weights (`src/models/segmented_hybrid.py`).

## commit three
- Unified adapters over vendored HW models, evaluation harness, baseline eval.

## commit two
- Vendored baseline models from `smart_bookshelf` (HW) with SHA-256 provenance.

## commit one
- Real Goodreads loader, leave-one-out split, feature builder.

## commit init
- Initial project skeleton with Goodreads-10k downloader.

## License

MIT