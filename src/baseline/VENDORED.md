# Vendored Baseline Models

The modules in this directory are ported **verbatim** from the homework repository `smart_bookshelf`, which implemented the initial baseline recommendation models. They are kept unchanged to preserve provenance and to allow verification of the exact source code.

## Source

- Repository: https://github.com/DevSpecOps/smart_bookshelf
- Commit: `d09071a2dfbfd674d78d8d0178771bf09d9b6f2a`
- Vendored on: 2026-11-08

## Files and SHA-256

| File | SHA-256 |
|---|---|
| `popularity.py` | `48B4E2B9CD2A9DE67082A757856BE246124AEA9EAA99AB6AEC4D4BE09DE397EF` |
| `content_based.py` | `86D53FD9AB52988F399148F4EC5731BA76887CA0F22DB983521EB5797956530A` |
| `item_cf.py` | `0C48542E692DEC4B868E7142B4BE33B5E7FA79A5BDD5791AE7A2BE78C56C1446` |
| `svd_model.py` | `BF1047A3C8AD202A25627716C0811FF449E828F606CD95BDD2A8C874EEFE03A7` |
| `hybrid.py` | `A080E7204A2279A7D262FDA48B64FEC7B02FAE97BEDCDF413D3F54F8ADCBCC3A` |
| `metrics.py` | `E4341A926E71DC4110B452F9DDC92F7FFE76138022AD3FBCFD4D740A550379E4` |

## Not Vendored

The following modules from the source repository were intentionally omitted because their responsibilities are replaced by new pipeline components:

- ``data_loader.py`` - replaced by ``src/data/load_real.py``
- ``evaluate.py`` - replaced by ``src/evaluation/harness.py`` with leave-one-out protocol
- ``eda.py`` - replaced by notebooks
- ``config.py`` - replaced by ``configs/config.yaml``

## Notes

- ``svd_model.py`` depends on ``scikit-surprise``. If the dependency is missing, ``SVDRecommender`` is set to ``None`` in ``src/baseline/__init__.py`` and downstream code must handle its absence.
- ``hybrid.py`` implements weighted Reciprocal Rank Fusion without user segmentation. Segment-aware hybridization is implemented separately in ``src/models/segmented_hybrid.py``.
