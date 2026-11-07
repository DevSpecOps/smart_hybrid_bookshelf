from src.baseline.popularity import PopularityRecommender
from src.baseline.content_based import ContentBasedRecommender
from src.baseline.item_cf import ItemBasedCF
from src.baseline.hybrid import HybridRecommender
from src.baseline.metrics import precision_at_k, recall_at_k, ndcg_at_k

try:
    from src.baseline.svd_model import SVDRecommender
    _HAS_SVD = True
except ImportError:
    SVDRecommender = None
    _HAS_SVD = False

__all__ = [
    'PopularityRecommender',
    'ContentBasedRecommender',
    'ItemBasedCF',
    'HybridRecommender',
    'SVDRecommender',
    'precision_at_k',
    'recall_at_k',
    'ndcg_at_k',
]
