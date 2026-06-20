# credit_scoring/__init__.py
"""
Credit Scoring package.
"""

from .data_loader         import load_all
from .feature_engineering import build_feature_matrix
from .target              import define_target, merge_target
from .train               import train
from .evaluate            import evaluate, evaluate_model
from .explain             import compute_shap
from .score               import score_portfolio

__all__ = [
    "load_all",
    "build_feature_matrix",
    "define_target",
    "merge_target",
    "train",
    "evaluate",
    "evaluate_model",
    "compute_shap",
    "score_portfolio",
]