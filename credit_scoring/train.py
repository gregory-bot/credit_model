# credit_scoring/train.py
"""
Model training pipeline.

Trains Logistic Regression, Random Forest, XGBoost, LightGBM.
Handles class imbalance via scale_pos_weight / class_weight.
Selects and saves the best model by ROC-AUC.

IMPROVED: Stronger regularization, more estimators, better imbalance handling
to ensure high-risk customers get appropriately low scores.
"""

import logging
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.impute           import SimpleImputer
import xgboost  as xgb
import lightgbm as lgb

from .config import (
    RANDOM_STATE, TEST_SIZE, CV_FOLDS,
    MODEL_PATH, FEATURES_PATH, SCALER_PATH,
    COL_LOAN_ID,
)
from .feature_engineering import get_model_features, encode_categoricals

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Encode categoricals, select numeric features, and impute missing values.
    Returns X (DataFrame), y (Series), feature_cols (list).
    """
    df = encode_categoricals(df.copy())
    feature_cols = get_model_features(df)

    if not feature_cols:
        raise ValueError("No model features found. Run feature_engineering first.")

    # Drop leakage columns (these contain target information)
    leakage = ["ever_hard_default", "current_status_l2", "par90_ever", "par60_ever"]
    feature_cols = [c for c in feature_cols if c not in leakage]

    X = df[feature_cols].copy()
    
    # Use "target" column
    y = df["target"].astype(int)

    # Simple median imputation for numeric NaNs
    X = X.apply(lambda col: col.fillna(col.median()) if col.dtype in [np.float64, np.float32] else col)
    X = X.fillna(0)  # remaining (e.g. int cols)

    logger.info("Feature matrix: %d rows × %d features | Bad rate: %.1f%%",
                len(X), len(feature_cols), y.mean() * 100)
    return X, y, feature_cols


# ── Model definitions (IMPROVED PARAMETERS) ────────────────────────────────────

def _get_model_configs(pos_weight: float, n_jobs: int = -1) -> dict:
    """
    Return dict of model_name → sklearn-compatible estimator.
    
    IMPROVEMENTS:
    - LogisticRegression: Stronger regularization (C=0.01)
    - RandomForest: More trees (500), deeper (12), smaller leaves (10)
    - XGBoost: More trees (1000), lower LR (0.03), 5x pos_weight
    - LightGBM: More trees (1000), deeper (8), lower LR (0.03), 5x pos_weight
    """
    return {
        "LogisticRegression": LogisticRegression(
            C=0.01,
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
            n_jobs=n_jobs,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight * 5,   # 5x penalty for missing bad loans
            use_label_encoder=False,
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbosity=0,
            reg_alpha=0.1,
            reg_lambda=1.0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight * 5,   # 5x penalty for missing bad loans
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbose=-1,
            reg_alpha=0.1,
            reg_lambda=1.0,
        ),
    }


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Run stratified k-fold CV for all models.
    Returns a DataFrame with AUC scores per model.
    """
    pos_weight  = (y == 0).sum() / max((y == 1).sum(), 1)
    model_cfgs  = _get_model_configs(pos_weight)
    skf         = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    imputer     = SimpleImputer(strategy="median")

    results = {}
    for name, model in model_cfgs.items():
        # Tree models don't need scaling - they handle raw values better
        pipe = Pipeline([("imp", imputer), ("clf", model)])

        scores = cross_val_score(pipe, X[feature_cols], y, cv=skf, scoring="roc_auc", n_jobs=-1)
        results[name] = scores
        logger.info("%-20s AUC = %.4f ± %.4f", name, scores.mean(), scores.std())

    return pd.DataFrame(results)


# ── Full training ──────────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> tuple:
    """
    Full training workflow:
      1. Prepare X, y
      2. Train/test split (stratified, 80/20)
      3. Cross-validate all models
      4. Re-train best model on full training set
      5. Save model, features to disk

    Returns (best_model_pipeline, X_test, y_test, feature_cols, cv_results_df)
    """
    X, y, feature_cols = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        "Train: %d rows (bad=%.1f%%) | Test: %d rows (bad=%.1f%%)",
        len(X_train), y_train.mean() * 100,
        len(X_test),  y_test.mean()  * 100,
    )

    logger.info("=== Cross-validating all models ===")
    cv_results = cross_validate_models(X_train, y_train, feature_cols)
    best_name  = cv_results.mean().idxmax()
    best_auc   = cv_results.mean().max()
    logger.info("Best model: %s (CV AUC = %.4f)", best_name, best_auc)

    print("\n📊 Cross-Validation Results (ROC-AUC):")
    print(cv_results.agg(["mean", "std"]).round(4).to_string())

    pos_weight  = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model_cfgs  = _get_model_configs(pos_weight)
    best_model  = model_cfgs[best_name]

    # No scaling for tree models - they handle raw values better
    pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", best_model),
    ])

    pipeline.fit(X_train[feature_cols], y_train)
    logger.info("Re-trained %s on full training set", best_name)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_cols, f)

    logger.info("Model saved → %s", MODEL_PATH)
    logger.info("Features saved → %s", FEATURES_PATH)

    return pipeline, X_test, y_test, feature_cols, cv_results


# ── Load saved model ──────────────────────────────────────────────────────────

def load_model() -> tuple:
    """Load trained model pipeline and feature list from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run `python run_pipeline.py` first."
        )
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_cols = pickle.load(f)
    logger.info("Loaded model from %s", MODEL_PATH)
    return pipeline, feature_cols