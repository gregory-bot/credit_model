# credit_scoring/score.py
"""
Credit score generation.

Converts model Probability of Default (PD) → credit score (300–850).
Uses the industry-standard log-odds linear scaling:

  score = BASE_SCORE − PDO × log2(odds)

where odds = (1 − PD) / PD  (Good:Bad ratio)

A higher score = lower risk.
"""

import logging
import math
import numpy as np
import pandas as pd

from .config import (
    SCORE_MIN, SCORE_MAX,
    PDO, BASE_SCORE, BASE_ODDS,
    RISK_BANDS, COL_LOAN_ID,
    AGE_BINS, AGE_LABELS, INCOME_BINS, INCOME_LABELS,
)

logger = logging.getLogger(__name__)


# ── Core conversion ───────────────────────────────────────────────────────────

def pd_to_score(pd_value: float) -> int:
    """
    Convert a single Probability of Default to a credit score.

    Formula:
        Factor = PDO / ln(2)
        Offset = BASE_SCORE − Factor × ln(BASE_ODDS)
        score  = Offset − Factor × ln(pd / (1 − pd))

    Clipped to [SCORE_MIN, SCORE_MAX].
    """
    pd_value = float(np.clip(pd_value, 1e-6, 1 - 1e-6))
    factor   = PDO / math.log(2)
    offset   = BASE_SCORE - factor * math.log(BASE_ODDS)
    log_odds = math.log(pd_value / (1.0 - pd_value))
    raw      = offset - factor * log_odds
    return int(np.clip(round(raw), SCORE_MIN, SCORE_MAX))


def scores_from_probs(y_prob: np.ndarray) -> np.ndarray:
    """Vectorised version of pd_to_score."""
    return np.array([pd_to_score(p) for p in y_prob])


# ── Risk band ─────────────────────────────────────────────────────────────────

def get_risk_band(score: int) -> str:
    """Map numeric score to risk label."""
    for label, (lo, hi) in RISK_BANDS.items():
        if lo <= score <= hi:
            return label
    return "Unknown"


# ── Feature encoding helper ───────────────────────────────────────────────────

def _encode_categoricals(feature_dict: dict) -> dict:
    """
    Auto-generate one-hot encoded features from raw age and income values.
    This ensures the API can accept simple raw values and convert them
    to the format the model expects.
    """
    data = feature_dict.copy()
    
    # Encode age band
    if "age_years" in data and pd.notna(data["age_years"]):
        age = data["age_years"]
        try:
            age_band = pd.cut(
                [age], bins=AGE_BINS, labels=AGE_LABELS, right=True
            ).astype(str)[0]
        except:
            age_band = "Unknown"
        
        for label in AGE_LABELS:
            col_name = f"age_band_{label}"
            data[col_name] = 1 if str(label) == str(age_band) else 0
    
    # Encode income band
    if "avg_monthly_income" in data and pd.notna(data["avg_monthly_income"]):
        income = data["avg_monthly_income"]
        try:
            income_band = pd.cut(
                [income], bins=INCOME_BINS, labels=INCOME_LABELS, right=False
            ).astype(str)[0]
        except:
            income_band = "Unknown"
        
        for label in INCOME_LABELS:
            col_name = f"income_band_{label}"
            data[col_name] = 1 if str(label) == str(income_band) else 0
    
    return data


# ── Score portfolio ───────────────────────────────────────────────────────────

def score_portfolio(
    pipeline,
    features: pd.DataFrame,
    feature_list: list[str],
    scaler=None,
    output_path=None,
) -> pd.DataFrame:
    """
    Score all loans in the portfolio.
    
    Parameters
    ----------
    pipeline    : trained sklearn Pipeline
    features    : DataFrame with all features (including non-model columns)
    feature_list: list of feature column names used by model
    scaler      : optional pre-fitted scaler
    output_path : optional path to save CSV
    
    Returns
    -------
    DataFrame with columns: LOAN_ID, pd, score, risk_band
    """
    # Extract and prepare features
    X = features[feature_list].copy()
    X = X.fillna(X.median()).fillna(0)
    
    # Apply scaler if provided
    if scaler is not None:
        X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_list)
    
    # Get loan IDs
    if COL_LOAN_ID in features.columns:
        loan_ids = features[COL_LOAN_ID]
    else:
        loan_ids = pd.Series(range(len(features)), name=COL_LOAN_ID)
    
    # Score
    result = score_dataframe(pipeline, X, loan_ids)
    
    # Rename for consistency
    result = result.rename(columns={
        "pd_score": "pd",
        "credit_score": "score"
    })
    
    # Save if path provided
    if output_path:
        result.to_csv(output_path, index=False)
        logger.info("Portfolio scores → %s", output_path)
    
    return result


# ── Score a DataFrame ─────────────────────────────────────────────────────────

def score_dataframe(
    pipeline,
    X: pd.DataFrame,
    loan_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Score a feature matrix.

    Parameters
    ----------
    pipeline  : trained sklearn Pipeline
    X         : feature matrix (same columns as training)
    loan_ids  : optional Series of LOAN_ID values

    Returns
    -------
    DataFrame with columns: loan_id, pd_score, credit_score, risk_band
    """
    y_prob   = pipeline.predict_proba(X)[:, 1]
    scores   = scores_from_probs(y_prob)
    bands    = [get_risk_band(s) for s in scores]

    result = pd.DataFrame({
        "pd_score":    np.round(y_prob, 4),
        "credit_score": scores,
        "risk_band":   bands,
    })

    if loan_ids is not None:
        result.insert(0, COL_LOAN_ID, loan_ids.values)

    logger.info(
        "Scored %d loans | Avg score: %.0f | Avg PD: %.2f%%",
        len(result), result["credit_score"].mean(), result["pd_score"].mean() * 100,
    )
    return result


# ── Single customer scorer ────────────────────────────────────────────────────

def score_single(pipeline, feature_dict: dict, feature_cols: list[str]) -> dict:
    """
    Score a single customer from a dict of feature values.
    
    Automatically encodes categorical features (age_band, income_band)
    from raw age_years and avg_monthly_income values.

    Parameters
    ----------
    pipeline      : trained pipeline
    feature_dict  : {feature_name: value} - raw values, API will encode
    feature_cols  : list of expected feature names (includes OHE columns)

    Returns
    -------
    dict with pd_score, credit_score, risk_band
    """
    # Encode categorical features
    data = _encode_categoricals(feature_dict)
    
    # Build row with all required features
    row_dict = {}
    for col in feature_cols:
        if col in data and pd.notna(data[col]):
            row_dict[col] = data[col]
        else:
            row_dict[col] = 0  # Default for missing features
    
    row = pd.DataFrame([row_dict])
    row = row.fillna(0)
    
    # Predict
    y_prob = float(pipeline.predict_proba(row)[0, 1])
    score  = pd_to_score(y_prob)
    band   = get_risk_band(score)

    return {
        "pd_score":    round(y_prob, 4),
        "pd_pct":      f"{y_prob * 100:.2f}%",
        "credit_score": score,
        "risk_band":   band,
    }


# ── Portfolio score distribution ──────────────────────────────────────────────

def score_summary(score_df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of score distribution by risk band."""
    total = len(score_df)
    rows  = []
    for band, (lo, hi) in RISK_BANDS.items():
        mask  = (score_df["credit_score"] >= lo) & (score_df["credit_score"] <= hi)
        n     = mask.sum()
        avg_pd = score_df.loc[mask, "pd_score"].mean() if n > 0 else 0
        rows.append({
            "risk_band":   band,
            "score_range": f"{lo}–{hi}",
            "count":       n,
            "pct":         f"{n / total * 100:.1f}%" if total > 0 else "0%",
            "avg_pd":      f"{avg_pd * 100:.2f}%",
        })

    return pd.DataFrame(rows)