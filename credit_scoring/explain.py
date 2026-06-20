# credit_scoring/explain.py
"""
Explainability framework using SHAP.

Produces:
  - Global feature importance (SHAP bar chart)
  - Individual explanation with positive/negative drivers
  - Human-readable reason codes (suitable for regulated environments)
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed — explainability limited to feature importance.")

from .config import OUTPUTS_DIR


# ── Reason-code dictionary ────────────────────────────────────────────────────
# Maps feature names to human-readable descriptions.
# "negative" = higher value → higher risk (bad for customer)
# "positive" = higher value → lower risk (good for customer)

REASON_CODES = {
    # ── Early payment behavior (negative = higher risk) ────────────────
    "early_max_dpd":            ("High early delinquency detected",    "negative"),
    "early_avg_dpd":            ("Consistent early payment delays",    "negative"),
    "early_dpd_std":            ("Inconsistent early payment pattern", "negative"),
    "early_dpd_trend":          ("Worsening payment trend over time",  "negative"),
    "early_had_arrears":        ("Account was in arrears early on",    "negative"),
    
    # ── Payment consistency ────────────────────────────────────────────
    "early_avg_collection_rate": ("Consistent early payment history",  "positive"),
    "early_min_collection_rate": ("Reliable minimum payment rate",     "positive"),
    "early_std_collection_rate": ("Irregular payment amounts",         "negative"),
    
    # ── Loan characteristics ───────────────────────────────────────────
    "loan_amount":               ("Adequate loan amount",              "positive"),
    "loan_term_months":          ("Manageable loan duration",          "positive"),
    "markup_ratio":              ("Higher interest cost margin",       "negative"),
    
    # ── Demographics ───────────────────────────────────────────────────
    "age_years":                 ("Established age (lower risk)",      "positive"),
    "avg_monthly_income":        ("Stable demonstrated income",        "positive"),
    "is_male":                   ("Gender indicator",                  "neutral"),
    
    # ── NPS / Customer satisfaction ────────────────────────────────────
    "nps_score":                 ("Customer satisfaction score",       "positive"),
    "is_promoter":               ("Highly satisfied customer",         "positive"),
    "is_detractor":              ("Dissatisfied customer indicator",   "negative"),
    "phone_locked":              ("Device locked for non-payment",     "negative"),
    "payment_delay":             ("History of payment delays",         "negative"),
    
    # ── Account meta ───────────────────────────────────────────────────
    "n_snapshots_early":         ("Complete early payment record",     "positive"),
    
    # ── OHE encoded features ───────────────────────────────────────────
    "age_band_18-25":            ("Younger borrower age group",        "negative"),
    "age_band_26-35":            ("Young adult age group",             "neutral"),
    "age_band_36-45":            ("Mid-career age group",              "positive"),
    "age_band_46-55":            ("Established age group",             "positive"),
    "age_band_55+":              ("Senior age group",                  "positive"),
    "income_band_Below 5K":      ("Very low income bracket",           "negative"),
    "income_band_5K-10K":        ("Low income bracket",                "negative"),
    "income_band_10K-20K":       ("Moderate income bracket",           "neutral"),
    "income_band_20K-30K":       ("Good income bracket",               "positive"),
    "income_band_30K-50K":       ("Strong income bracket",             "positive"),
    "income_band_50K-100K":      ("High income bracket",               "positive"),
    "income_band_100K-150K":     ("Very high income bracket",          "positive"),
    "income_band_150K+":         ("Top income bracket",                "positive"),
}


def _get_explainer(pipeline, X_background: pd.DataFrame):
    """Create a SHAP explainer appropriate for the pipeline's classifier."""
    clf = pipeline.named_steps["clf"]
    # Get transformed features through the non-clf pipeline steps
    transforms = [s for name, s in pipeline.steps if name != "clf"]
    X_bg = X_background.copy()
    for t in transforms:
        try:
            X_bg = pd.DataFrame(t.transform(X_bg), columns=X_background.columns)
        except:
            pass

    try:
        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
        else:
            explainer = shap.LinearExplainer(clf, X_bg)
        return explainer, X_bg
    except Exception as e:
        logger.warning("Could not create SHAP explainer: %s", e)
        return None, X_bg


# ── compute_shap function (called by run_pipeline.py) ─────────────────────────

def compute_shap(
    pipeline,
    labelled_data: pd.DataFrame,
    feature_list: list[str],
    scaler=None,
    n_samples: int = 500,
) -> pd.DataFrame | None:
    """
    Compute and save global SHAP feature importance.
    
    This is the main function called by the pipeline.
    """
    # Prepare features
    X = labelled_data[feature_list].copy()
    
    # Handle missing values
    X = X.fillna(X.median()).fillna(0)
    
    # Sample if dataset is large
    if len(X) > n_samples:
        X_sample = X.sample(n_samples, random_state=42)
    else:
        X_sample = X
    
    # Apply scaler if provided
    if scaler is not None:
        X_sample_scaled = scaler.transform(X_sample)
        X_sample = pd.DataFrame(X_sample_scaled, columns=feature_list)
    
    return global_shap_importance(pipeline, X_sample, feature_list)


# ── Global explainability ──────────────────────────────────────────────────────

def global_shap_importance(
    pipeline,
    X: pd.DataFrame,
    feature_cols: list[str],
    max_display: int = 20,
) -> pd.DataFrame | None:
    """
    Compute global SHAP feature importance and save a summary plot.
    Returns DataFrame of mean |SHAP| per feature.
    """
    if not SHAP_AVAILABLE:
        logger.warning("shap not installed — skipping global SHAP")
        return None

    X_sample = X.sample(min(500, len(X)), random_state=42) if len(X) > 500 else X
    explainer, X_bg = _get_explainer(pipeline, X_sample)

    if explainer is None:
        return None

    try:
        shap_values = explainer.shap_values(X_bg)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]   # class=1 (Bad)

        mean_abs = np.abs(shap_values).mean(axis=0)
        fi = pd.DataFrame({
            "feature":     feature_cols,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)

        # Plot
        top = fi.head(max_display)
        fig, ax = plt.subplots(figsize=(10, max_display * 0.4 + 1))
        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="steelblue")
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Global Feature Importance (SHAP)")
        out_path = OUTPUTS_DIR / "shap_importance.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("SHAP importance chart → %s", out_path)
        return fi
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return None


# ── Individual explanation ─────────────────────────────────────────────────────

def explain_customer(
    pipeline,
    customer_features: pd.DataFrame,
    feature_cols: list[str],
    pd_score: float,
    credit_score: int,
    risk_band: str,
    top_n: int = 4,
) -> dict:
    """
    Generate a customer-facing explanation for a single loan.

    SHAP logic for default prediction (class=1 = Bad):
    - POSITIVE SHAP value = pushes prediction HIGHER = MORE likely to default = BAD for customer
    - NEGATIVE SHAP value = pushes prediction LOWER = LESS likely to default = GOOD for customer

    Returns dict with:
      - credit_score
      - risk_band
      - pd_pct
      - positive_drivers: list of strings (features that HELPED the score)
      - negative_drivers: list of strings (features that HURT the score)
      - reason_codes: list of dicts {code, description, direction, shap}
    """
    result = {
        "credit_score":      credit_score,
        "risk_band":         risk_band,
        "pd_pct":            f"{pd_score * 100:.2f}%",
        "positive_drivers":  [],
        "negative_drivers":  [],
        "reason_codes":      [],
    }

    if SHAP_AVAILABLE:
        try:
            explainer, X_bg = _get_explainer(pipeline, customer_features)
            if explainer is None:
                return _fallback_explanation(customer_features, feature_cols, result, top_n)
                
            shap_values = explainer.shap_values(X_bg)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (Bad)
            
            shap_row = shap_values[0]
            shap_series = pd.Series(shap_row, index=feature_cols).sort_values()

            # Most NEGATIVE SHAP = features that REDUCED default probability = GOOD
            negative_shap = shap_series.head(top_n)
            
            # Most POSITIVE SHAP = features that INCREASED default probability = BAD
            positive_shap = shap_series.tail(top_n)[::-1]

            # Build positive drivers (helped the score)
            for feat, val in negative_shap.items():
                desc = REASON_CODES.get(feat, (feat.replace("_", " ").title(), "positive"))
                if abs(val) > 0.001:  # Only include if meaningful
                    result["positive_drivers"].append(f"✓ {desc[0]}")
                    result["reason_codes"].append({
                        "code": feat, 
                        "description": desc[0], 
                        "direction": "positive", 
                        "shap": round(float(val), 4)
                    })

            # Build negative drivers (hurt the score)
            for feat, val in positive_shap.items():
                desc = REASON_CODES.get(feat, (feat.replace("_", " ").title(), "negative"))
                if abs(val) > 0.001:  # Only include if meaningful
                    result["negative_drivers"].append(f"✗ {desc[0]}")
                    result["reason_codes"].append({
                        "code": feat, 
                        "description": desc[0], 
                        "direction": "negative", 
                        "shap": round(float(val), 4)
                    })

        except Exception as e:
            logger.warning("SHAP explanation failed: %s — falling back to feature values", e)
            result = _fallback_explanation(customer_features, feature_cols, result, top_n)
    else:
        result = _fallback_explanation(customer_features, feature_cols, result, top_n)

    return result


def _fallback_explanation(customer_features, feature_cols, result, top_n):
    """Rule-based explanation when SHAP is unavailable."""
    row = customer_features[feature_cols].iloc[0]
    
    pos_drivers = []
    neg_drivers = []
    
    for feat, (desc, direction) in REASON_CODES.items():
        if feat not in row.index:
            continue
        val = row[feat]
        if pd.isna(val) or val == 0:
            continue
        
        if direction == "positive" and val > 0:
            pos_drivers.append((feat, desc, abs(val)))
        elif direction == "negative" and val > 0:
            neg_drivers.append((feat, desc, abs(val)))
    
    # Sort by value magnitude
    pos_drivers.sort(key=lambda x: x[2], reverse=True)
    neg_drivers.sort(key=lambda x: x[2], reverse=True)
    
    result["positive_drivers"] = [f"✓ {d[1]}" for d in pos_drivers[:top_n]]
    result["negative_drivers"] = [f"✗ {d[1]}" for d in neg_drivers[:top_n]]
    
    return result


# ── Print explanation card ─────────────────────────────────────────────────────

def print_explanation(explanation: dict, loan_id: str | None = None):
    """Pretty-print a customer credit explanation."""
    print("\n" + "═" * 50)
    if loan_id:
        print(f"  Credit Explanation — Loan: {loan_id}")
    else:
        print("  Credit Score Explanation")
    print("═" * 50)
    print(f"  Credit Score   : {explanation['credit_score']}")
    print(f"  Risk Band      : {explanation['risk_band']}")
    print(f"  Probability of Default : {explanation['pd_pct']}")
    print("─" * 50)
    if explanation.get("positive_drivers"):
        print("  ✅ Positive Factors (Helping Your Score):")
        for d in explanation["positive_drivers"]:
            print(f"     {d}")
    if explanation.get("negative_drivers"):
        print("  ❌ Risk Factors (Hurting Your Score):")
        for d in explanation["negative_drivers"]:
            print(f"     {d}")
    print("═" * 50 + "\n")