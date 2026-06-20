# credit_scoring/feature_engineering.py
"""
Feature engineering pipeline.

Produces one row per loan_id with:
  - Demographic features  (age, gender, income)
  - Loan features         (amount, term, markup)
  - Early payment behavior (from FIRST 3 months only - NO LEAKAGE)
  - NPS signals

CRITICAL: All time-dependent features use ONLY TRAIN_SNAPSHOTS (Jan/Mar/Jun 2025)
Target is from TEST_SNAPSHOTS (Sep/Dec 2025) - completely separate time periods.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from .config import (
    COL_LOAN_ID, COL_SNAPSHOT_DATE,
    COL_DPD, COL_STATUS_L2, COL_ARREARS_STATUS,
    COL_TOTAL_PAID, COL_TOTAL_DUE, COL_ARREARS, COL_BALANCE,
    COL_CUSTOMER_AGE_DAYS,
    AGE_BINS, AGE_LABELS, INCOME_BINS, INCOME_LABELS,
    PAR30_THRESHOLD, PAR60_THRESHOLD, PAR90_THRESHOLD,
    TRAIN_SNAPSHOTS,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


# ── Helper functions ──────────────────────────────────────────────────────────

def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """Convert to datetime and remove timezone info if present."""
    result = pd.to_datetime(series, errors='coerce')
    if hasattr(result, 'dt') and result.dt.tz is not None:
        result = result.dt.tz_localize(None)
    return result


def _safe_col(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    """Return column or a series of `default` if missing."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index)


# ── Demographic features ──────────────────────────────────────────────────────

def add_age_features(
    credit: pd.DataFrame,
    dob_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge DOB and compute age at each snapshot date."""
    if dob_df is None or dob_df.empty or "date_of_birth" not in dob_df.columns:
        logger.warning("DOB data unavailable — age features will be missing")
        credit["age_years"] = np.nan
        credit["age_band"]  = "Unknown"
        return credit

    dob_slim = dob_df[[COL_LOAN_ID, "date_of_birth"]].drop_duplicates(COL_LOAN_ID)
    dob_slim["date_of_birth"] = _safe_to_datetime(dob_slim["date_of_birth"])
    
    credit = credit.merge(dob_slim, on=COL_LOAN_ID, how="left")

    snap_date = _safe_to_datetime(credit[COL_SNAPSHOT_DATE])
    
    credit["age_years"] = (
        (snap_date - credit["date_of_birth"]).dt.days / 365.25
    ).round(1)

    credit["age_band"] = pd.cut(
        credit["age_years"],
        bins=AGE_BINS, labels=AGE_LABELS, right=True
    ).astype(str)

    return credit


def add_income_features(
    credit: pd.DataFrame,
    income_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate avg monthly income = Received / Duration and band it."""
    if income_df is None or income_df.empty:
        logger.warning("Income data unavailable")
        credit["avg_monthly_income"] = np.nan
        credit["income_band"]        = "Unknown"
        return credit

    inc_slim = income_df[[COL_LOAN_ID, "Received", "Duration"]].drop_duplicates(COL_LOAN_ID)
    credit   = credit.merge(inc_slim, on=COL_LOAN_ID, how="left")

    credit["avg_monthly_income"] = (
        pd.to_numeric(credit["Received"], errors="coerce") /
        pd.to_numeric(credit["Duration"], errors="coerce").replace(0, np.nan)
    ).round(0)

    credit["income_band"] = pd.cut(
        credit["avg_monthly_income"],
        bins=INCOME_BINS, labels=INCOME_LABELS, right=False
    ).astype(str)

    return credit


def add_gender_features(
    credit: pd.DataFrame,
    gender_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge gender and one-hot encode (Male=1)."""
    if gender_df is None or gender_df.empty:
        credit["is_male"] = np.nan
        return credit

    g_slim  = gender_df[[COL_LOAN_ID, "Gender"]].drop_duplicates(COL_LOAN_ID)
    credit  = credit.merge(g_slim, on=COL_LOAN_ID, how="left")
    credit["is_male"] = (
        credit["Gender"].str.strip().str.lower()
        .map({"male": 1, "m": 1, "female": 0, "f": 0})
    )
    return credit


def add_sales_features(
    credit: pd.DataFrame,
    sales_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge loan term, cash price, loan price from sales sheet."""
    if sales_df is None or sales_df.empty:
        return credit

    cols_want = [COL_LOAN_ID, "Sale Date", "Cash Price", "Loan Price", "Loan Term"]
    cols_have = [c for c in cols_want if c in sales_df.columns]
    s_slim    = sales_df[cols_have].drop_duplicates(COL_LOAN_ID)
    credit    = credit.merge(s_slim, on=COL_LOAN_ID, how="left")
    return credit


# ── Snapshot-level features (EARLY SNAPSHOTS ONLY - NO LEAKAGE) ───────────────

def compute_snapshot_features(credit: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ONLY early snapshots (first 3) per loan_id.
    
    CRITICAL: Uses ONLY TRAIN_SNAPSHOTS (Jan, Mar, Jun 2025).
    Target comes from TEST_SNAPSHOTS (Sep, Dec 2025).
    This completely prevents temporal data leakage.
    
    REMOVED LEAKERS:
    - months_on_book (uses last snapshot - same as target period)
    - utilisation_ratio (uses current_balance from last snapshot)
    - current_balance (from last snapshot - same as target period)
    - All PAR flags and arrears counters (these ARE the target)
    """
    credit = credit.copy()
    credit[COL_SNAPSHOT_DATE] = _safe_to_datetime(credit[COL_SNAPSHOT_DATE])
    
    # USE ONLY EARLY SNAPSHOTS for feature computation
    early_dates = pd.to_datetime(TRAIN_SNAPSHOTS)
    early_credit = credit[credit[COL_SNAPSHOT_DATE].isin(early_dates)].copy()
    
    if len(early_credit) == 0:
        logger.error("No early snapshots found! Check TRAIN_SNAPSHOTS config.")
        early_credit = credit
    
    early_credit[COL_DPD] = _safe_col(early_credit, COL_DPD)
    
    total_paid = _safe_col(early_credit, COL_TOTAL_PAID)
    total_due = _safe_col(early_credit, COL_TOTAL_DUE)
    
    # Collection rate from EARLY snapshots only
    early_credit["_collection_rate"] = (
        total_paid / total_due.replace(0, np.nan)
    ).clip(0, 1)
    
    # Early arrears indicator
    if COL_ARREARS_STATUS in early_credit.columns:
        early_credit["_early_arrears"] = (
            early_credit[COL_ARREARS_STATUS].str.lower() == "arrears"
        ).astype(int)
    else:
        early_credit["_early_arrears"] = 0
    
    grp = early_credit.groupby(COL_LOAN_ID)
    
    agg = pd.DataFrame({
        # Early payment behavior (first 3 months)
        "early_max_dpd":              grp[COL_DPD].max().fillna(0),
        "early_avg_dpd":              grp[COL_DPD].mean().fillna(0).round(1),
        "early_dpd_std":              grp[COL_DPD].std().fillna(0).round(2),
        "early_had_arrears":          grp["_early_arrears"].max().fillna(0).astype(int),
        
        # Early payment consistency
        "early_avg_collection_rate":   grp["_collection_rate"].mean().round(3),
        "early_min_collection_rate":   grp["_collection_rate"].min().round(3),
        "early_std_collection_rate":   grp["_collection_rate"].std().fillna(0).round(3),
        
        # Account info
        "n_snapshots_early":          grp[COL_SNAPSHOT_DATE].count(),
    })
    
    # DPD trend from early snapshots only
    def dpd_slope(group):
        g = group.sort_values(COL_SNAPSHOT_DATE)
        y = g[COL_DPD].values.astype(float)
        y = np.nan_to_num(y, nan=0.0)
        if len(y) < 2:
            return 0.0
        x = np.arange(len(y))
        try:
            slope, *_ = stats.linregress(x, y)
            return round(float(slope), 3)
        except:
            return 0.0
    
    agg["early_dpd_trend"] = grp.apply(dpd_slope).fillna(0).round(3)
    
    # Get loan-level data from ORIGINATION (first snapshot ever)
    first_snap = credit.sort_values(COL_SNAPSHOT_DATE).groupby(COL_LOAN_ID).first()
    
    for col in ["Loan Price", "Cash Price", "Loan Term"]:
        if col in first_snap.columns:
            agg[col.lower().replace(" ", "_")] = first_snap[col]
    
    agg = agg.reset_index()
    logger.info("Early snapshot aggregation → %d loan features", len(agg))
    return agg


# ── Loan features (from origination only - no time-dependent values) ──────────

def add_loan_features(
    features: pd.DataFrame,
    credit: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add loan-level features from ORIGINATION only.
    
    REMOVED: utilisation_ratio, current_balance, months_on_book
    (these use values from the target time period)
    """
    
    # Loan amount (from origination)
    loan_amt_col = next(
        (c for c in ["loan_price", "LOAN_AMOUNT", "LOAN_PRICE"] 
         if c in features.columns),
        None
    )
    
    if loan_amt_col:
        features["loan_amount"] = pd.to_numeric(
            features[loan_amt_col], errors="coerce"
        )
    else:
        features["loan_amount"] = np.nan
    
    # Markup ratio = (Loan Price - Cash Price) / Cash Price
    if "cash_price" in features.columns and loan_amt_col:
        cp = pd.to_numeric(features["cash_price"], errors="coerce")
        la = pd.to_numeric(features[loan_amt_col], errors="coerce")
        features["markup_ratio"] = ((la - cp) / cp.replace(0, np.nan)).round(3)
    else:
        features["markup_ratio"] = np.nan
    
    # Loan term (months)
    if "loan_term" in features.columns:
        features["loan_term_months"] = pd.to_numeric(
            features["loan_term"], errors="coerce"
        )
    else:
        features["loan_term_months"] = np.nan
    
    return features


# ── NPS features ──────────────────────────────────────────────────────────────

def add_nps_features(
    features: pd.DataFrame,
    nps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge NPS signals onto the feature matrix."""
    if nps_df is None or nps_df.empty:
        for col in ("nps_score", "is_promoter", "is_detractor", 
                    "phone_locked", "payment_delay"):
            features[col] = np.nan
        return features

    nps_cols = [COL_LOAN_ID]
    for col in ("nps_score", "phone_locked", "payment_delay", 
                "happy_device", "happy_service"):
        if col in nps_df.columns:
            nps_cols.append(col)
    if "nps_category" in nps_df.columns:
        nps_cols.append("nps_category")

    nps_slim = nps_df[nps_cols].drop_duplicates(COL_LOAN_ID)
    features = features.merge(nps_slim, on=COL_LOAN_ID, how="left")

    if "nps_score" in features.columns:
        features["is_promoter"]  = (features["nps_score"] >= 9).astype(float)
        features["is_detractor"] = (features["nps_score"] <= 6).astype(float)
    else:
        features["is_promoter"]  = np.nan
        features["is_detractor"] = np.nan

    return features


# ── Master pipeline ───────────────────────────────────────────────────────────

def build_feature_matrix(
    credit: pd.DataFrame,
    demographics: dict,
    nps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Full feature-engineering pipeline. NO TARGET LEAKAGE."""
    logger.info("=== Building feature matrix ===")

    credit[COL_SNAPSHOT_DATE] = _safe_to_datetime(credit[COL_SNAPSHOT_DATE])

    # 1. Add demographic columns
    credit = add_age_features(credit, demographics.get("dob"))
    credit = add_income_features(credit, demographics.get("income"))
    credit = add_gender_features(credit, demographics.get("gender"))
    credit = add_sales_features(credit, demographics.get("sales"))

    # 2. Aggregate EARLY snapshots → 1 row per loan
    snapshot_feats = compute_snapshot_features(credit)

    # 3. Add loan features from origination
    snapshot_feats = add_loan_features(snapshot_feats, credit)

    # 4. Carry over demographic features (last observed value)
    demo_cols = ["age_years", "age_band", "avg_monthly_income", 
                 "income_band", "is_male"]
    demo_cols = [c for c in demo_cols if c in credit.columns]

    if demo_cols:
        last_demo = (
            credit.sort_values(COL_SNAPSHOT_DATE)
            .groupby(COL_LOAN_ID)[demo_cols]
            .last()
            .reset_index()
        )
        snapshot_feats = snapshot_feats.merge(
            last_demo, on=COL_LOAN_ID, how="left"
        )

    # 5. NPS features
    snapshot_feats = add_nps_features(snapshot_feats, nps_df)

    logger.info(
        "Feature matrix: %d loans × %d features",
        len(snapshot_feats), snapshot_feats.shape[1],
    )
    return snapshot_feats


# ── Categorical encoding ──────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode age_band and income_band."""
    cat_cols = [c for c in ("age_band", "income_band") if c in df.columns]
    if not cat_cols:
        return df

    dummies = pd.get_dummies(df[cat_cols], prefix=cat_cols, drop_first=False)
    df      = pd.concat([df.drop(columns=cat_cols), dummies], axis=1)
    return df


# ── Final numeric feature list (NO LEAKAGE) ───────────────────────────────────

NUMERIC_FEATURES = [
    # Early payment behavior (from first 3 months ONLY)
    "early_max_dpd", "early_avg_dpd", "early_dpd_std",
    "early_dpd_trend", "early_had_arrears",
    
    # Early payment consistency
    "early_avg_collection_rate", "early_min_collection_rate", 
    "early_std_collection_rate",
    
    # Loan characteristics (from origination - not time-dependent)
    "loan_amount", "loan_term_months", "markup_ratio",
    
    # Demographic (borrower characteristics - not time-dependent)
    "age_years", "avg_monthly_income", "is_male",
    
    # NPS signals
    "nps_score", "is_promoter", "is_detractor", 
    "phone_locked", "payment_delay",
    
    # Account meta
    "n_snapshots_early",
]

# REMOVED LEAKERS:
# "months_on_book" - used last snapshot (target period)
# "utilisation_ratio" - used current_balance from last snapshot
# "current_balance" - from last snapshot (target period)
# "current_dpd", "max_dpd", "avg_dpd" - from all snapshots including target
# "par30_ever", "par60_ever", "par90_ever" - these ARE the target
# "ever_hard_default" - this IS the target
# "n_times_in_arrears", "max_arrears", "avg_arrears" - target leakage


def get_model_features(df: pd.DataFrame) -> list[str]:
    """Return the subset of NUMERIC_FEATURES that actually exist in df."""
    available = [f for f in NUMERIC_FEATURES if f in df.columns]
    # Add one-hot encoded age/income bands
    ohe_cols = [c for c in df.columns if c.startswith("age_band_") 
                or c.startswith("income_band_")]
    return available + ohe_cols