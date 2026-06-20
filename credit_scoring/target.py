# credit_scoring/target.py
"""
Target variable definition for credit scoring.

Defines what constitutes a 'bad' loan (default).
Uses ONLY information from the LAST snapshot to avoid look-ahead bias.
"""

import logging
import numpy as np
import pandas as pd

from .config import (
    COL_LOAN_ID, COL_SNAPSHOT_DATE, COL_DPD, COL_STATUS_L2,
    BAD_STATUSES, PAR30_THRESHOLD, PAR60_THRESHOLD, PAR90_THRESHOLD,
)

logger = logging.getLogger(__name__)


def define_target(credit: pd.DataFrame, target_definition: str = None) -> pd.DataFrame:
    """
    Create binary target variable based on LAST snapshot outcome.
    
    Target definitions:
    - "fpd_fmd":  Hard default (First Payment Default or Formal Missed Default)
    - "par30":    Soft default (Days Past Due > 30 at last snapshot)
    - "par60":    Severe delinquency (DPD > 60 at last snapshot)
    - "par90":    Near write-off (DPD > 90 at last snapshot)
    - "combined": FPD/FMD OR DPD > 30 (default setting)
    
    IMPORTANT: Target is computed ONLY from the LAST snapshot
    to prevent temporal leakage.
    """
    # Use passed target_definition, or fall back to config
    if target_definition is None:
        from .config import TARGET_DEFINITION
        target_definition = TARGET_DEFINITION
    
    logger.info(f"Target definition: {target_definition}")
    
    # Get the last snapshot for each loan
    credit = credit.copy()
    credit["_snap_date"] = pd.to_datetime(credit[COL_SNAPSHOT_DATE])
    
    last_snap = (
        credit.sort_values("_snap_date")
        .groupby(COL_LOAN_ID)
        .last()
        .reset_index()
    )
    
    # Initialize target
    last_snap["target"] = 0
    
    # Hard defaults (FPD/FMD)
    if target_definition in ["fpd_fmd", "combined"]:
        if COL_STATUS_L2 in last_snap.columns:
            hard_default = last_snap[COL_STATUS_L2].str.upper().isin(BAD_STATUSES)
            last_snap.loc[hard_default, "target"] = 1
            n_hard = hard_default.sum()
            logger.info(f"Hard defaults (FPD/FMD): {n_hard} loans")
        else:
            logger.warning(f"Column {COL_STATUS_L2} not found — cannot detect hard defaults")
    
    # Soft defaults (DPD thresholds)
    if target_definition in ["par30", "par60", "par90", "combined"]:
        if COL_DPD in last_snap.columns:
            dpd = pd.to_numeric(last_snap[COL_DPD], errors="coerce").fillna(0)
            
            if target_definition == "par30":
                soft_default = dpd > PAR30_THRESHOLD
                threshold_label = "30"
            elif target_definition == "par60":
                soft_default = dpd > PAR60_THRESHOLD
                threshold_label = "60"
            elif target_definition == "par90":
                soft_default = dpd > PAR90_THRESHOLD
                threshold_label = "90"
            elif target_definition == "combined":
                soft_default = dpd > PAR30_THRESHOLD
                threshold_label = "30"
            else:
                soft_default = pd.Series(False, index=last_snap.index)
                threshold_label = "?"
            
            n_soft = soft_default.sum()
            last_snap.loc[soft_default, "target"] = 1
            logger.info(f"Soft defaults (DPD > {threshold_label}): {n_soft} loans")
        else:
            logger.warning(f"Column {COL_DPD} not found — cannot detect soft defaults")
    
    # Summary statistics
    n_bad = last_snap["target"].sum()
    n_total = len(last_snap)
    bad_rate = n_bad / n_total * 100 if n_total > 0 else 0
    
    logger.info(
        f"Target [{target_definition}]: {n_total} loans | "
        f"Bad={n_bad} ({bad_rate:.1f}%) | "
        f"Good={n_total - n_bad} ({100-bad_rate:.1f}%)"
    )
    
    # Return only loan_id and target
    return last_snap[[COL_LOAN_ID, "target"]]


def merge_target(features: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge target variable with feature matrix.
    Drops loans that don't have a target outcome.
    """
    n_before = len(features)
    
    merged = features.merge(target_df, on=COL_LOAN_ID, how="inner")
    
    n_after = len(merged)
    n_dropped = n_before - n_after
    
    if n_dropped > 0:
        logger.warning(
            f"Dropped {n_dropped} loans without target outcome "
            f"({n_dropped/n_before*100:.1f}%)"
        )
    else:
        logger.info(f"After target merge: {n_after} rows (dropped 0 loans without outcome)")
    
    bad_rate = merged["target"].mean() * 100
    logger.info(
        f"Final dataset: {n_after} loans | Bad rate: {bad_rate:.1f}%"
    )
    
    return merged


# Alias for backward compatibility
merge_target_with_features = merge_target