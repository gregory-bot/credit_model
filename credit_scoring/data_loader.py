# credit_scoring/data_loader.py
"""Load all raw data sources: credit snapshots, demographics, NPS."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

from .config import (
    CREDIT_DIR, SALES_FILE, NPS_FILE, CREDIT_FILES,
    COL_LOAN_ID, COL_SNAPSHOT_DATE,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df


def _normalise_loan_id(df: pd.DataFrame) -> pd.DataFrame:
    """Find the Loan-ID column (regardless of spacing) and rename to COL_LOAN_ID."""
    df = _strip_columns(df)
    # Drop columns whose name is empty after stripping (trailing blank columns)
    df = df.loc[:, df.columns != ""]
    # Locate loan-id column
    candidates = [c for c in df.columns if "loan" in c.lower() and "id" in c.lower()]
    if candidates and candidates[0] != COL_LOAN_ID:
        df = df.rename(columns={candidates[0]: COL_LOAN_ID})
    if COL_LOAN_ID not in df.columns:
        raise KeyError(f"Cannot find a loan-ID column in {list(df.columns)[:10]}")
    df[COL_LOAN_ID] = df[COL_LOAN_ID].astype(str).str.strip()
    df = df[df[COL_LOAN_ID].notna() & (df[COL_LOAN_ID] != "nan")]
    return df


def _find_col(df: pd.DataFrame, *keywords: str) -> str | None:
    """Return the first column whose lower-case name contains ALL keywords."""
    for col in df.columns:
        cl = col.lower().replace(" ", "").replace("_", "")
        if all(kw.lower().replace(" ", "").replace("_", "") in cl for kw in keywords):
            return col
    return None


# ── Credit snapshots ──────────────────────────────────────────────────────────

def load_credit_data() -> pd.DataFrame:
    """Stack all 5 quarterly credit CSV snapshots into a single DataFrame."""
    frames = []
    for fname, snap_date in CREDIT_FILES:
        fpath = CREDIT_DIR / fname
        if not fpath.exists():
            logger.warning("Missing credit file: %s", fpath)
            continue
        df = pd.read_csv(fpath, encoding="utf-8-sig", low_memory=False)
        df = _strip_columns(df)
        # Drop the spurious blank column that appears in Jun/Sep/Dec files
        df = df.loc[:, df.columns.str.strip() != ""]
        df[COL_SNAPSHOT_DATE] = pd.to_datetime(snap_date)
        frames.append(df)
        logger.info("Loaded %s → %d rows", fname, len(df))

    if not frames:
        raise FileNotFoundError(
            f"No credit files found under {CREDIT_DIR}. "
            "Check that the 'Credit Data' folder is in the project root."
        )

    credit = pd.concat(frames, ignore_index=True)
    credit = _normalise_loan_id(credit)

    # Coerce numeric columns
    numeric_cols = [c for c in credit.columns if c not in (COL_LOAN_ID, COL_SNAPSHOT_DATE)]
    for col in numeric_cols:
        try:
            credit[col] = pd.to_numeric(credit[col], errors="ignore")
        except Exception:
            pass

    logger.info(
        "Credit data: %d rows × %d cols | unique loans: %d",
        len(credit), credit.shape[1], credit[COL_LOAN_ID].nunique(),
    )
    return credit


# ── Demographics ──────────────────────────────────────────────────────────────

def load_demographics() -> dict:
    """Return dict with keys: 'gender', 'dob', 'income', 'sales'."""
    if not SALES_FILE.exists():
        raise FileNotFoundError(f"Sales & Customer Data not found: {SALES_FILE}")

    sheets = {}
    xl = pd.ExcelFile(SALES_FILE)

    for sheet in xl.sheet_names:
        key = sheet.lower().replace(" ", "_")
        df  = xl.parse(sheet, dtype=str)
        try:
            df = _normalise_loan_id(df)
        except KeyError:
            logger.warning("Sheet '%s' has no loan-ID column — skipping", sheet)
            continue

        # Excel row-limit: drop padding rows (all fields null except maybe index)
        df = df.dropna(how="all")

        if "dob" in key or "date_of_birth" in key or ("dob" in sheet.lower()):
            dob_col = _find_col(df, "birth") or _find_col(df, "dob")
            if dob_col:
                df["date_of_birth"] = pd.to_datetime(df[dob_col], errors="coerce")
            sheets["dob"] = df
            logger.info("DOB sheet → %d rows", len(df))

        elif "income" in key:
            recv_col = _find_col(df, "receiv")
            dur_col  = _find_col(df, "duration")
            if recv_col:
                df["Received"] = pd.to_numeric(df[recv_col], errors="coerce")
            if dur_col:
                df["Duration"] = pd.to_numeric(df[dur_col], errors="coerce")
            sheets["income"] = df
            logger.info("Income sheet → %d rows", len(df))

        elif "gender" in key:
            gender_col = _find_col(df, "gender")
            if gender_col and gender_col != "Gender":
                df = df.rename(columns={gender_col: "Gender"})
            sheets["gender"] = df
            logger.info("Gender sheet → %d rows", len(df))

        elif "sale" in key:
            sale_date_col = _find_col(df, "date")
            if sale_date_col:
                df["Sale Date"] = pd.to_datetime(df[sale_date_col], errors="coerce")
            for col in ["Cash Price", "Loan Price", "Loan Term"]:
                src = _find_col(df, *col.lower().split())
                if src:
                    df[col] = pd.to_numeric(df[src], errors="coerce")
            sheets["sales"] = df
            logger.info("Sales sheet → %d rows", len(df))

    return sheets


# ── NPS survey ────────────────────────────────────────────────────────────────

def load_nps_data() -> pd.DataFrame:
    """Load NPS survey responses."""
    if not NPS_FILE.exists():
        logger.warning("NPS file not found: %s", NPS_FILE)
        return pd.DataFrame()

    nps = pd.read_excel(NPS_FILE, dtype=str)
    nps = _normalise_loan_id(nps)
    nps = nps.dropna(how="all")

    score_col = _find_col(nps, "nps", "score") or _find_col(nps, "score")
    if score_col:
        nps["nps_score"] = pd.to_numeric(nps[score_col], errors="coerce")

    for flag in ("phone_locked", "payment_delay", "happy_device", "happy_service"):
        src = _find_col(nps, *flag.split("_"))
        if src:
            nps[flag] = (
                nps[src].str.strip().str.lower()
                .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
            )

    ppd_col = _find_col(nps, "promoter") or _find_col(nps, "detractor")
    if ppd_col:
        nps["nps_category"] = nps[ppd_col].str.strip()

    logger.info("NPS data → %d rows", len(nps))
    return nps


# ── Load everything ───────────────────────────────────────────────────────────

def load_all() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Convenience wrapper. Returns (credit_df, demographics_dict, nps_df)."""
    logger.info("=== Loading all data sources ===")
    credit       = load_credit_data()
    demographics = load_demographics()
    nps          = load_nps_data()
    return credit, demographics, nps
