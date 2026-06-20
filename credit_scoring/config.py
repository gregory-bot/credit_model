# credit_scoring/config.py
"""Central configuration — paths, column names, model settings, score scale."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
CREDIT_DIR  = BASE_DIR / "Credit Data"
MODELS_DIR  = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
SALES_FILE  = BASE_DIR / "Sales and Customer Data.xlsx"
NPS_FILE    = BASE_DIR / "NPS Data.xlsx"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── Credit snapshot files ──────────────────────────────────────────────────────
CREDIT_FILES = [
    ("Credit Data - 01-01-2025.csv", "2025-01-01"),
    ("Credit Data - 30-03-2025.csv", "2025-03-31"),
    ("Credit Data - 30-06-2025.csv", "2025-06-30"),
    ("Credit Data - 30-09-2025.csv", "2025-09-30"),
    ("Credit Data - 30-12-2025.csv", "2025-12-30"),
]

# ── Temporal split: Train on early snapshots, test on later ───────────────────
# Features must use only information available at prediction time
TRAIN_SNAPSHOTS = ["2025-01-01", "2025-03-31", "2025-06-30"]
TEST_SNAPSHOTS  = ["2025-09-30", "2025-12-30"]

# ── Raw column names (as they appear in source data) ──────────────────────────
COL_LOAN_ID           = "LOAN_ID"
COL_SNAPSHOT_DATE     = "SNAPSHOT_DATE"
COL_DPD               = "DAYS_PAST_DUE"
COL_STATUS_L1         = "ACCOUNT_STATUS_L1"
COL_STATUS_L2         = "ACCOUNT_STATUS_L2"
COL_ARREARS_STATUS    = "BALANCE_DUE_STATUS"
COL_TOTAL_PAID        = "TOTAL_PAID"
COL_TOTAL_DUE         = "TOTAL_DUE_TODAY"
COL_ARREARS           = "ARREARS"
COL_BALANCE           = "BALANCE"
COL_CUSTOMER_AGE_DAYS = "CUSTOMER_AGE"   # days since sale — NOT biological age

# ── Target variable ────────────────────────────────────────────────────────────
BAD_STATUSES      = ["FPD", "FMD"]          # hard default statuses
PAR30_THRESHOLD   = 30
PAR60_THRESHOLD   = 60
PAR90_THRESHOLD   = 90
# Options: "fpd_fmd" | "par30" | "par60" | "combined"
TARGET_DEFINITION = "combined"              # FPD/FMD  OR  DPD > 30

# ── Age bands ─────────────────────────────────────────────────────────────────
AGE_BINS   = [0, 25, 35, 45, 55, 200]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "55+"]

# ── Income bands (KES / month) ────────────────────────────────────────────────
INCOME_BINS   = [0, 5_000, 10_000, 20_000, 30_000, 50_000, 100_000, 150_000, float("inf")]
INCOME_LABELS = [
    "Below 5K", "5K-10K", "10K-20K", "20K-30K",
    "30K-50K", "50K-100K", "100K-150K", "150K+"
]

# ── Credit score scaling (300 – 850) ──────────────────────────────────────────
SCORE_MIN  = 300
SCORE_MAX  = 850
PDO        = 20    # points to double the odds
BASE_SCORE = 600
BASE_ODDS  = 5.0   # Good:Bad ratio at the base score

RISK_BANDS = {
    "Excellent": (800, 850),
    "Good":      (700, 799),
    "Fair":      (600, 699),
    "Risky":     (500, 599),
    "High Risk": (300, 499),
}

# ── Model training ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# ── Saved artefact paths ───────────────────────────────────────────────────────
MODEL_PATH    = MODELS_DIR / "best_model.pkl"
FEATURES_PATH = MODELS_DIR / "feature_columns.pkl"
SCALER_PATH   = MODELS_DIR / "scaler.pkl"
REPORT_PATH   = OUTPUTS_DIR / "model_report.txt"