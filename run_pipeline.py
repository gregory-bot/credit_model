# run_pipeline.py
"""
run_pipeline.py — End-to-end credit‑scoring pipeline.

Usage
-----
  python run_pipeline.py                  # full pipeline (combined target)
  python run_pipeline.py --target fpd_fmd # hard defaults only
  python run_pipeline.py --target par60   # DPD > 60
  python run_pipeline.py --target par90   # DPD > 90

Steps
-----
  1.  Load data              (5 credit snapshots + demographics + NPS)
  2.  Feature engineering    (1 row per loan)
  3.  Target definition      (hard / soft default)
  4.  Model training         (LR, RF, XGBoost, LightGBM) → best model
  5.  Evaluation             (ROC, PR, KS, confusion matrix, SHAP)
  6.  Portfolio scoring      (score every loan, risk bands)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so we can run the script from anywhere
sys.path.insert(0, str(Path(__file__).parent))

# ── MUST set config BEFORE importing other modules ────────────────────────
import credit_scoring.config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)-28s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(args: argparse.Namespace) -> None:
    """Execute the complete pipeline from raw data to portfolio scores."""
    
    # Now import the rest (after config is set)
    from credit_scoring import (
        load_all, build_feature_matrix, define_target, merge_target,
        train, compute_shap, score_portfolio,
    )
    from credit_scoring.evaluate import evaluate, plot_feature_importance
    from credit_scoring.feature_engineering import encode_categoricals
    
    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("Step 1 / 6 — Loading data")
    credit, demographics, nps = load_all()

    # ── 2. Feature engineering ────────────────────────────────────────────
    logger.info("Step 2 / 6 — Feature engineering")
    features = build_feature_matrix(credit, demographics, nps)

    # ── 3. Target definition ──────────────────────────────────────────────
    logger.info("Step 3 / 6 — Defining target variable")
    # FIXED: Pass the target definition from command line
    target_df = define_target(credit, target_definition=cfg.TARGET_DEFINITION)
    labelled  = merge_target(features, target_df)
    
    bad_rate = labelled["target"].mean()
    logger.info(
        "Training dataset: %d loans | Bad rate: %.1f%%",
        len(labelled), bad_rate * 100,
    )

    # ── 4. Model training ─────────────────────────────────────────────────
    logger.info("Step 4 / 6 — Training models (this may take a few minutes)")
    model_pipeline, X_test, y_test, feature_list, cv_results = train(labelled)

    # ── 5. Evaluation ─────────────────────────────────────────────────────
    logger.info("Step 5 / 6 — Evaluating model")
    evaluate(model_pipeline, X_test, y_test, model_name="Best Model", save_plots=True)
    plot_feature_importance(model_pipeline, feature_list)

    # Global SHAP
    logger.info("Computing global SHAP importance...")
    labelled_encoded = encode_categoricals(labelled.copy())
    compute_shap(model_pipeline, labelled_encoded, feature_list)

    # ── 6. Portfolio scoring ─────────────────────────────────────────────
    logger.info("Step 6 / 6 — Scoring full portfolio")
    features_encoded = encode_categoricals(features.copy())
    scores = score_portfolio(
        model_pipeline, features_encoded, feature_list,
        output_path=cfg.OUTPUTS_DIR / "portfolio_scores.csv",
    )
    _print_score_summary(scores)


def _print_score_summary(scores_df: "pd.DataFrame") -> None:
    """Nicely print the score distribution."""
    if "risk_band" not in scores_df.columns or "score" not in scores_df.columns:
        return

    bands_order = ["Excellent", "Good", "Fair", "Risky", "High Risk"]
    summary = (
        scores_df.groupby("risk_band", observed=False)
        .agg(count=("score", "count"), avg_pd=("pd", "mean"))
        .reindex([b for b in bands_order if b in scores_df["risk_band"].unique()])
        .reset_index()
    )

    total = summary["count"].sum()
    summary["pct"] = (summary["count"] / total * 100).round(1)
    summary["avg_pd"] = (summary["avg_pd"] * 100).round(2)

    print("\n📈 Portfolio Score Distribution:")
    print(summary.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Credit Scoring Pipeline")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--target", default="combined",
                        choices=["fpd_fmd", "par30", "par60", "par90", "combined"],
                        help="Target definition (default: combined)")
    args = parser.parse_args()

    # SET CONFIG BEFORE ANYTHING ELSE
    cfg.TARGET_DEFINITION = args.target
    logger.info("Using target definition: %s", cfg.TARGET_DEFINITION)

    if args.score_only:
        raise NotImplementedError("--score-only not wired up yet")
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()