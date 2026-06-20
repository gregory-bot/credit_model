# credit_scoring/evaluate.py
"""
Model evaluation — credit-risk-appropriate metrics and charts.

Metrics that matter most in credit risk:
  KS Statistic   — separation power (regulatory standard)
  ROC-AUC        — overall discrimination
  PR-AUC         — performance under class imbalance
  Precision/Recall at operating threshold
  Confusion Matrix at 0.5 and optimal threshold
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

from .config import OUTPUTS_DIR, REPORT_PATH, RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)


# ── KS Statistic ─────────────────────────────────────────────────────────────

def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """KS = max difference between cumulative good and bad distributions."""
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    n_bad  = df["y"].sum()
    n_good = len(df) - n_bad
    if n_bad == 0 or n_good == 0:
        return 0.0
    df["cum_bad"]  = (df["y"] == 1).cumsum() / n_bad
    df["cum_good"] = (df["y"] == 0).cumsum() / n_good
    return float((df["cum_bad"] - df["cum_good"]).abs().max())


# ── Optimal threshold (Youden's J) ───────────────────────────────────────────

def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximises Sensitivity + Specificity − 1 (Youden's J)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


# ── NEW: evaluate_model function (called by run_pipeline.py) ──────────────────

def evaluate_model(
    pipeline,
    labelled_data: pd.DataFrame,
    feature_list: list[str],
    scaler=None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Evaluate model on a held-out test set.
    
    Parameters:
    - pipeline: trained sklearn pipeline
    - labelled_data: DataFrame with features + "target" column
    - feature_list: list of feature column names
    - scaler: optional pre-fitted scaler (if not in pipeline)
    - test_size: fraction for test split
    - random_state: random seed
    """
    # Prepare data
    X = labelled_data[feature_list].copy()
    y = labelled_data["target"].copy()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # If scaler provided and not in pipeline, apply it
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Evaluate
    return evaluate(pipeline, X_test_scaled, y_test)


# ── Full evaluation ────────────────────────────────────────────────────────────

def evaluate(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Best Model",
    save_plots: bool = True,
) -> dict:
    """
    Compute all evaluation metrics and (optionally) save charts to outputs/.
    Returns dict of metric name → value.
    """
    y_prob  = pipeline.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)
    ks      = ks_statistic(y_test.values, y_prob)
    thresh  = optimal_threshold(y_test.values, y_prob)

    y_pred  = (y_prob >= thresh).astype(int)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)

    metrics = {
        "model":      model_name,
        "roc_auc":    round(auc,    4),
        "pr_auc":     round(pr_auc, 4),
        "ks_stat":    round(ks,     4),
        "precision":  round(prec,   4),
        "recall":     round(rec,    4),
        "f1":         round(f1,     4),
        "threshold":  round(thresh, 4),
        "n_test":     len(y_test),
        "bad_rate":   round(y_test.mean(), 4),
    }

    # Print summary
    print("\n" + "═" * 55)
    print(f"  Model Evaluation — {model_name}")
    print("═" * 55)
    print(f"  ROC-AUC   : {auc:.4f}   (>0.75 good | >0.85 excellent)")
    print(f"  PR-AUC    : {pr_auc:.4f}   (baseline = bad rate {y_test.mean():.3f})")
    print(f"  KS Stat   : {ks:.4f}   (>0.30 good | >0.40 excellent)")
    print(f"  Threshold : {thresh:.4f}   (Youden's J)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print("─" * 55)
    print(f"  Confusion Matrix (threshold = {thresh:.2f}):")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print("═" * 55 + "\n")

    if save_plots:
        _plot_evaluation(y_test.values, y_prob, model_name, cm, thresh)
        plot_feature_importance(pipeline, X_test.columns.tolist())

    # Write text report
    with open(REPORT_PATH, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=["Good", "Bad"]))

    logger.info("Evaluation complete. Report → %s", REPORT_PATH)
    return metrics


# ── Charts ────────────────────────────────────────────────────────────────────

def _plot_evaluation(y_true, y_prob, model_name, cm, thresh):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Model Evaluation — {model_name}", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve"); ax1.legend()

    # 2. Precision-Recall Curve
    ax2 = fig.add_subplot(gs[0, 1])
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    ax2.plot(rec_arr, prec_arr, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax2.axhline(y_true.mean(), color="r", linestyle="--", label=f"Baseline = {y_true.mean():.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve"); ax2.legend()

    # 3. KS Plot
    ax3 = fig.add_subplot(gs[0, 2])
    df  = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    n_bad  = df["y"].sum();  n_good = len(df) - n_bad
    x_axis = np.linspace(0, 1, len(df))
    cum_bad  = (df["y"] == 1).values.cumsum() / n_bad
    cum_good = (df["y"] == 0).values.cumsum() / n_good
    ax3.plot(x_axis, cum_bad,  label="Bads",  lw=2)
    ax3.plot(x_axis, cum_good, label="Goods", lw=2)
    ks = ks_statistic(y_true, y_prob)
    ax3.set_title(f"KS Plot (KS = {ks:.4f})")
    ax3.set_xlabel("Population (sorted by score)"); ax3.legend()

    # 4. Confusion Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    im  = ax4.imshow(cm, interpolation="nearest", cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
    ax4.set_xticklabels(["Pred Good", "Pred Bad"])
    ax4.set_yticklabels(["True Good", "True Bad"])
    ax4.set_title(f"Confusion Matrix\n(threshold = {thresh:.2f})")

    # 5. Score distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(y_prob[y_true == 0], bins=40, alpha=0.6, label="Good (0)", color="green")
    ax5.hist(y_prob[y_true == 1], bins=40, alpha=0.6, label="Bad (1)",  color="red")
    ax5.axvline(thresh, color="k", linestyle="--", label=f"Threshold = {thresh:.2f}")
    ax5.set_xlabel("P(Default)"); ax5.set_ylabel("Count")
    ax5.set_title("Score Distribution"); ax5.legend()

    # 6. Calibration
    ax6 = fig.add_subplot(gs[1, 2])
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax6.plot(prob_pred, prob_true, "s-", label="Model")
    ax6.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax6.set_xlabel("Mean Predicted Prob"); ax6.set_ylabel("Fraction of Positives")
    ax6.set_title("Calibration Plot"); ax6.legend()

    out_path = OUTPUTS_DIR / "model_evaluation.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Evaluation chart saved → %s", out_path)


# ── Feature importance ─────────────────────────────────────────────────────────

def plot_feature_importance(pipeline, feature_cols: list[str], top_n: int = 20):
    """Extract and plot feature importance from the trained estimator."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imps = np.abs(clf.coef_[0])
    else:
        logger.warning("Model does not expose feature importances")
        return

    fi = pd.Series(imps, index=feature_cols).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, top_n * 0.4 + 1))
    fi[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    out_path = OUTPUTS_DIR / "feature_importance.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance chart → %s", out_path)
    return fi