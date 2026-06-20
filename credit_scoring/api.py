# credit_scoring/api.py
"""
FastAPI scoring service.

Endpoints:
  GET  /health                 — liveness check
  POST /score                  — score a single customer
  POST /batch-score            — score many customers

Run:
    uvicorn credit_scoring.api:app --reload --port 8000
"""

from __future__ import annotations
import logging
import pickle
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config  import MODEL_PATH, FEATURES_PATH
from .score   import score_single, get_risk_band, pd_to_score
from .explain import explain_customer, SHAP_AVAILABLE

logger = logging.getLogger(__name__)
app    = FastAPI(
    title="MoPhones Credit Scoring API",
    description="Probability of Default and credit score for smartphone loans.",
    version="1.0.0",
)

# ── Load artefacts at startup ──────────────────────────────────────────────────
_PIPELINE     = None
_FEATURE_COLS = None


def _load_artefacts():
    global _PIPELINE, _FEATURE_COLS
    if _PIPELINE is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                "Model not trained yet. Run `python run_pipeline.py` first."
            )
        with open(MODEL_PATH, "rb") as f:
            _PIPELINE = pickle.load(f)
        with open(FEATURES_PATH, "rb") as f:
            _FEATURE_COLS = pickle.load(f)
        logger.info("Model loaded by API")


@app.on_event("startup")
def startup_event():
    try:
        _load_artefacts()
    except RuntimeError as e:
        logger.warning("Startup warning: %s", e)


# ── Schemas ───────────────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """Feature payload for a single customer. Send raw values - API handles encoding."""
    loan_id: str | None = Field(None, description="Loan identifier")
    
    # Early payment behavior
    early_max_dpd: float = Field(0, description="Max DPD in first 3 months")
    early_avg_dpd: float = Field(0, description="Avg DPD in first 3 months")
    early_dpd_std: float = Field(0, description="DPD volatility in first 3 months")
    early_dpd_trend: float = Field(0, description="DPD trend (slope) in first 3 months")
    early_had_arrears: int = Field(0, description="Had arrears in first 3 months (0/1)")
    
    # Payment consistency
    early_avg_collection_rate: float = Field(0, description="Avg collection rate in first 3 months")
    early_min_collection_rate: float = Field(0, description="Min collection rate in first 3 months")
    early_std_collection_rate: float = Field(0, description="Std of collection rate in first 3 months")
    
    # Loan characteristics
    loan_amount: float = Field(0, description="Loan amount")
    loan_term_months: float = Field(0, description="Loan term in months")
    markup_ratio: float = Field(0, description="Markup ratio")
    
    # Demographics (raw values - auto-encoded to bands)
    age_years: float = Field(0, description="Age in years")
    avg_monthly_income: float = Field(0, description="Average monthly income")
    is_male: int = Field(0, description="Gender (1=Male, 0=Female)")
    
    # NPS signals
    nps_score: float = Field(0, description="NPS score (0-10)")
    is_promoter: int = Field(0, description="Is promoter (NPS >= 9)")
    is_detractor: int = Field(0, description="Is detractor (NPS <= 6)")
    phone_locked: int = Field(0, description="Phone locked due to non-payment")
    payment_delay: int = Field(0, description="Experienced payment delays")
    
    # Meta
    n_snapshots_early: int = Field(3, description="Number of early snapshots")

    class Config:
        schema_extra = {
            "example": {
                "loan_id": "CUST-001",
                "early_max_dpd": 5,
                "early_avg_dpd": 2.5,
                "early_dpd_std": 1.2,
                "early_dpd_trend": 0.3,
                "early_had_arrears": 0,
                "early_avg_collection_rate": 0.95,
                "early_min_collection_rate": 0.85,
                "early_std_collection_rate": 0.05,
                "loan_amount": 50000,
                "loan_term_months": 12,
                "markup_ratio": 0.15,
                "age_years": 35,
                "avg_monthly_income": 25000,
                "is_male": 1,
                "nps_score": 8,
                "n_snapshots_early": 3
            }
        }


class ScoreResponse(BaseModel):
    loan_id: str | None
    pd_score: float
    pd_pct: str
    credit_score: int
    risk_band: str
    positive_drivers: list[str] | None = None
    negative_drivers: list[str] | None = None


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures]


class BatchResponse(BaseModel):
    scores: list[ScoreResponse]
    summary: dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_customer_payload(customer: CustomerFeatures, explain: bool = True) -> ScoreResponse:
    _load_artefacts()
    
    # Convert customer to dict (exclude loan_id and None values)
    feat_dict = {}
    for k, v in customer.dict().items():
        if k != "loan_id" and v is not None:
            feat_dict[k] = v

    # Score the customer
    result = score_single(_PIPELINE, feat_dict, _FEATURE_COLS)
    pos_drv = neg_drv = None

    # Generate explanation if requested
    if explain and SHAP_AVAILABLE:
        try:
            row = pd.DataFrame([{k: feat_dict.get(k, 0) for k in _FEATURE_COLS}])
            row = row.fillna(0)
            expl = explain_customer(
                _PIPELINE, row, _FEATURE_COLS,
                result["pd_score"], result["credit_score"], result["risk_band"],
            )
            pos_drv = expl["positive_drivers"]
            neg_drv = expl["negative_drivers"]
        except Exception as e:
            logger.warning("Explanation failed: %s", e)

    return ScoreResponse(
        loan_id=customer.loan_id,
        pd_score=result["pd_score"],
        pd_pct=result["pd_pct"],
        credit_score=result["credit_score"],
        risk_band=result["risk_band"],
        positive_drivers=pos_drv,
        negative_drivers=neg_drv,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": _PIPELINE is not None,
        "shap_available": SHAP_AVAILABLE,
    }


@app.post("/score", response_model=ScoreResponse, summary="Score a single customer")
def score_one(customer: CustomerFeatures, explain: bool = True):
    """
    Score a single customer and return their credit score and explanation.

    - **pd_score**: Probability of Default (0–1)
    - **credit_score**: Score on the 300–850 scale
    - **risk_band**: Excellent / Good / Fair / Risky / High Risk
    - **positive_drivers**: What's helping their score
    - **negative_drivers**: What's hurting their score
    """
    try:
        return _score_customer_payload(customer, explain)
    except Exception as e:
        logger.exception("Scoring error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-score", response_model=BatchResponse, summary="Score many customers")
def score_batch(request: BatchRequest):
    """Score up to 1,000 customers in one call."""
    if len(request.customers) > 1000:
        raise HTTPException(status_code=400, detail="Max 1,000 customers per batch.")
    
    _load_artefacts()

    out = []
    for customer in request.customers:
        try:
            result = _score_customer_payload(customer, explain=False)
            out.append(result)
        except Exception as e:
            logger.error("Failed to score %s: %s", customer.loan_id, e)
            out.append(ScoreResponse(
                loan_id=customer.loan_id,
                pd_score=0,
                pd_pct="Error",
                credit_score=0,
                risk_band="Error",
            ))

    scores = [s.credit_score for s in out if s.credit_score > 0]
    pd_scores = [s.pd_score for s in out if s.pd_score > 0]
    
    summary = {
        "total": len(out),
        "avg_score": round(float(np.mean(scores)), 1) if scores else 0,
        "avg_pd_pct": f"{float(np.mean(pd_scores)) * 100:.2f}%" if pd_scores else "0%",
        "band_counts": pd.Series([s.risk_band for s in out]).value_counts().to_dict(),
    }

    return BatchResponse(scores=out, summary=summary)