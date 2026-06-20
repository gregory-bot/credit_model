# MoPhones Credit Scoring Engine

> **From descriptive analytics to a production credit scoring model.**
> Predicts Probability of Default, generates credit scores (300–850), and explains decisions using SHAP.

---

## What This Does

| Stage | Description |
|-------|-------------|
| **Data Pipeline** | Loads 5 credit snapshots + demographics + NPS |
| **Feature Engineering** | 30+ features: delinquency, payment behaviour, demographics, NPS |
| **Target Definition** | `is_bad` = FPD/FMD OR PAR30 (configurable) |
| **Model Training** | Logistic Regression, Random Forest, XGBoost, LightGBM — selects best by AUC |
| **Credit Score** | 300–850 using log-odds scaling (industry standard) |
| **Explainability** | SHAP global importance + per-customer reason codes |
| **API** | FastAPI scoring endpoint (`/score`, `/batch-score`) |

---

## Project Structure

```
MoPhones-Case-Study/
│
├── Credit Data/                         ← Your existing data
│   ├── Credit Data - 01-01-2025.csv
│   ├── Credit Data - 30-03-2025.csv
│   ├── Credit Data - 30-06-2025.csv
│   ├── Credit Data - 30-09-2025.csv
│   └── Credit Data - 30-12-2025.csv
│
├── Sales and Customer Data.xlsx         ← Demographics (4 sheets)
├── NPS Data.xlsx                        ← NPS survey responses
│
├── credit_scoring/                      ← NEW: ML package
│   ├── __init__.py
│   ├── config.py                        ← paths, constants, score scale
│   ├── data_loader.py                   ← load all raw data
│   ├── feature_engineering.py           ← build feature matrix
│   ├── target.py                        ← define is_bad target
│   ├── train.py                         ← train 4 models, save best
│   ├── evaluate.py                      ← AUC, KS, confusion matrix, plots
│   ├── score.py                         ← PD → 300-850 credit score
│   ├── explain.py                       ← SHAP + reason codes
│   └── api.py                           ← FastAPI endpoints
│
├── models/                              ← Auto-created: saved model
│   ├── best_model.pkl
│   └── feature_columns.pkl
│
├── outputs/                             ← Auto-created: charts, reports
│   ├── portfolio_scores.csv
│   ├── model_evaluation.png
│   ├── feature_importance.png
│   ├── shap_importance.png
│   └── model_report.txt
│
├── MoPhones_Case_Study_Analysis.ipynb   ← Your existing EDA notebook
├── run_pipeline.py                      ← NEW: single entry point
├── requirements.txt                     ← NEW: all dependencies
└── README.md                            ← You are here
```

---

## Quick Start

### Step 1 — Clone / set up virtual environment

```powershell
# Windows PowerShell
cd C:\Users\HP\MoPhones-Case-Study

python -m venv .venv
.venv\Scripts\activate
```

```bash
# macOS / Linux
cd ~/MoPhones-Case-Study
python -m venv .venv
source .venv/bin/activate
```

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⏱ Takes 2–5 minutes. LightGBM and XGBoost are the heaviest packages.

---

### Step 3 — Run the full pipeline

```bash
python run_pipeline.py
```

This runs all 6 stages in sequence:

```
1. Load data       — credit CSVs + demographics + NPS
2. Feature eng     — 30+ features per loan
3. Target          — is_bad flag (FPD/FMD or PAR30)
4. Train models    — LR, RF, XGBoost, LightGBM → saves best
5. Evaluate        — AUC, KS, confusion matrix, charts
6. Score           — outputs/portfolio_scores.csv
```

Expected output (abridged):

```
08:01:02  INFO  pipeline  Step 1 / 6 — Loading data
08:01:05  INFO  pipeline  Credit data: 85,000 rows × 34 cols | unique loans: 20,742
...
08:01:12  INFO  pipeline  Step 4 / 6 — Training models
          LogisticRegression  AUC = 0.8121 ± 0.0082
          RandomForest        AUC = 0.8934 ± 0.0061
          XGBoost             AUC = 0.9108 ± 0.0057   ← winner
          LightGBM            AUC = 0.9094 ± 0.0059

═══════════════════════════════════════════════════════
  Model Evaluation — XGBoost
═══════════════════════════════════════════════════════
  ROC-AUC   : 0.9108   (>0.75 good | >0.85 excellent)
  KS Stat   : 0.6742   (>0.30 good | >0.40 excellent)
  PR-AUC    : 0.8451
  Precision : 0.7812
  Recall    : 0.6994
  F1        : 0.7381
═══════════════════════════════════════════════════════

📈 Portfolio Score Distribution:
 risk_band  score_range  count   pct   avg_pd
 Excellent    800–850    1,240   6.0%   0.82%
 Good         700–799    4,891  23.6%   4.21%
 Fair         600–699    6,334  30.5%  12.40%
 Risky        500–599    5,218  25.2%  28.61%
 High Risk    300–499    3,059  14.7%  58.32%

✅ Pipeline complete!
   Model:   models/best_model.pkl
   Scores:  outputs/portfolio_scores.csv
   Charts:  outputs/
   Report:  outputs/model_report.txt
```

---

### Step 4 — View outputs

| File | What it shows |
|------|--------------|
| `outputs/portfolio_scores.csv` | Loan ID, PD, credit score, risk band for all loans |
| `outputs/model_evaluation.png` | ROC, PR, KS, confusion matrix, score distribution, calibration |
| `outputs/feature_importance.png` | Top 20 features by model importance |
| `outputs/shap_importance.png` | Top 20 features by mean SHAP value |
| `outputs/model_report.txt` | Full metrics + classification report |

---

### Step 5 — Explain a single customer

```bash
python run_pipeline.py --loan-id LOAN_001234
```

Output:
```
══════════════════════════════════════════════
  Credit Explanation — Loan: LOAN_001234
══════════════════════════════════════════════
  Credit Score   : 742
  Risk Band      : Good
  Probability of Default : 4.81%
──────────────────────────────────────────────
  ✅ Positive Drivers:
     ✓ Consistent payment history
     ✓ Stable income level
     ✓ Established credit history
  ❌ Risk Factors:
     ✗ History of delayed payments
     ✗ High balance vs. loan amount
══════════════════════════════════════════════
```

---

### Step 6 — Score with saved model (no retraining)

```bash
python run_pipeline.py --score-only
```

---

## Launch the Scoring API

```bash
uvicorn credit_scoring.api:app --reload --port 8000
```

Open the interactive docs: **http://localhost:8000/docs**

### Score one customer (cURL)

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "loan_id":             "LOAN_001234",
    "max_dpd":             0,
    "avg_dpd":             2.1,
    "current_dpd":         0,
    "avg_collection_rate": 0.98,
    "avg_monthly_income":  35000,
    "age_years":           32,
    "par30_ever":          0,
    "months_on_book":      8.5
  }'
```

Response:
```json
{
  "loan_id":          "LOAN_001234",
  "pd_score":         0.0481,
  "pd_pct":           "4.81%",
  "credit_score":     742,
  "risk_band":        "Good",
  "positive_drivers": ["✓ Consistent payment history", "✓ Stable income level"],
  "negative_drivers": ["✗ History of delayed payments"]
}
```

### Batch score (Python)

```python
import requests

batch = {
  "customers": [
    {"loan_id": "A001", "max_dpd": 0,  "avg_collection_rate": 0.99, "age_years": 34},
    {"loan_id": "A002", "max_dpd": 65, "avg_collection_rate": 0.52, "age_years": 22},
    {"loan_id": "A003", "max_dpd": 15, "avg_collection_rate": 0.88, "age_years": 45},
  ]
}

resp = requests.post("http://localhost:8000/batch-score", json=batch).json()
for s in resp["scores"]:
    print(f"{s['loan_id']}: score={s['credit_score']} ({s['risk_band']}) PD={s['pd_pct']}")
```

---

## Configuration

Edit `credit_scoring/config.py` to change:

| Setting | Default | Effect |
|---------|---------|--------|
| `TARGET_DEFINITION` | `"combined"` | `"fpd_fmd"` / `"par30"` / `"par60"` / `"combined"` |
| `TEST_SIZE` | `0.20` | Fraction held out for evaluation |
| `SCORE_MIN / SCORE_MAX` | `300 / 850` | Credit score range |
| `PDO` | `20` | Points to double the odds |
| `BASE_SCORE` | `600` | Score at base odds |

---

## Understanding the Credit Score

```
PD → Credit Score conversion (log-odds linear scaling)

  factor = PDO / ln(2) = 20 / 0.693 = 28.85
  offset = BASE_SCORE − factor × ln(BASE_ODDS)
         = 600 − 28.85 × ln(5)
         = 600 − 46.43 = 553.57

  score  = offset − factor × ln(pd / (1 − pd))

  PD = 1%  → score ≈ 793  (Excellent)
  PD = 5%  → score ≈ 697  (Fair)
  PD = 15% → score ≈ 597  (Risky)
  PD = 40% → score ≈ 462  (High Risk)
```

---

## Feature Catalog

| Category | Feature | Why It Matters |
|----------|---------|----------------|
| **Delinquency** | `max_dpd` | Worst-ever payment delay |
| | `avg_dpd` | Typical delay pattern |
| | `dpd_trend` | Is borrower improving or deteriorating? |
| | `par30_ever` | Ever breached the PAR30 threshold |
| | `par60_ever`, `par90_ever` | Severity of delinquency |
| **Payment** | `avg_collection_rate` | % of dues actually paid |
| | `min_collection_rate` | Worst payment month |
| | `n_times_in_arrears` | How often they fall behind |
| **Loan** | `utilisation_ratio` | Outstanding / Original (high = stressed) |
| | `loan_term_months` | Longer term = more risk window |
| | `months_on_book` | Account maturity |
| **Demographic** | `age_years` | 18-25 is the highest-risk band |
| | `avg_monthly_income` | Income level |
| | `is_male` | Gender indicator |
| **NPS/CX** | `nps_score` | Customer satisfaction (correlated with default) |
| | `phone_locked` | Past locking events signal risk |
| | `payment_delay` | System-latency delays vs intent |
| **Behavioural** | `dpd_volatility` | Erratic payment patterns |
| | `ever_hard_default` | FPD/FMD history |

---

## Model Architecture (Production)

```
Raw Data (CSV / PostgreSQL)
        ↓
   data_loader.py         ← ETL / ingestion
        ↓
feature_engineering.py    ← 30+ features per loan
        ↓
    target.py             ← is_bad label
        ↓
     train.py             ← 4 models → best by AUC
        ↓
    evaluate.py           ← AUC, KS, Precision, Recall
        ↓
     score.py             ← PD → 300-850 score
        ↓
    explain.py            ← SHAP reason codes
        ↓
      api.py              ← FastAPI /score endpoint
```

For production scale, replace file I/O with:
- **PostgreSQL** — store snapshots and scores
- **dbt** — transform raw data into features
- **MLflow** — track experiments and model registry
- **Airflow** — schedule nightly re-scoring
- **FastAPI + Docker** — deploy scoring service

---

## M-Pesa SME Extension Roadmap

The MoPhones consumer model maps directly to SME scoring via M-Pesa:

| MoPhones Feature | M-Pesa Equivalent |
|-----------------|-------------------|
| `avg_monthly_income` | Monthly M-Pesa revenue |
| `months_on_book` | Business age |
| `avg_collection_rate` | Cashflow stability |
| `max_arrears` | Liquidity stress |
| `dpd_trend` | Revenue trend |
| `n_times_in_arrears` | Late payment episodes |
| `ever_hard_default` | Future default risk |

**Migration path:**

```
Phase 1 (Now)    Consumer credit scoring on MoPhones data
Phase 2 (3 mo)   Add M-Pesa transaction features for MoPhones customers
Phase 3 (6 mo)   Retrain with M-Pesa as primary features
Phase 4 (12 mo)  Score SMEs with no credit history using M-Pesa only
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `FileNotFoundError: Credit Data` | Check folder is at `MoPhones-Case-Study/Credit Data/` |
| `No module named 'xgboost'` | Run `pip install -r requirements.txt` |
| `Model not trained yet` | Run `python run_pipeline.py` before `--score-only` |
| `KeyError: LOAN_ID` | The credit CSVs may use a different column name — check `config.py` |
| `lightgbm not found` | `pip install lightgbm` |
| `shap install fails` | `pip install shap --no-build-isolation` |

---

## Author

**Kipngeno Gregory**
- GitHub: [gregory-bot](https://github.com/gregory-bot)
- Portfolio: [gregory.co.ke](https://gregory.co.ke)

---

## License

This repository is a personal project for learning and portfolio demonstration.
