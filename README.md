<p align="center">
  <h1 align="center">AMC Customer Churn Prediction</h1>
  <p align="center">
    <strong>Predict which customers will leave -- before they actually do.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python"/>
    <img src="https://img.shields.io/badge/CatBoost-Champion-orange" alt="CatBoost"/>
    <img src="https://img.shields.io/badge/XGBoost-Challenger-green" alt="XGBoost"/>
    <img src="https://img.shields.io/badge/SHAP-Explainability-red" alt="SHAP"/>
    <img src="https://img.shields.io/badge/ROC--AUC-0.80-brightgreen" alt="AUC"/>
  </p>
</p>

---

## What is This?

This is the **baseline branch** for the AMC churn prediction system. It trains 4 models (Heuristic, Logistic Regression, XGBoost, CatBoost), evaluates them on a rigorous leaderboard, and explains predictions using SHAP.

> **TL;DR:** CatBoost is the champion model with ROC-AUC 0.8035. See the other branches for advanced extensions (WTTE Survival Analysis, Dataset Expansion).

### Branch Overview

| Branch | Focus | Best ROC-AUC |
|--------|-------|-------------|
| **`main`** (this) | Baseline: 4 models on Synthetic 100K | 0.8035 |
| `rnn-extension-byJB` | WTTE Survival + Ensemble (CatBoost+WTTE) | 0.8035 |
| `dataset-expansion-byJB` | Telco 7K expanded to 50K via SMOTE | **0.9350** |

---

## Model Leaderboard (Synthetic 100K Dataset)

All models evaluated on the **held-out test set** (15% of data, never seen during training):

| Model | ROC-AUC | PR-AUC | Top-Decile Lift | Precision@10% | Recall@10% | ECE | Brier |
|-------|---------|--------|-----------------|---------------|------------|-----|-------|
| Heuristic | 0.7745 | 0.6012 | 2.20 | 72.8% | 22.0% | 0.0806 | 0.1762 |
| Logistic | 0.7929 | 0.6529 | 2.23 | 74.0% | 22.3% | 0.1198 | 0.1830 |
| **XGBoost** | **0.7999** | **0.6681** | **2.33** | **77.2%** | **23.3%** | **0.0470** | **0.1631** |
| **CatBoost** | **0.8035** | **0.6710** | **2.34** | **77.5%** | **23.4%** | **0.0393** | **0.1611** |

> CatBoost wins with the best ROC-AUC (0.8035), lowest calibration error (ECE = 0.039), and lowest Brier score.

### PRD Gate Results

| Gate | Threshold | XGBoost | CatBoost |
|------|-----------|---------|----------|
| ROC-AUC >= 0.75 | -- | PASS | PASS |
| PR-AUC >= 0.40 | -- | PASS | PASS |
| ECE <= 0.05 | -- | PASS | PASS |
| Lift >= 2.5 | -- | FAIL | FAIL |

---

## Project Structure

```
AMC-master/
|
|-- src/                          # All source code
|   |-- data_processing.py        # Load, clean, engineer features, split data
|   |-- train.py                  # Train all 4 models (heuristic -> CatBoost)
|   |-- evaluate.py               # Full evaluation suite with PRD gates
|   |-- explainability.py         # SHAP global + local explanations
|   |-- plot_heatmap.py           # Correlation heatmap visualization
|
|-- artifacts/                    # Saved models, splits, and results
|   |-- model_heuristic.joblib
|   |-- model_logistic.joblib
|   |-- model_xgboost.joblib
|   |-- model_catboost.joblib
|   |-- splits.joblib
|   |-- evaluation_leaderboard.csv
|   |-- shap_feature_importance.csv
|   |-- plots/
|
|-- synthetic_customer_churn_100k.csv   # Main dataset (100K rows)
|-- Telco_churn.csv                     # Alternate telecom dataset (7K)
|-- PRD.md                              # Product Requirements Document
```

---

## Pipeline

```
Raw Data -> Clean & Engineer Features -> Train 4 Models -> Evaluate -> Explain with SHAP
```

| Step | What Happens | Script |
|------|-------------|--------|
| **1. Data Processing** | Load 100K rows, clean, engineer 15+ features, stratified 70/15/15 split | `data_processing.py` |
| **2. Training** | Train Heuristic, Logistic Regression, XGBoost, CatBoost | `train.py` |
| **3. Evaluation** | ROC-AUC, PR-AUC, Lift, Precision/Recall@10%, ECE, Brier Score | `evaluate.py` |
| **4. Explainability** | SHAP bar plots, beeswarm, waterfall plots | `explainability.py` |

---

## SHAP Explainability

### Global Feature Importance
<p align="center">
  <img src="artifacts/plots/shap_global_bar.png" width="700" alt="SHAP Global Feature Importance"/>
</p>

**Top 5 churn drivers:**
1. `estimated_salary` -- salary level strongly impacts retention
2. `calls_to_data_ratio` -- imbalanced usage patterns signal risk
3. `salary_per_dependent` -- financial burden indicator
4. `calls_made` -- engagement level
5. `data_used` -- service utilization depth

### Beeswarm Plot
<p align="center">
  <img src="artifacts/plots/shap_beeswarm.png" width="700" alt="SHAP Beeswarm Plot"/>
</p>

### Individual Predictions

**High-Risk Customer:**
<p align="center">
  <img src="artifacts/plots/shap_waterfall_high_risk.png" width="700" alt="High Risk Waterfall"/>
</p>

**Low-Risk Customer:**
<p align="center">
  <img src="artifacts/plots/shap_waterfall_low_risk.png" width="700" alt="Low Risk Waterfall"/>
</p>

### Correlation Heatmap
<p align="center">
  <img src="artifacts/plots/heatmap_correlation.png" width="600" alt="Feature Correlation Heatmap"/>
</p>

---

## Feature Engineering

| Feature Group | Examples | Why It Matters |
|--------------|----------|---------------|
| **Tenure** | `tenure_years`, `is_new_customer`, `tenure_bucket` | New customers churn 64% more |
| **Charges** | `avg_charge_per_month`, `is_high_spender`, `charges_deviation` | High spenders (>$120/mo) churn at 52% |
| **Interactions** | `mtm_high_charges`, `mtm_new_customer` | Month-to-month + new + expensive = highest risk |
| **Demographics** | `age_group`, encoded categoricals | Segment-specific behavior |

---

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost catboost shap matplotlib joblib

# Run the full pipeline
python src/data_processing.py && python src/train.py && python src/evaluate.py && python src/explainability.py
```

---

## Key Takeaways

1. **CatBoost is the champion** -- best overall performance with excellent calibration
2. **Even simple heuristics work** -- rule-based model gets ROC-AUC 0.77, proving the signal is real
3. **Salary and usage patterns** are the strongest churn predictors
4. **Month-to-month contracts** with high charges and short tenure = highest churn risk
5. **Calibration matters** -- XGBoost and CatBoost both pass the ECE <= 0.05 gate

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Framework | scikit-learn, XGBoost, CatBoost |
| Explainability | SHAP |
| Data | pandas, numpy |
| Visualization | matplotlib |

---

<p align="center">
  <sub>Built for the Promptathon Hackathon</sub>
</p>
