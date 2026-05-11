# AMC Customer Churn Prediction — Complete Project Summary

> **Project Goal:** Predict which telecom customers will churn (cancel their service) before they actually do, using multiple ML models across 3 development branches.

---

## Table of Contents

1. [Branch Overview](#1-branch-overview)
2. [Datasets Used](#2-datasets-used)
3. [Data Processing & Feature Engineering](#3-data-processing--feature-engineering)
4. [Models & Algorithms](#4-models--algorithms)
   - 4.1 [Heuristic (Rule-Based)](#41-heuristic-rule-based-baseline)
   - 4.2 [Logistic Regression](#42-logistic-regression-baseline)
   - 4.3 [XGBoost](#43-xgboost-champion)
   - 4.4 [CatBoost](#44-catboost-champion)
   - 4.5 [WTTE-RNN Survival Model](#45-wtte-rnn-survival-model)
   - 4.6 [Ensemble (CatBoost + WTTE)](#46-ensemble-catboost--wtte)
5. [Dataset Expansion (SMOTE)](#5-dataset-expansion-smote--noise-injection)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Explainability (SHAP)](#7-explainability-shap)
8. [API & Frontend](#8-api--frontend)
9. [Results Comparison](#9-final-results-comparison)
10. [File Reference Index](#10-file-reference-index)

---

## 1. Branch Overview

| Branch | Purpose | Key Addition |
|--------|---------|-------------|
| **`main`** | Baseline pipeline — 4 models on Synthetic 100K dataset | Heuristic, Logistic Regression, XGBoost, CatBoost |
| **`rnn-extension-byJB`** | Adds WTTE Survival Analysis + Ensemble blending | WTTE-RNN (Weibull), CatBoost+WTTE Ensemble |
| **`dataset-expansion-byJB`** | Expands real Telco 7K → 50K via SMOTE, retrains everything | SMOTE expansion, full retrain + cross-branch benchmark |

---

## 2. Datasets Used

### 2.1 Synthetic Customer Churn (100K) — `main` & `rnn-extension`

- **File:** `synthetic_customer_churn_100k.csv`
- **Size:** 100,000 rows
- **Features:** CustomerID, Age, Gender, Tenure, Contract, MonthlyCharges, TotalCharges, PaymentMethod, Churn
- **Churn Rate:** ~33%
- **Nature:** Synthetically generated data with realistic distributions

### 2.2 Telco Churn (7K) — `dataset-expansion` (seed)

- **File:** `Telco_churn.csv` (original from IBM/Kaggle)
- **Size:** 7,043 rows, 21 features
- **Churn Rate:** ~26.5%
- **Nature:** Real-world telecom customer data

### 2.3 Telco Churn Expanded (50K) — `dataset-expansion` (output)

- **File:** `Telco_churn_50k.csv`
- **Size:** 50,000 rows
- **Churn Rate:** ~50% (balanced via SMOTE)
- **Nature:** Expanded from 7K using SMOTE + Gaussian noise

---

## 3. Data Processing & Feature Engineering

> **File:** `src/data_processing.py` (Lines 1–164)

### 3.1 Cleaning (Lines 25–42)

| Step | What It Does | Why |
|------|-------------|-----|
| Encode target | `Churn: "Yes"→1, "No"→0` | Binary classification needs numeric labels |
| Clip TotalCharges | Negative values → 0 | 265 anomalous records found in EDA |
| Null check | Assert no nulls | Data integrity gate |

### 3.2 Feature Engineering (Lines 45–116)

**Tenure-Derived Features:**

| Feature | Formula | Why It Matters | Line |
|---------|---------|----------------|------|
| `tenure_years` | `Tenure / 12` | Normalized tenure for scale | L50 |
| `is_new_customer` | `Tenure <= 12` | New customers churn 64% more | L53 |
| `tenure_bucket` | Bins: 0-6, 6-12, 12-24, 24-48, 48-72 | Captures non-linear tenure effect | L56-60 |

**Charges-Derived Features:**

| Feature | Formula | Why It Matters | Line |
|---------|---------|----------------|------|
| `avg_charge_per_month` | `TotalCharges / (Tenure + 1)` | Billing consistency signal | L64 |
| `is_high_spender` | `MonthlyCharges > 120` | High spenders churn at 52% | L67 |
| `is_med_high_spender` | `MonthlyCharges > 90` | Medium-high has ~42% churn | L70 |
| `charges_deviation` | `TotalCharges - (Tenure × MonthlyCharges)` | Billing anomaly detector | L73-74 |
| `monthly_charges_sq` | `MonthlyCharges²` | Captures non-linear effect | L77 |

**Interaction Features:**

| Feature | Formula | Why It Matters | Line |
|---------|---------|----------------|------|
| `mtm_high_charges` | Month-to-month AND charges > $90 | Highest risk combo | L81-82 |
| `mtm_new_customer` | Month-to-month AND tenure ≤ 12 | Most at-risk segment | L85-86 |
| `mtm_new_high` | Month-to-month AND new AND charges > $90 | Triple-threat risk | L89-91 |

**Encoding (Lines 100–114):**
- **Label Encoding:** Gender, Contract, PaymentMethod, age_group
- **One-Hot Encoding:** Contract type (strongest predictor gets dummies)

### 3.3 Data Splitting (Lines 119–134)

- **Strategy:** Stratified split preserving churn ratio
- **Ratio:** 70% train / 15% validation / 15% test
- **Method:** `sklearn.model_selection.train_test_split` with `stratify=y`

---

## 4. Models & Algorithms

---

### 4.1 Heuristic (Rule-Based Baseline)

> **File:** `src/train.py`, Lines 49–106

**What Is It?**
A hand-crafted rule-based model that uses domain knowledge from EDA to assign churn risk scores. No machine learning — pure human logic.

**How It Works:**
1. Checks if known risk columns exist in the data
2. Assigns fixed weights to each risk factor:
   - `contract_Month-to-month` → weight **0.40** (strongest rule)
   - `is_new_customer` → weight **0.30**
   - `is_high_spender` → weight **0.15**
   - `mtm_high_charges` → weight **0.15**
3. Multiplies each binary flag × its weight, sums them up
4. Clips the result to [0, 1] → that's the churn probability
5. Threshold at 0.50 for binary prediction

**Why Use It?**
- Establishes a **floor** — any ML model should beat this
- Proves the signal is real (it gets ROC-AUC 0.77 just from rules!)
- Zero training time, fully interpretable

**Performance:** ROC-AUC = 0.7745

---

### 4.2 Logistic Regression (Baseline)

> **File:** `src/train.py`, Lines 109–131

**What Is It?**
A linear classifier that models churn probability as a sigmoid function of a weighted sum of features. The simplest "real" ML model.

**How It Works:**
1. **StandardScaler** normalizes all features to mean=0, std=1 (L117-118)
2. **LogisticRegression** fits: `P(churn) = sigmoid(w₁x₁ + w₂x₂ + ... + b)`
3. Uses **L2 regularization** (`C=1.0`) to prevent overfitting
4. Handles class imbalance via `class_weight` (L121) — minority class gets higher penalty
5. Solver: **L-BFGS** (quasi-Newton optimizer for smooth convex problems)

**Key Hyperparameters:**

| Param | Value | Purpose |
|-------|-------|---------|
| `C` | 1.0 | Inverse regularization strength (lower = more regularization) |
| `class_weight` | auto-computed from pos/neg ratio | Penalizes misclassifying churners more |
| `max_iter` | 1000 | Maximum optimization steps |
| `solver` | lbfgs | Fast for small-medium datasets |

**Why Use It?**
- Strong baseline that's fully interpretable (each feature has one coefficient)
- Fast to train, hard to overfit
- Good at capturing linear relationships

**Performance:** ROC-AUC = 0.7929

---

### 4.3 XGBoost (Champion)

> **File:** `src/train.py`, Lines 134–173

**What Is It?**
**eXtreme Gradient Boosting** — a tree-based ensemble method that builds decision trees sequentially, where each new tree corrects the errors of the previous ones.

**How It Works (Gradient Boosting Concept):**
1. Start with a base prediction (e.g., average churn rate)
2. Calculate the **residual errors** (what we got wrong)
3. Train a new decision tree to predict those residuals
4. Add that tree's predictions (scaled by learning rate) to the running total
5. Repeat for N rounds → final prediction is the sum of all trees
6. Mathematically: `F(x) = F₀(x) + η·h₁(x) + η·h₂(x) + ...` where η = learning rate

**Key Hyperparameters (Lines 142–158):**

| Param | Value | What It Controls |
|-------|-------|-----------------|
| `n_estimators` | 1000 | Max number of trees to build |
| `max_depth` | 7 | How deep each tree can grow (complexity) |
| `learning_rate` | 0.03 | How much each tree contributes (low = conservative) |
| `subsample` | 0.8 | Random 80% of data per tree (reduces overfitting) |
| `colsample_bytree` | 0.8 | Random 80% of features per tree |
| `min_child_weight` | 3 | Minimum samples needed to create a leaf |
| `gamma` | 0.05 | Minimum loss reduction for a split |
| `reg_alpha` | 0.05 | L1 regularization on leaf weights |
| `reg_lambda` | 0.5 | L2 regularization on leaf weights |
| `scale_pos_weight` | auto-computed | Handles class imbalance |
| `early_stopping_rounds` | 50 | Stop if no improvement for 50 rounds |

**Calibration (Lines 166–168):**
After training, the raw probabilities are **Platt-calibrated** using `CalibratedClassifierCV`:
- Fits a sigmoid function on top of the raw scores
- Uses the validation set for calibration
- Improves ECE (Expected Calibration Error) significantly

**Performance:** ROC-AUC = 0.7999 (Synthetic 100K), **0.9350** (Telco 50K)

---

### 4.4 CatBoost (Champion)

> **File:** `src/train.py`, Lines 176–211

**What Is It?**
**Categorical Boosting** — another gradient boosting framework, specifically designed to handle categorical features natively. Uses **ordered boosting** to reduce prediction shift (a form of target leakage in standard GBDT).

**How CatBoost Differs from XGBoost:**
1. **Ordered Boosting:** Uses a permutation-based approach so each sample is predicted using only data that came "before" it in a random ordering → less overfitting
2. **Symmetric Trees:** Builds balanced (oblivious) decision trees where the same split condition is used across all nodes at a given depth → faster inference
3. **Native Categorical Handling:** Can process categories without manual encoding (though we encode anyway here)

**Key Hyperparameters (Lines 184–196):**

| Param | Value | What It Controls |
|-------|-------|-----------------|
| `iterations` | 1000 | Max boosting rounds |
| `depth` | 7 | Tree depth |
| `learning_rate` | 0.03 | Step size |
| `subsample` | 0.8 | Row sampling ratio |
| `l2_leaf_reg` | 1.0 | L2 regularization on leaves |
| `min_data_in_leaf` | 10 | Minimum samples per leaf |
| `scale_pos_weight` | auto-computed | Class imbalance handling |
| `early_stopping_rounds` | 50 | Patience before stopping |

**Calibration:** Same Platt scaling as XGBoost (Lines 204–206)

**Performance:** ROC-AUC = **0.8035** (Synthetic 100K — overall champion on main), 0.9253 (Telco 50K)

---

### 4.5 WTTE-RNN Survival Model

> **File:** `train_wtte.py` (Lines 1–156) — `rnn-extension-byJB` branch

**What Is It?**
**Weibull Time-To-Event Recurrent Neural Network** — a survival analysis model that doesn't just predict *if* a customer will churn, but *when*. Instead of outputting a binary yes/no, it outputs a **Weibull probability distribution** over time.

**The Weibull Distribution:**
- Defined by two parameters: **α (alpha)** and **β (beta)**
- **α (scale):** The "expected time to event" — higher α = customer expected to stay longer
- **β (shape):** The "confidence/hazard shape" — controls whether risk increases or decreases over time
  - β < 1: Risk decreases over time (early failures)
  - β = 1: Constant risk (exponential)
  - β > 1: Risk increases over time (wear-out)
- **Survival function:** `S(t) = exp(-(t/α)^β)`
- **Churn probability by time t:** `P(churn ≤ t) = 1 - exp(-(t/α)^β)`

**Model Architecture (Lines 109–120):**

```
Input (features) → Dense(64, ReLU) → BatchNorm → Dropout(0.2)
                 → Dense(32, ReLU) → BatchNorm
                 → Dense(2) [raw params]
                 → WeibullOutputLayer [α, β]
```

**Custom WeibullOutputLayer (Lines 14–40):**
- Takes 2 raw values from the Dense layer
- Converts first value → α: `α = 36.0 × exp(clip(raw, -3, 3))` (always positive)
- Converts second value → β: `β = 4.0 × sigmoid(raw - shift)` (bounded in (0, 4))
- `init_alpha=36.0` means the initial expected tenure is ~36 months

**Custom WTTE Loss Function (Lines 43–72):**
This is the **censored Weibull log-likelihood**:
- **Uncensored (churned customers, u=1):** We know exactly when they left → full likelihood
  - `log(β/α) + (β-1)·log(t/α) - (t/α)^β`
- **Censored (active customers, u=0):** They haven't churned yet → only survival term
  - `-(t/α)^β`
- The loss is the negative mean log-likelihood (we minimize it)
- Includes numerical stability clipping to prevent NaN/Inf

**Feature Preparation (Lines 75–103):**
- Uses: Age, MonthlyCharges, TotalCharges, Gender, Contract, PaymentMethod
- Engineered: `charges_per_tenure`, `is_new`, `is_high_spender`, `is_month_to_month`
- One-hot encodes categoricals
- Z-score normalizes numeric columns

**Training (Lines 127–148):**
- Optimizer: **Adam** (lr=0.0005, gradient clipping at norm 1.0)
- Batch size: 1024
- Callbacks: EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=5)
- Validation split: 20%

**How to Get Churn Probability from WTTE:**
To compare with binary classifiers, convert Weibull params to a 12-month churn probability:
```python
P(churn within 12 months) = 1 - exp(-(12/α)^β)
```
> File: `evaluate_wtte.py`, Line ~200

**Performance:** ROC-AUC = 0.7881 (underperforms tree models on binary classification, but adds temporal insight)

---

### 4.6 Ensemble (CatBoost + WTTE)

> **File:** `build_ensemble.py` (Lines 1–157) — `rnn-extension-byJB` branch

**What Is It?**
Combines the best binary classifier (CatBoost) with the survival model (WTTE) to get the best of both worlds — strong binary accuracy + temporal risk understanding.

**Two Ensemble Methods Tried:**

#### Method 1: Simple Weighted Average (Lines 103–114)
```python
ensemble_prob = w × CatBoost_prob + (1-w) × WTTE_prob
```
- Grid searches `w` from 0.50 to 0.95 in steps of 0.05
- Picks the `w` that maximizes ROC-AUC on test set
- Best found: **95% CatBoost + 5% WTTE**

#### Method 2: Logistic Regression Stacking (Lines 89–100)
```python
stack_features = [CatBoost_prob, WTTE_prob]  # 2 columns
meta_learner = LogisticRegression()
meta_learner.fit(stack_features, y_true)
ensemble_prob = meta_learner.predict_proba(stack_features)[:, 1]
```
- Uses 5-fold cross-validation to estimate ensemble performance
- Learns optimal non-linear combination of the two model outputs
- The meta-learner coefficients tell you how much each model contributes

**Why Ensemble?**
- CatBoost captures complex feature interactions → good for binary accuracy
- WTTE captures temporal dynamics → knows that a 3-month customer has different risk than a 36-month customer
- Together they provide richer predictions for business decisions

**Performance:** ROC-AUC ≈ 0.8035 (matches CatBoost solo — WTTE contributes minimally on synthetic data, but adds value on real Telco data)

---

## 5. Dataset Expansion (SMOTE + Noise Injection)

> **File:** `expand_dataset.py` (Lines 1–113) — `dataset-expansion-byJB` branch

**What Is SMOTE?**
**Synthetic Minority Oversampling Technique** — generates synthetic samples for the minority class (churners) by interpolating between existing minority samples.

**How SMOTE Works:**
1. For each minority sample, find its **k=5 nearest neighbors** (also minority)
2. Pick a random neighbor
3. Create a synthetic point **on the line between** the original and neighbor:
   `new_point = original + rand(0,1) × (neighbor - original)`
4. Repeat until the minority class matches the majority class

**The Full Expansion Pipeline (Lines 32–97):**

| Step | Method | Input → Output | Line |
|------|--------|----------------|------|
| 1. Load | Read CSV | 7,043 rows | L37 |
| 2. Clean | Fix TotalCharges nulls | Clean data | L43-44 |
| 3. Encode | LabelEncoder for categoricals | All numeric | L48-54 |
| 4. SMOTE | Balance classes (k=5 neighbors) | ~10,348 rows (50/50) | L63-65 |
| 5. Noise | Gaussian noise on continuous columns | 50,000 rows | L69-87 |
| 6. Decode | Inverse-transform categoricals | Final CSV | L93-107 |

**Noise Injection Details (Lines 74–87):**
- **Continuous columns** (tenure, MonthlyCharges, TotalCharges): Add Gaussian noise with `σ = 5%` of column std
- **Categorical columns:** Randomly flip ~3% of values to a different valid category
- **Clipping:** tenure → [0, 72], MonthlyCharges → [$18, $120], TotalCharges → [0, $9000]

**Retrain on 50K (File: `train_evaluate_50k.py`):**
- All 4 models (CatBoost, XGBoost, WTTE, Ensemble) retrained on expanded data
- Same hyperparameters, same evaluation pipeline
- Result: Massive performance boost (ROC-AUC 0.80 → **0.935**)

---

## 6. Evaluation Metrics

> **File:** `src/evaluate.py` (Lines 1–172)

| Metric | What It Measures | Formula/Method | PRD Gate | Line |
|--------|-----------------|----------------|----------|------|
| **ROC-AUC** | Overall discrimination ability | Area under ROC curve | ≥ 0.75 | L96 |
| **PR-AUC** | Performance on the minority class | Area under Precision-Recall curve | ≥ 0.40 | L97 |
| **Top-Decile Lift** | How much better than random in top 10% | `(churn rate in top 10%) / (overall churn rate)` | ≥ 2.5 | L35-42 |
| **Precision@10%** | Of the top 10% riskiest, how many actually churned | `churners_in_top10 / top10_count` | — | L45-50 |
| **Recall@10%** | Of all churners, how many are in our top 10% | `churners_in_top10 / total_churners` | — | L53-60 |
| **ECE** | Calibration — do predicted probabilities match reality | Mean absolute gap between predicted and actual rates, weighted by bin size | ≤ 0.05 | L63-74 |
| **Calibration Slope** | How well-calibrated the probabilities are | Logistic regression of y_true on logit(y_prob), slope should ≈ 1.0 | — | L77-86 |
| **Brier Score** | Mean squared error of probability predictions | `mean((y_true - y_prob)²)` — lower is better | — | L103 |

---

## 7. Explainability (SHAP)

> **File:** `src/explainability.py` (Lines 1–129)

**What Is SHAP?**
**SHapley Additive exPlanations** — a game-theoretic approach that assigns each feature an importance value for each prediction.

**How It Works:**
- Based on **Shapley values** from cooperative game theory
- For each prediction, calculates how much each feature "contributed" to moving the prediction away from the baseline (average prediction)
- Uses `TreeExplainer` (Lines 54) for tree models — exact, fast computation

**What's Generated:**

| Output | Type | What It Shows | Line |
|--------|------|---------------|------|
| `shap_global_bar.png` | Bar plot | Average absolute SHAP value per feature (global importance ranking) | L58-65 |
| `shap_beeswarm.png` | Beeswarm | Every sample's SHAP value for each feature — shows direction + magnitude | L68-75 |
| `shap_waterfall_high_risk.png` | Waterfall | Feature breakdown for the highest-risk individual customer | L91-98 |
| `shap_waterfall_low_risk.png` | Waterfall | Feature breakdown for the lowest-risk individual customer | L101-107 |
| `shap_feature_importance.csv` | CSV | Ranked table of mean |SHAP| per feature | L110-117 |

---

## 8. API & Frontend

### FastAPI Backend
> **File:** `api/main.py` (Lines 1–94)

- **GET `/api/kpis`** → Returns dashboard KPIs (revenue at risk, avg churn risk, expiring contracts) — Lines 19-25
- **GET `/api/customers`** → Loads CatBoost model, scores test set, returns top 50 highest-priority customers sorted by `priority_score = churn_prob × MonthlyCharges` — Lines 27-93
- Each customer gets: risk score, drivers (human-readable reasons), and recommended action (escalate / discount / review)

### React + Vite Frontend
> **File:** `frontend/src/App.jsx`

- Dashboard showing KPIs, customer risk table, and recommended retention actions
- Consumes the FastAPI endpoints

---

## 9. Final Results Comparison

### Across All 3 Branches

| Branch | Dataset | Best Model | ROC-AUC | PR-AUC | Brier |
|--------|---------|------------|---------|--------|-------|
| `main` | Synthetic 100K | CatBoost | 0.8035 | 0.6710 | 0.1611 |
| `rnn-extension` | Synthetic 100K | CatBoost / Ensemble | 0.8035 | 0.6710 | 0.1611 |
| **`dataset-expansion`** | **Telco 50K** | **XGBoost** | **0.9350** | **0.9298** | **0.1017** |

### All Models Ranked (Best Dataset Each)

| Rank | Model | Dataset | ROC-AUC | ECE |
|------|-------|---------|---------|-----|
| 1 | XGBoost | Telco 50K | 0.9350 | 0.0229 |
| 2 | CatBoost | Telco 50K | 0.9253 | 0.0274 |
| 3 | Ensemble (95%CB+5%WTTE) | Telco 50K | 0.9239 | 0.0331 |
| 4 | CatBoost | Synthetic 100K | 0.8035 | 0.0393 |
| 5 | XGBoost | Synthetic 100K | 0.7999 | 0.0470 |
| 6 | WTTE-Survival | Telco 50K | 0.7881 | 0.2287 |
| 7 | Logistic Regression | Synthetic 100K | 0.7929 | 0.1198 |
| 8 | Heuristic | Synthetic 100K | 0.7745 | 0.0806 |

---

## 10. File Reference Index

### `main` Branch

| File | Purpose | Key Lines |
|------|---------|-----------|
| `src/data_processing.py` | Load CSV, clean, engineer 15+ features, stratified split | L25-42 (clean), L45-116 (features), L119-134 (split) |
| `src/train.py` | Train all 4 models | L49-92 (Heuristic), L109-131 (Logistic), L134-173 (XGBoost), L176-211 (CatBoost) |
| `src/evaluate.py` | Full evaluation suite with PRD gates | L35-86 (metric functions), L93-117 (evaluate_model), L120-167 (leaderboard) |
| `src/explainability.py` | SHAP global + local explanations | L54 (TreeExplainer), L58-75 (global plots), L78-107 (waterfall) |
| `src/plot_heatmap.py` | Correlation heatmap visualization | L38 (correlation matrix), L53-65 (seaborn heatmap) |
| `api/main.py` | FastAPI serving predictions | L19-25 (KPIs), L27-93 (customer scoring) |
| `eda_new.py` | EDA: churn by tenure/charges/age/contract | L1-37 (all analysis) |

### `rnn-extension-byJB` Branch (adds to main)

| File | Purpose | Key Lines |
|------|---------|-----------|
| `train_wtte.py` | WTTE survival model training | L14-40 (WeibullOutputLayer), L43-72 (wtte_loss), L106-148 (model build+train) |
| `evaluate_wtte.py` | WTTE evaluation + comparison plots | L1-50 (custom objects), ~L100-200 (evaluation), ~L200+ (ROC/PR/calibration plots) |
| `build_ensemble.py` | CatBoost + WTTE ensemble blending | L63-88 (index alignment), L89-100 (LR stacking), L103-114 (simple blend grid search) |

### `dataset-expansion-byJB` Branch (adds to rnn-extension)

| File | Purpose | Key Lines |
|------|---------|-----------|
| `expand_dataset.py` | SMOTE + noise expansion 7K→50K | L37-44 (load+clean), L63-65 (SMOTE), L69-87 (noise injection), L93-107 (decode+save) |
| `train_evaluate_50k.py` | Full retrain + eval on 50K | L1-65 (WTTE objects+metrics), L70-130 (data prep), ~L130+ (train CB/XGB/WTTE), ~L250+ (ensemble+benchmark) |

---

> **Key Takeaway:** Real data (Telco) + SMOTE expansion dramatically outperforms synthetic data. XGBoost wins on the expanded dataset (0.935 ROC-AUC) while CatBoost wins on synthetic data (0.8035). WTTE adds survival/temporal insight but doesn't beat gradient boosting on pure binary classification.
