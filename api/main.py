from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import keras
import keras.ops as ops

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
WTTE_DIR = os.path.join(ARTIFACTS_DIR, "wtte_data")

# ── Register custom objects for WTTE model loading ──
from keras.layers import Layer

@keras.saving.register_keras_serializable(package="Custom")
class WeibullOutputLayer(Layer):
    def __init__(self, init_alpha=36.0, max_beta_value=4.0, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = init_alpha; self.max_beta_value = max_beta_value
    def call(self, x):
        a = x[..., 0]; b = x[..., 1]
        a = ops.clip(a, -3.0, 3.0); a = self.init_alpha * ops.exp(a)
        shift = float(np.log(self.max_beta_value - 1.0))
        b = self.max_beta_value * ops.sigmoid(b - shift)
        return ops.stack([a, b], axis=-1)
    def get_config(self):
        c = super().get_config()
        c.update({"init_alpha": self.init_alpha, "max_beta_value": self.max_beta_value})
        return c

@keras.saving.register_keras_serializable(package="Custom")
def wtte_loss(y_true, y_pred):
    y = tf.cast(y_true[..., 0], tf.float32); u = tf.cast(y_true[..., 1], tf.float32)
    a = tf.cast(y_pred[..., 0], tf.float32); b = tf.cast(y_pred[..., 1], tf.float32)
    eps = 1e-6; a = tf.maximum(a, eps); b = tf.maximum(b, eps); y = tf.maximum(y, eps)
    ya = y / a; log_ya = tf.math.log(ya + eps)
    survival = tf.clip_by_value(-tf.pow(ya, b), -50.0, 0.0)
    hazard = tf.clip_by_value(tf.math.log(b / a + eps) + (b - 1.0) * log_ya, -50.0, 50.0)
    return -tf.reduce_mean(u * hazard + survival)


# ── Cached model/data loading ──
_cache = {}

def get_models():
    if 'loaded' not in _cache:
        # CatBoost (the accuracy champ)
        _cache['catboost'] = joblib.load(os.path.join(ARTIFACTS_DIR, "model_catboost.joblib"))
        # WTTE (the "when" predictor)
        _cache['wtte'] = keras.models.load_model(os.path.join(WTTE_DIR, "wtte_model.keras"))
        # Ensemble config
        _cache['ensemble'] = joblib.load(os.path.join(ARTIFACTS_DIR, "ensemble_config.joblib"))
        # Data splits (for CatBoost features)
        _cache['splits'] = joblib.load(os.path.join(ARTIFACTS_DIR, "splits.joblib"))
        # WTTE flat features
        _cache['x_flat'] = np.load(os.path.join(WTTE_DIR, "x_flat.npy"))
        # Raw dataset
        _cache['df'] = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "synthetic_customer_churn_100k.csv"))
        _cache['loaded'] = True
    return _cache


@app.get("/api/kpis")
def get_kpis():
    try:
        c = get_models()
        df = c['df']
        catboost = c['catboost']
        wtte = c['wtte']
        x_flat = c['x_flat']
        X_train, X_val, X_test, y_train, y_val, y_test = c['splits']
        ensemble_cfg = c['ensemble']
        cb_w = ensemble_cfg['catboost_weight']
        wtte_w = ensemble_cfg['wtte_weight']
        
        # CatBoost probs on test set
        cb_probs = catboost.predict_proba(X_test)[:, 1]
        
        # WTTE probs on test set (align by index)
        wtte_preds = wtte.predict(x_flat, verbose=0)
        alphas_all = wtte_preds[:, 0]
        betas_all = wtte_preds[:, 1]
        wtte_risk_all = np.clip(1.0 - np.exp(-np.power(12.0 / alphas_all, betas_all)), 0, 1)
        test_indices = X_test.index.values
        wtte_probs = wtte_risk_all[test_indices]
        
        # Ensemble blend
        ensemble_probs = cb_w * cb_probs + wtte_w * wtte_probs
        
        high_risk_mask = ensemble_probs > 0.5
        test_df = df.iloc[test_indices]
        rev_at_risk = test_df.loc[high_risk_mask, 'MonthlyCharges'].sum()
        avg_risk = float(np.mean(ensemble_probs) * 100)
        high_risk_count = int(np.sum(high_risk_mask))
        
        return {
            "revenue_at_risk": f"${rev_at_risk/1000:.0f}K",
            "avg_churn_risk": f"{avg_risk:.1f}%",
            "expiring_30_days": f"{high_risk_count:,}"
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"revenue_at_risk": "Error", "avg_churn_risk": str(e), "expiring_30_days": "0"}


@app.get("/api/customers")
def get_customers():
    try:
        c = get_models()
        df = c['df']
        catboost = c['catboost']
        wtte = c['wtte']
        x_flat = c['x_flat']
        X_train, X_val, X_test, y_train, y_val, y_test = c['splits']
        ensemble_cfg = c['ensemble']
        cb_w = ensemble_cfg['catboost_weight']
        wtte_w = ensemble_cfg['wtte_weight']
        
        # CatBoost probs
        cb_probs = catboost.predict_proba(X_test)[:, 1]
        
        # WTTE predictions
        wtte_preds = wtte.predict(x_flat, verbose=0)
        alphas_all = wtte_preds[:, 0]
        betas_all = wtte_preds[:, 1]
        wtte_risk_all = np.clip(1.0 - np.exp(-np.power(12.0 / alphas_all, betas_all)), 0, 1)
        
        test_indices = X_test.index.values
        wtte_probs = wtte_risk_all[test_indices]
        wtte_alphas = alphas_all[test_indices]
        
        # Ensemble blend
        ensemble_probs = cb_w * cb_probs + wtte_w * wtte_probs
        
        # Build result dataframe
        result_df = df.iloc[test_indices].copy()
        result_df['cb_prob'] = cb_probs
        result_df['wtte_prob'] = wtte_probs
        result_df['ensemble_prob'] = ensemble_probs
        result_df['alpha'] = wtte_alphas
        result_df['priority_score'] = ensemble_probs * result_df['MonthlyCharges']
        
        # Diverse mix of risk levels
        high = result_df[result_df['ensemble_prob'] > 0.7].nlargest(15, 'priority_score')
        med = result_df[(result_df['ensemble_prob'] > 0.3) & (result_df['ensemble_prob'] <= 0.7)].nlargest(20, 'priority_score')
        low = result_df[(result_df['ensemble_prob'] > 0.05) & (result_df['ensemble_prob'] <= 0.3)].nlargest(15, 'priority_score')
        result_df = pd.concat([high, med, low]).sort_values(by='priority_score', ascending=False)
        
        customers = []
        for idx, row in result_df.iterrows():
            drivers = []
            
            # Feature-based drivers
            if row['Contract'] == 'Month-to-month':
                drivers.append("Month-to-month contract")
            if row['MonthlyCharges'] > 100:
                drivers.append("High monthly spend (>$100)")
            if row['Tenure'] <= 6:
                drivers.append("New customer (<6 months)")
            
            # WTTE survival driver
            expected_months = max(1, int(row['alpha']))
            if expected_months <= 6:
                drivers.append(f"Survival Alert: ~{expected_months} months expected lifespan")
            elif expected_months <= 18:
                drivers.append(f"Survival Watch: ~{expected_months} months expected lifespan")
            
            # CatBoost vs WTTE agreement driver
            cb_risk = row['cb_prob']
            wtte_risk = row['wtte_prob']
            if cb_risk > 0.5 and wtte_risk > 0.5:
                drivers.append("Both models agree: HIGH risk")
            elif cb_risk > 0.5 and wtte_risk <= 0.3:
                drivers.append("CatBoost flags risk, WTTE says stable")
            
            # Action based on ensemble
            if row['ensemble_prob'] > 0.85:
                action = "Escalate to Retention Specialist immediately"
            elif row['ensemble_prob'] > 0.7:
                action = f"Offer Loyalty Discount within {expected_months} months"
            elif row['ensemble_prob'] > 0.5:
                action = "Schedule Proactive Check-in Call"
            elif row['ensemble_prob'] > 0.3:
                action = "Monitor & Send Engagement Email"
            else:
                action = "Low Risk — Standard Monitoring"
            
            customers.append({
                "id": f"CUST-{row['CustomerID']}",
                "risk_score": float(row['ensemble_prob']),
                "monthly_charges": float(row['MonthlyCharges']),
                "priority_score": float(row['priority_score']),
                "drivers": drivers,
                "recommended_action": action,
                "expected_days": expected_months * 30,
                "confidence": float(row['cb_prob']),  # CatBoost confidence as secondary metric
            })
        
        return customers
    except Exception as e:
        import traceback; traceback.print_exc()
        return []
