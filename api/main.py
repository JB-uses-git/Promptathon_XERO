from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
import numpy as np
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

# Register custom objects for Keras model loading
from keras.layers import Layer

@keras.saving.register_keras_serializable(package="Custom")
class WeibullOutputLayer(Layer):
    def __init__(self, init_alpha=36.0, max_beta_value=4.0, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = init_alpha
        self.max_beta_value = max_beta_value
    def call(self, x):
        a = x[..., 0]; b = x[..., 1]
        a = ops.clip(a, -3.0, 3.0)
        a = self.init_alpha * ops.exp(a)
        shift = float(np.log(self.max_beta_value - 1.0))
        b = self.max_beta_value * ops.sigmoid(b - shift)
        return ops.stack([a, b], axis=-1)
    def get_config(self):
        config = super().get_config()
        config.update({"init_alpha": self.init_alpha, "max_beta_value": self.max_beta_value})
        return config

@keras.saving.register_keras_serializable(package="Custom")
def wtte_loss(y_true, y_pred):
    y = tf.cast(y_true[..., 0], tf.float32)
    u = tf.cast(y_true[..., 1], tf.float32)
    a = tf.cast(y_pred[..., 0], tf.float32)
    b = tf.cast(y_pred[..., 1], tf.float32)
    eps = 1e-6
    a = tf.maximum(a, eps); b = tf.maximum(b, eps); y = tf.maximum(y, eps)
    ya = y / a
    log_ya = tf.math.log(ya + eps)
    survival = tf.clip_by_value(-tf.pow(ya, b), -50.0, 0.0)
    hazard = tf.clip_by_value(tf.math.log(b / a + eps) + (b - 1.0) * log_ya, -50.0, 50.0)
    loglik = u * hazard + survival
    return -tf.reduce_mean(loglik)


# Preload model and data at startup
_model = None
_df = None
_x_flat = None

def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(WTTE_DIR, "wtte_model.keras")
        _model = keras.models.load_model(model_path)
    return _model

def get_data():
    global _df, _x_flat
    if _df is None:
        _df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "synthetic_customer_churn_100k.csv"))
        _x_flat = np.load(os.path.join(WTTE_DIR, "x_flat.npy"))
    return _df, _x_flat


@app.get("/api/kpis")
def get_kpis():
    try:
        model = get_model()
        df, x_flat = get_data()
        
        preds = model.predict(x_flat, verbose=0)
        alphas = preds[:, 0]
        betas = preds[:, 1]
        
        # 6-month churn risk using Weibull CDF
        t = 6
        risk_probs = 1.0 - np.exp(-np.power(t / alphas, betas))
        
        # Revenue at risk: sum of MonthlyCharges for high-risk customers
        high_risk_mask = risk_probs > 0.5
        rev_at_risk = df.loc[high_risk_mask, 'MonthlyCharges'].sum()
        
        avg_risk = float(np.mean(risk_probs) * 100)
        high_risk_count = int(np.sum(high_risk_mask))
        
        return {
            "revenue_at_risk": f"${rev_at_risk/1000:.0f}K",
            "avg_churn_risk": f"{avg_risk:.1f}%",
            "expiring_30_days": f"{high_risk_count:,}"
        }
    except Exception as e:
        return {
            "revenue_at_risk": "Error",
            "avg_churn_risk": str(e),
            "expiring_30_days": "0"
        }


@app.get("/api/customers")
def get_customers():
    try:
        model = get_model()
        df, x_flat = get_data()
        
        preds = model.predict(x_flat, verbose=0)
        alphas = preds[:, 0]
        betas = preds[:, 1]
        
        # 12-month churn risk (gives a wider, more realistic spread)
        t = 12
        risk_probs = 1.0 - np.exp(-np.power(t / alphas, betas))
        
        result_df = df.copy()
        result_df['alpha'] = alphas
        result_df['beta'] = betas
        result_df['churn_prob'] = risk_probs
        result_df['priority_score'] = risk_probs * result_df['MonthlyCharges']
        
        # Select a diverse mix of risk levels for the dashboard
        high = result_df[result_df['churn_prob'] > 0.7].nlargest(15, 'priority_score')
        med = result_df[(result_df['churn_prob'] > 0.3) & (result_df['churn_prob'] <= 0.7)].nlargest(20, 'priority_score')
        low = result_df[(result_df['churn_prob'] > 0.05) & (result_df['churn_prob'] <= 0.3)].nlargest(15, 'priority_score')
        result_df = pd.concat([high, med, low]).sort_values(by='priority_score', ascending=False)
        
        customers = []
        for idx, row in result_df.iterrows():
            drivers = []
            if row['Contract'] == 'Month-to-month':
                drivers.append("Month-to-month contract")
            if row['MonthlyCharges'] > 100:
                drivers.append("High monthly spend (>$100)")
            if row['Tenure'] <= 6:
                drivers.append("New customer (<6 months)")
            
            expected_months = max(1, int(row['alpha']))
            
            if expected_months <= 6:
                drivers.append(f"WTTE Alert: Expected lifespan ~{expected_months} months")
            elif expected_months <= 12:
                drivers.append(f"WTTE Alert: Moderate risk, ~{expected_months} months expected")
                
            if row['churn_prob'] > 0.85:
                action = "Escalate to Retention Specialist immediately"
            elif row['churn_prob'] > 0.7:
                action = f"Offer Loyalty Discount within {expected_months} months"
            elif row['churn_prob'] > 0.5:
                action = "Schedule Proactive Check-in Call"
            else:
                action = "Monitor Usage Patterns"
                
            customers.append({
                "id": f"CUST-{row['CustomerID']}",
                "risk_score": float(row['churn_prob']),
                "monthly_charges": float(row['MonthlyCharges']),
                "priority_score": float(row['priority_score']),
                "drivers": drivers,
                "recommended_action": action,
                "expected_days": expected_months * 30,  # Convert to days for frontend
                "confidence": float(row['beta'])
            })
            
        return customers
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []
