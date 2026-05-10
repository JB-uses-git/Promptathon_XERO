import numpy as np
import os
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Dropout, Layer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras.ops as ops
import pandas as pd
import joblib


@keras.saving.register_keras_serializable(package="Custom")
class WeibullOutputLayer(Layer):
    """Custom layer that converts raw dense output into Weibull alpha/beta."""
    def __init__(self, init_alpha=36.0, max_beta_value=4.0, **kwargs):
        super().__init__(**kwargs)
        self.init_alpha = init_alpha
        self.max_beta_value = max_beta_value
        
    def call(self, x):
        a = x[..., 0]
        b = x[..., 1]
        
        # Clamp raw values to prevent overflow
        a = ops.clip(a, -3.0, 3.0)
        a = self.init_alpha * ops.exp(a)
        
        # Beta: sigmoid keeps it bounded in (0, max_beta_value)
        shift = float(np.log(self.max_beta_value - 1.0))
        b = self.max_beta_value * ops.sigmoid(b - shift)
        
        return ops.stack([a, b], axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "init_alpha": self.init_alpha,
            "max_beta_value": self.max_beta_value
        })
        return config


@keras.saving.register_keras_serializable(package="Custom")
def wtte_loss(y_true, y_pred):
    """Continuous censored Weibull log-likelihood loss (numerically stable)."""
    # Use tf directly here because loss functions run eagerly in TF backend
    y = tf.cast(y_true[..., 0], tf.float32)  # TTE (tenure in months)
    u = tf.cast(y_true[..., 1], tf.float32)  # Event (1=churned, 0=censored)
    a = tf.cast(y_pred[..., 0], tf.float32)  # Alpha
    b = tf.cast(y_pred[..., 1], tf.float32)  # Beta
    
    eps = 1e-6
    a = tf.maximum(a, eps)
    b = tf.maximum(b, eps)
    y = tf.maximum(y, eps)
    
    # Continuous Weibull log-likelihood:
    # For uncensored (u=1): log(b/a) + (b-1)*log(y/a) - (y/a)^b
    # For censored   (u=0): -(y/a)^b
    # Combined: u * [log(b/a) + (b-1)*log(y/a)] - (y/a)^b
    
    ya = y / a
    log_ya = tf.math.log(ya + eps)
    
    # The survival term (always present)
    survival = -tf.pow(ya, b)
    survival = tf.clip_by_value(survival, -50.0, 0.0)
    
    # The hazard term (only for events)
    hazard = tf.math.log(b / a + eps) + (b - 1.0) * log_ya
    hazard = tf.clip_by_value(hazard, -50.0, 50.0)
    
    loglik = u * hazard + survival
    
    return -tf.reduce_mean(loglik)


def prepare_features():
    """Load and prepare features from the synthetic dataset."""
    print("Loading synthetic_customer_churn_100k.csv...")
    df = pd.read_csv("synthetic_customer_churn_100k.csv")
    
    tte = df['Tenure'].values.astype(np.float32)
    events = (df['Churn'] == 'Yes').astype(np.float32).values
    
    features_df = df[['Age', 'MonthlyCharges', 'TotalCharges', 'Gender', 'Contract', 'PaymentMethod']].copy()
    
    features_df['charges_per_tenure'] = df['TotalCharges'] / (df['Tenure'] + 1)
    features_df['is_new'] = (df['Tenure'] <= 6).astype(float)
    features_df['is_high_spender'] = (df['MonthlyCharges'] > 100).astype(float)
    features_df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(float)
    
    features_encoded = pd.get_dummies(features_df, columns=['Gender', 'Contract', 'PaymentMethod'], drop_first=True)
    
    numeric_cols = ['Age', 'MonthlyCharges', 'TotalCharges', 'charges_per_tenure']
    norm_stats = {}
    for col in numeric_cols:
        mean = features_encoded[col].mean()
        std = features_encoded[col].std()
        norm_stats[col] = {'mean': float(mean), 'std': float(std)}
        features_encoded[col] = (features_encoded[col] - mean) / (std + 1e-7)
    
    x = features_encoded.values.astype(np.float32)
    
    print(f"Features: {x.shape[1]}, Samples: {x.shape[0]}")
    print(f"TTE range: {tte.min():.0f}-{tte.max():.0f} months, mean: {tte.mean():.1f}")
    print(f"Churn rate: {events.mean()*100:.1f}%")
    
    out_dir = "artifacts/wtte_data"
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(features_encoded.columns.tolist(), os.path.join(out_dir, 'feature_names.joblib'))
    joblib.dump(norm_stats, os.path.join(out_dir, 'norm_stats.joblib'))
    np.save(os.path.join(out_dir, 'x_flat.npy'), x)
    np.save(os.path.join(out_dir, 'y_tte.npy'), tte)
    np.save(os.path.join(out_dir, 'y_events.npy'), events)
    
    return x, tte, events


def train_model():
    print("=" * 60)
    print("WTTE Survival Model — Feedforward + Weibull Head")
    print("=" * 60)
    
    x, tte, events = prepare_features()
    
    y_true = np.stack([tte, events], axis=-1)
    num_features = x.shape[1]
    
    # Build model
    inputs = Input(shape=(num_features,), name="features")
    h = Dense(64, activation='relu')(inputs)
    h = BatchNormalization()(h)
    h = Dropout(0.2)(h)
    h = Dense(32, activation='relu')(h)
    h = BatchNormalization()(h)
    raw_ab = Dense(2, name="raw_params")(h)
    ab = WeibullOutputLayer(init_alpha=36.0, max_beta_value=4.0, name="weibull_output")(raw_ab)
    
    model = Model(inputs=inputs, outputs=ab)
    
    optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(loss=wtte_loss, optimizer=optimizer)
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    print("\nTraining...")
    history = model.fit(
        x, y_true,
        epochs=100,
        batch_size=1024,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Validate
    preds = model.predict(x[:2000])
    alphas = preds[:, 0]
    betas = preds[:, 1]
    
    print(f"\n{'='*60}")
    print(f"Post-Training Prediction Stats (first 2000 customers)")
    print(f"{'='*60}")
    print(f"Alpha (expected months): min={alphas.min():.1f}, max={alphas.max():.1f}, mean={alphas.mean():.1f}")
    print(f"Beta  (shape/confidence): min={betas.min():.2f}, max={betas.max():.2f}, mean={betas.mean():.2f}")
    
    for t in [3, 6, 12, 24]:
        risk = 1.0 - np.exp(-np.power(t / alphas, betas))
        high_risk = np.sum(risk > 0.5)
        print(f"  Churn risk within {t:2d} months: min={risk.min():.1%}, max={risk.max():.1%}, mean={risk.mean():.1%}, high-risk(>50%)={high_risk}")
    
    model_path = os.path.join("artifacts/wtte_data", "wtte_model.keras")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_model()
