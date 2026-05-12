import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# --- Set Page Config ---
st.set_page_config(page_title="AMC Retention Dashboard", layout="wide")

# --- Phase 4: Business Logic Layer ---
def get_intervention(churn_prob, contract_value, unresolved_complaints):
    if churn_prob > 0.75 and contract_value > 15000:
        return "Escalate to manager — schedule free inspection call"
    elif churn_prob > 0.75 and unresolved_complaints > 2:
        return "Resolve complaints first, then offer 10% renewal discount"
    elif churn_prob > 0.5:
        return "Send personalized renewal reminder with service report"
    else:
        return "Standard renewal flow"

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('processed_amc_data.csv')
    return df

@st.cache_resource
def load_models():
    xgb_model = joblib.load('xgboost_churn_model.joblib')
    X_train_ref = pd.read_csv('X_train_reference.csv')
    explainer = shap.TreeExplainer(xgb_model)
    return xgb_model, explainer, X_train_ref

try:
    df = load_data()
    xgb_model, explainer, X_train_ref = load_models()
except Exception as e:
    st.error("Data or models not found. Please run data generation and pipeline first.")
    st.stop()

# --- Apply Business Logic ---
df['intervention'] = df.apply(lambda row: get_intervention(row['churn_probability'], row['contract_value_inr'], row['unresolved_complaints']), axis=1)

# --- Phase 6: Presentation Framing ---
st.markdown("""
<div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #2e86c1;'>
<strong>💡 Demo Context:</strong> "An AMC manager handling contracts in Nagpur opens this dashboard Monday morning. 
In 30 seconds he knows which customers are about to leave, why they're leaving, and exactly what his team should do today to retain high-value revenue."
</div>
""", unsafe_allow_html=True)

st.title("🛡️ AMC Retention & Churn Command Center")

# --- Filters ---
st.sidebar.header("Filters")
selected_city = st.sidebar.multiselect("City", options=df['city'].unique(), default=df['city'].unique())
selected_brand = st.sidebar.multiselect("Equipment Brand", options=df['equipment_brand'].unique(), default=df['equipment_brand'].unique())
selected_tier = st.sidebar.multiselect("Contract Tier", options=df['contract_tier'].unique(), default=df['contract_tier'].unique())

filtered_df = df[
    (df['city'].isin(selected_city)) & 
    (df['equipment_brand'].isin(selected_brand)) & 
    (df['contract_tier'].isin(selected_tier))
].copy()

# --- Phase 5: Overview Tab ---
tab1, tab2 = st.tabs(["📊 Overview", "👥 Customer Details"])

with tab1:
    col1, col2, col3 = st.columns(3)
    
    total_contracts = len(filtered_df)
    at_risk_df = filtered_df[filtered_df['churn_probability'] > 0.5]
    revenue_at_risk = at_risk_df['contract_value_inr'].sum()
    avg_churn_prob = filtered_df['churn_probability'].mean()
    
    col1.metric("Total Contracts", f"{total_contracts}")
    col2.metric("Revenue at Risk", f"₹ {revenue_at_risk:,.0f}")
    col3.metric("Avg Churn Risk", f"{avg_churn_prob:.1%}")
    
    st.markdown("### ⚠️ Customers sorted by Churn Risk")
    
    display_cols = ['customer_id', 'customer_name', 'city', 'contract_tier', 'contract_value_inr', 'churn_probability', 'intervention']
    
    def color_risk(val):
        color = '#a5d6a7' # green
        if val > 0.75: color = '#ef9a9a' # red
        elif val > 0.5: color = '#ffe082' # orange
        return f'background-color: {color}; color: black;'
        
    st.dataframe(
        filtered_df[display_cols].sort_values('churn_probability', ascending=False)
        .style.map(color_risk, subset=['churn_probability'])
        .format({'churn_probability': '{:.1%}', 'contract_value_inr': '₹ {:,.0f}'}),
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.markdown("### 🔍 Deep Dive: Customer Explainability")
    
    selected_customer = st.selectbox("Select Customer ID", filtered_df.sort_values('churn_probability', ascending=False)['customer_id'])
    
    if selected_customer:
        customer_data = df[df['customer_id'] == selected_customer].iloc[0]
        st.subheader(f"{customer_data['customer_name']} ({selected_customer})")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{customer_data['churn_probability']:.1%}")
        c2.metric("Contract Value", f"₹ {customer_data['contract_value_inr']:,}")
        c3.metric("Unresolved Complaints", f"{customer_data['unresolved_complaints']}")
        
        st.info(f"**Recommended Action:** {customer_data['intervention']}")
        
        st.markdown("#### Why is this customer at risk? (SHAP Explanation)")
        
        features = [
            'contract_value_inr', 'equipment_age_years', 'total_service_calls',
            'avg_resolution_time_days', 'unresolved_complaints', 'missed_scheduled_visits',
            'repeat_complaints', 'previous_renewals', 'days_to_expiry', 
            'renewal_reminder_sent', 'last_renewal_delay_days',
            'contract_duration_days', 'service_call_rate', 'days_since_last_service',
            'renewal_loyalty_score',
            'city_encoded', 'equipment_brand_encoded', 'contract_tier_encoded', 'equipment_type_encoded'
        ]
        
        X_customer = pd.DataFrame([customer_data[features]])
        
        # Ensure datatypes match X_train_ref
        for col in features:
            X_customer[col] = pd.to_numeric(X_customer[col])

        shap_values = explainer.shap_values(X_customer)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                             base_values=explainer.expected_value, 
                                             data=X_customer.iloc[0], 
                                             feature_names=features), show=False)
        st.pyplot(fig)
