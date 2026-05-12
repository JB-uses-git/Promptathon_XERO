import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="AMC Retention Dashboard", layout="wide", page_icon="🛡️")

# --- Phase 4: Business Logic Layer (Intervention Engine) ---
def get_intervention(row):
    prob = row['churn_probability']
    val = row['contract_value_inr']
    complaints = row['unresolved_complaints']
    age = row['equipment_age_years']
    missed_visits = row['missed_scheduled_visits']
    
    if prob > 0.75:
        if complaints > 0 or missed_visits > 0:
            return "🔴 Service Escalation: Dispatch Senior Tech today to resolve pending issues. Offer 15% renewal discount."
        elif age >= 8:
            return "🔴 Upgrade Pitch: Unit is end-of-life. Sales call to pitch new AC installation with free 1-year AMC."
        elif val > 18000:
            return "🔴 High-Value At-Risk: Area Manager to personally call and schedule free proactive inspection."
        else:
            return "🔴 Aggressive Retention: SMS & Email campaign offering flat 20% off if renewed within 48 hours."
    elif prob > 0.5:
        if age >= 5:
            return "🟡 Pre-emptive Checkup: Schedule deep cleaning visit. Pitch 10% discount on multi-year renewal."
        else:
            return "🟡 personalized outreach: Send detailed service history report showing value delivered + renewal link."
    elif prob > 0.3:
        return "🟠 standard nudge: Automated WhatsApp reminder 30 days before expiry."
    else:
        return "🟢 auto-renewal flow: Standard Email reminder 15 days before expiry."

def get_outreach_action(prob):
    if prob > 0.75: return "📞 Priority Phone Call (Area Manager)"
    elif prob > 0.5: return "📞 Phone Call (Sales Rep)"
    elif prob > 0.3: return "💬 WhatsApp Notification"
    else: return "✉️ Standard Email"

def get_risk_label(prob):
    if prob > 0.75: return "🔴 Critical"
    elif prob > 0.5: return "🟡 High"
    elif prob > 0.3: return "🟠 Medium"
    else: return "🟢 Low"

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('processed_amc_data.csv')
    # add season
    df['expiry_month'] = pd.to_datetime(df['contract_end_date']).dt.month
    df['is_peak_cooling_season'] = df['expiry_month'].isin([4, 5])
    return df

@st.cache_resource
def load_models():
    xgb_model = joblib.load('xgboost_churn_model.joblib')
    features = joblib.load('feature_names.joblib')
    explainer = shap.TreeExplainer(xgb_model)
    return xgb_model, explainer, features

try:
    df = load_data()
    xgb_model, explainer, feature_names = load_models()
except FileNotFoundError:
    st.error("⚠️ Data or models not found. Run `python generate_amc_data.py` then `python train_pipeline.py` first.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading: {e}")
    st.stop()

# --- Apply Business Logic ---
df['intervention'] = df.apply(get_intervention, axis=1)
df['outreach_action'] = df['churn_probability'].apply(get_outreach_action)
df['risk_level'] = df['churn_probability'].apply(get_risk_label)

# --- Phase 6: Presentation Framing ---
st.markdown("""
<div style='background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); padding: 25px 30px; border-radius: 12px; margin-bottom: 25px; border-left: 6px solid #ff4b4b; color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <h3 style='margin-top: 0; color: #ffffff; font-weight: 600;'>🔥 The Demo Persona: "Rajesh, HVAC AMC Manager in Nagpur"</h3>
    <p style='font-size: 16px; line-height: 1.5; margin-bottom: 0;'>
        Rajesh manages 400+ commercial and residential HVAC contracts. It's Monday morning, one month before peak Indian summer. 
        Instead of guessing who might cancel their AMC, he opens this dashboard. <strong>In 30 seconds</strong>, he sees exactly which ₹8 Lakhs 
        are walking out the door, exactly <em>why</em> (e.g., 7-year old Daikin unit with 3 pending complaints), and exactly which technicians 
        need to be dispatched today to save those contracts.
    </p>
</div>
""", unsafe_allow_html=True)

st.title("🛡️ AMC Retention & Operations Command Center")

# --- Sidebar Filters ---
st.sidebar.header("🔎 Filters")
selected_city = st.sidebar.multiselect("City", options=sorted(df['city'].unique()), default=sorted(df['city'].unique()))
selected_brand = st.sidebar.multiselect("Equipment Brand", options=sorted(df['equipment_brand'].unique()), default=sorted(df['equipment_brand'].unique()))
selected_tier = st.sidebar.multiselect("Contract Tier", options=sorted(df['contract_tier'].unique()), default=sorted(df['contract_tier'].unique()))
risk_filter = st.sidebar.multiselect("Risk Level", options=["🔴 Critical", "🟡 High", "🟠 Medium", "🟢 Low"], default=["🔴 Critical", "🟡 High"])

filtered_df = df[
    (df['city'].isin(selected_city)) & 
    (df['equipment_brand'].isin(selected_brand)) & 
    (df['contract_tier'].isin(selected_tier)) &
    (df['risk_level'].isin(risk_filter))
].copy()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Revenue at Risk (Overview)", "👥 Customer Explainability", "🔧 Field Ops Planner", "📈 Model Insights"])

# ---- Tab 1: Overview ----
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    total_contracts = len(filtered_df)
    at_risk = filtered_df[filtered_df['churn_probability'] > 0.5]
    revenue_at_risk = at_risk['contract_value_inr'].sum()
    critical_count = len(filtered_df[filtered_df['churn_probability'] > 0.75])
    
    col1.metric("Contracts in View", f"{total_contracts:,}")
    col2.metric("💰 Revenue at Risk", f"₹ {revenue_at_risk:,.0f}", help="Sum of contract values for customers with >50% churn probability")
    col3.metric("🔴 Critical Flight Risks", f"{critical_count}")
    col4.metric("Avg Equipment Age", f"{filtered_df['equipment_age_years'].mean():.1f} yrs")
    
    st.markdown("---")
    st.markdown(f"### ⚠️ Losing these {len(at_risk)} high-risk customers = **₹ {revenue_at_risk:,.0f}** in lost revenue. Here's what to do:")
    
    display_cols = [
        'customer_id', 'customer_name', 'equipment_brand', 'equipment_age_years', 
        'contract_value_inr', 'churn_probability', 'intervention'
    ]
    
    sorted_df = filtered_df[display_cols].sort_values('churn_probability', ascending=False)
    
    st.dataframe(
        sorted_df.style.format({
            'churn_probability': '{:.1%}', 
            'contract_value_inr': '₹ {:,.0f}'
        }),
        use_container_width=True,
        hide_index=True,
        height=500
    )

# ---- Tab 2: Customer Deep Dive (SHAP Translation) ----
with tab2:
    st.markdown("### 🔍 Trust & Explainability: Why is this customer leaving?")
    
    customer_options = filtered_df.sort_values('churn_probability', ascending=False)['customer_id'].tolist()
    
    if not customer_options:
        st.warning("No customers match the selected filters.")
    else:
        selected_customer = st.selectbox("Select Customer ID to Analyze", customer_options)
        customer_row = df[df['customer_id'] == selected_customer].iloc[0]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Churn Risk (Confidence)", f"{customer_row['churn_probability']:.1%}")
        c2.metric("Contract Value", f"₹ {customer_row['contract_value_inr']:,.0f}")
        c3.metric("Unresolved Complaints", f"{int(customer_row['unresolved_complaints'])}")
        c4.metric("Equipment Age", f"{int(customer_row['equipment_age_years'])} yrs")
        
        st.success(f"**AI Prescribed Action:** {customer_row['intervention']}")
        st.info(f"**Outreach Channel:** {customer_row['outreach_action']}")
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### 🧠 AI Reasoning (Human Readable)")
            # Generate a text summary based on features
            reasons = []
            if customer_row['unresolved_complaints'] > 0:
                reasons.append(f"Customer has **{customer_row['unresolved_complaints']} unresolved complaints** pending.")
            if customer_row['equipment_age_years'] >= 8:
                reasons.append(f"Their {customer_row['equipment_brand']} AC is **{customer_row['equipment_age_years']} years old** (high replacement risk).")
            if customer_row['missed_scheduled_visits'] > 0:
                reasons.append(f"We missed **{customer_row['missed_scheduled_visits']} scheduled service visits**.")
            if customer_row['is_peak_cooling_season'] and customer_row['days_to_expiry'] < 30:
                reasons.append("Contract expires right before **peak summer**, highly vulnerable to competitor poaching.")
            if customer_row['previous_renewals'] == 0:
                reasons.append("First-year customer, no established loyalty yet.")
            
            if not reasons:
                reasons.append("Mix of minor factors (age, contract value, timing).")
                
            for i, r in enumerate(reasons):
                st.markdown(f"{i+1}. {r}")
                
        with col_right:
            st.markdown("#### 📊 Mathematical Proof (SHAP)")
            X_customer = pd.DataFrame([customer_row[feature_names]])
            for col in feature_names:
                X_customer[col] = pd.to_numeric(X_customer[col], errors='coerce')

            shap_values = explainer.shap_values(X_customer)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0], 
                    base_values=explainer.expected_value, 
                    data=X_customer.iloc[0].values, 
                    feature_names=feature_names
                ), 
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ---- Tab 3: Field Ops Planner ----
with tab3:
    st.markdown("### 🔧 HVAC Field Technician Workload Balancer")
    st.markdown("Prioritize physical visits for high-value customers at immediate risk of churn due to service failures.")
    
    ops_df = filtered_df[
        (filtered_df['churn_probability'] > 0.5) & 
        ((filtered_df['unresolved_complaints'] > 0) | (filtered_df['missed_scheduled_visits'] > 0))
    ].sort_values(['churn_probability', 'contract_value_inr'], ascending=[False, False])
    
    st.metric("Actionable Dispatches Today", len(ops_df))
    
    display_ops_cols = [
        'city', 'customer_id', 'customer_name', 'equipment_brand', 'unresolved_complaints', 
        'missed_scheduled_visits', 'days_since_last_service', 'churn_probability'
    ]
    
    st.dataframe(
        ops_df[display_ops_cols].style.format({'churn_probability': '{:.1%}', 'days_since_last_service': '{:.0f}'}),
        use_container_width=True,
        hide_index=True
    )

# ---- Tab 4: Model Insights ----
with tab4:
    st.markdown("### 📈 Machine Learning Transparency")
    st.markdown("""
    This system runs on two specialized models:
    1. **XGBoost Classifier**: Imbalanced learning (SMOTE/class weights) to predict binary churn probability.
    2. **Cox Proportional Hazards (Survival Analysis)**: Used by top telecom/SaaS companies to predict *time-to-churn*.
    """)
    
    importance = xgb_model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp_df['Feature'], imp_df['Importance'], color='#ff4b4b')
    ax.set_xlabel('Relative Importance (Gain)')
    ax.set_title('Global Feature Importance (What drives AMC churn?)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
