import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import altair as alt
import os
from google import genai

# --- Page Config ---
st.set_page_config(page_title="AMC Retention Dashboard", layout="wide")

# --- Custom Premium UI Styling ---
st.markdown("""
<style>
    /* Import modern Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Clean up the default Streamlit look (hide hamburger menu and footer) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Beautiful Metric Cards with hover effects */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
        border-color: #cbd5e1;
    }
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 15px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Premium Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 15px;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -1px rgba(37, 99, 235, 0.4);
        color: white;
    }

    /* Elegant Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0;
        padding: 12px 20px;
        font-weight: 600;
        font-size: 16px;
        color: #64748b;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb;
        border-bottom: 3px solid #2563eb !important;
        background-color: transparent;
    }

    /* Polished Search Bar */
    .stTextInput input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 14px 20px;
        font-size: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: all 0.2s ease;
    }
    .stTextInput input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.1);
    }

    /* DataFrame adjustments */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- Phase 4: Business Logic Layer (Intervention Engine) ---
def get_intervention(row):
    prob = row['churn_probability']
    val = row['contract_value_inr']
    complaints = row['unresolved_complaints']
    age = row['equipment_age_years']
    missed_visits = row['missed_scheduled_visits']
    
    if prob > 0.75:
        if complaints > 0 or missed_visits > 0:
            return "Service Escalation: Dispatch Senior Tech today to resolve pending issues. Offer 15% renewal discount."
        elif age >= 8:
            return "Upgrade Pitch: Unit is end-of-life. Sales call to pitch new AC installation with free 1-year AMC."
        elif val > 18000:
            return "High-Value At-Risk: Area Manager to personally call and schedule free proactive inspection."
        else:
            return "Aggressive Retention: SMS & Email campaign offering flat 20% off if renewed within 48 hours."
    elif prob > 0.5:
        if age >= 5:
            return "Pre-emptive Checkup: Schedule deep cleaning visit. Pitch 10% discount on multi-year renewal."
        else:
            return "Personalized Outreach: Send detailed service history report showing value delivered + renewal link."
    elif prob > 0.3:
        return "Standard Nudge: Automated WhatsApp reminder 30 days before expiry."
    else:
        return "Auto-Renewal Flow: Standard Email reminder 15 days before expiry."

def get_outreach_action(prob):
    if prob > 0.75: return "Priority Phone Call (Area Manager)"
    elif prob > 0.5: return "Phone Call (Sales Rep)"
    elif prob > 0.3: return "WhatsApp Notification"
    else: return "Standard Email"

def get_risk_label(prob):
    if prob > 0.75: return "Critical"
    elif prob > 0.5: return "High"
    elif prob > 0.3: return "Medium"
    else: return "Low"

# --- Gemini Retention Message Generator ---
def generate_retention_message(customer: dict, channel: str) -> str:
    """Calls Google Gemini to generate a personalized retention outreach message."""
    # Try getting from Streamlit secrets first, then fallback to environment variables
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key:
        return "ERROR: GEMINI_API_KEY is not set. Please set it in .streamlit/secrets.toml or as an environment variable."

    client = genai.Client(api_key=api_key)

    # Channel-specific tone and length instructions
    channel_instructions = {
        "WhatsApp": "Write in a casual, warm, friendly tone suitable for a WhatsApp message. Keep it under 120 words. Use short paragraphs. Do NOT use subject lines.",
        "Email": "Write in a professional, polished business-email tone. Keep it under 200 words. Include a subject line at the top prefixed with 'Subject:'.",
        "Formal Letter": "Write in a formal business-letter tone with proper salutation and sign-off. Keep it under 300 words. Include date, addressee header, and sender sign-off as 'AMC Services Team'."
    }

    prompt = f"""You are a senior customer retention specialist at a leading HVAC AMC (Annual Maintenance Contract) company in India.

Generate a personalized retention outreach message for the following customer. Use their ACTUAL data values directly in the message — do NOT use any placeholders like [Name] or {{brand}}.

Customer Details:
- Name: {customer.get('customer_name', 'Valued Customer')}
- City: {customer.get('city', 'N/A')}
- Equipment Brand: {customer.get('equipment_brand', 'N/A')}
- Equipment Type: {customer.get('equipment_type', 'AC')}
- Equipment Age: {customer.get('equipment_age_years', 'N/A')} years
- Contract Tier: {customer.get('contract_tier', 'Standard')}
- Contract Value: ₹{customer.get('contract_value_inr', 0):,.0f}
- Contract Expiry: {customer.get('contract_end_date', 'N/A')}
- Days Until Expiry: {customer.get('days_to_expiry', 'N/A')}
- Previous Renewals: {customer.get('previous_renewals', 0)}
- Unresolved Complaints: {int(customer.get('unresolved_complaints', 0))}
- Missed Scheduled Visits: {int(customer.get('missed_scheduled_visits', 0))}
- Historical Renewal Rate: {customer.get('renewal_rate', 0):.0%}
- Churn Risk Score: {customer.get('churn_probability', 0):.0%}

Channel: {channel}
{channel_instructions.get(channel, channel_instructions['Email'])}

Rules:
1. Address the customer by their actual name.
2. Mention their specific equipment brand and age naturally in the message.
3. Reference their contract expiry date or days remaining.
4. If they have unresolved complaints or missed visits, acknowledge and apologize.
5. If their historical renewal rate is below 50%, prominently offer a 15% discount or a free priority service visit.
6. Include a soft, non-pushy call to action (e.g., schedule a call, renew online, reply to this message).
7. Do NOT invent any data not provided above.
8. Output ONLY the message — no commentary, no notes, no meta-text.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"ERROR: Failed to generate message — {str(e)}"

# --- Natural Language Query Engine ---
def nl_to_pandas_filter(query: str) -> str:
    """Sends a plain English query to Gemini and returns a pandas boolean filter expression."""
    # Try getting from Streamlit secrets first, then fallback to environment variables
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key:
        return "ERROR: GEMINI_API_KEY not set."

    client = genai.Client(api_key=api_key)

    prompt = f"""You are a pandas expert. You will receive a plain English query about a dataframe called `df`.

The dataframe has these columns and types:
- customer_id (str)
- customer_name (str)
- city (str)
- contract_start_date (str, date format)
- contract_end_date (str, date format)
- contract_value_inr (float)
- contract_tier (str: 'Basic', 'Standard', 'Premium')
- equipment_brand (str)
- equipment_age_years (float)
- equipment_type (str)
- total_service_calls (int)
- avg_resolution_time_days (float)
- unresolved_complaints (int)
- last_service_date (str, date format)
- missed_scheduled_visits (int)
- repeat_complaints (int, 0 or 1)
- previous_renewals (int)
- days_to_expiry (int)
- renewal_reminder_sent (int, 0 or 1)
- last_renewal_delay_days (int)
- churn (int, 0 or 1)
- churn_reason (str)
- estimated_churn_date (str, date format)
- churn_probability (float, 0.0 to 1.0)

User query: "{query}"

Return ONLY a valid pandas boolean filter expression that can be used as `df[<expression>]`.
Rules:
1. Output ONLY the expression — no markdown, no backticks, no explanation, no code block fences.
2. Use proper pandas syntax (e.g., df['col'] > value, df['col'].str.contains('x'), df['col'].isin([...])).
3. For string comparisons, use .str.lower() for case-insensitive matching.
4. Combine conditions with & (and) or | (or), wrapping each in parentheses.
5. Do NOT include `df[` wrapper or `]` — just the boolean expression itself.
6. If the query is ambiguous, make a reasonable assumption.

Examples:
Query: "customers in Mumbai" → (df['city'].str.lower() == 'mumbai')
Query: "churn probability above 80%" → (df['churn_probability'] > 0.8)
Query: "premium contracts worth more than 20000" → (df['contract_tier'].str.lower() == 'premium') & (df['contract_value_inr'] > 20000)
Query: "old equipment with complaints" → (df['equipment_age_years'] > 7) & (df['unresolved_complaints'] > 0)
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        expr = response.text.strip()
        # Clean up common LLM artifacts
        expr = expr.replace('```python', '').replace('```', '').strip()
        return expr
    except Exception as e:
        return f"ERROR: {str(e)}"


def apply_nl_filter(df_input: pd.DataFrame, expression: str):
    """Safely applies a pandas filter expression to the dataframe using eval().
    Returns (filtered_df, expression, error_message)."""
    if not expression or expression.startswith("ERROR:"):
        return None, expression, expression if expression.startswith("ERROR:") else "Empty expression received."

    try:
        # Create a safe local namespace with only df and pd
        mask = eval(expression, {"__builtins__": {}}, {"df": df_input, "pd": pd, "np": np})
        result = df_input[mask]
        return result, expression, None
    except KeyError as e:
        return None, expression, f"Column not found: {e}. Check if the column name is valid."
    except SyntaxError:
        return None, expression, f"Invalid filter syntax generated. Try rephrasing your query."
    except Exception as e:
        return None, expression, f"Filter failed: {str(e)}. Try a simpler query."


# --- Data Loading ---
@st.cache_data(ttl=3600)  # Added ttl to clear the stale cache and load new columns
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
st.title("AMC Retention Dashboard")

# --- Natural Language Search Bar ---
st.subheader("Natural Language Search")
st.caption("Ask questions about your customer data in plain English")

nl_query = st.text_input(
    "Search",
    placeholder='e.g. "Show me premium customers in Mumbai with churn risk above 70%"',
    label_visibility="collapsed",
    key="nl_search_bar"
)

# Process NL query
if nl_query:
    with st.spinner("Translating query..."):
        expression = nl_to_pandas_filter(nl_query)
        nl_result, nl_expr, nl_error = apply_nl_filter(df, expression)

    if nl_error:
        st.error(f"Error: {nl_error}")
        if nl_expr and not nl_expr.startswith("ERROR:"):
            st.caption(f"Generated expression: `{nl_expr}`")
    else:
        st.caption(f"Pandas filter: `{nl_expr}`")
        st.success(f"Found {len(nl_result)} matching customers")
        st.dataframe(
            nl_result[['customer_id', 'customer_name', 'city', 'equipment_brand',
                       'equipment_age_years', 'contract_value_inr', 'contract_tier',
                       'churn_probability', 'days_to_expiry']].sort_values('churn_probability', ascending=False).style.format({
                'churn_probability': '{:.1%}',
                'contract_value_inr': '₹ {:,.0f}'
            }),
            use_container_width=True,
            hide_index=True,
            height=300
        )

st.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("Global Filters")
st.sidebar.caption("Leave empty to select all")

selected_city = st.sidebar.multiselect("City", options=sorted(df['city'].unique()))
selected_brand = st.sidebar.multiselect("Equipment Brand", options=sorted(df['equipment_brand'].unique()))
selected_tier = st.sidebar.multiselect("Contract Tier", options=sorted(df['contract_tier'].unique()))
risk_filter = st.sidebar.multiselect("Risk Level", options=["Critical", "High", "Medium", "Low"])

# Filter logic
mask = pd.Series(True, index=df.index)
if selected_city: mask = mask & df['city'].isin(selected_city)
if selected_brand: mask = mask & df['equipment_brand'].isin(selected_brand)
if selected_tier: mask = mask & df['contract_tier'].isin(selected_tier)
if risk_filter: mask = mask & df['risk_level'].isin(risk_filter)

filtered_df = df[mask].copy()

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Revenue at Risk", "Explainability", "Field Ops", "Model Insights", "Retention Outreach", "Renewal Funnel"])

# ---- Tab 1: Overview ----
with tab1:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_contracts = len(filtered_df)
    at_risk = filtered_df[filtered_df['churn_probability'] > 0.5]
    revenue_at_risk = at_risk['contract_value_inr'].sum()
    critical_count = len(filtered_df[filtered_df['churn_probability'] > 0.75])
    
    # Calculate Real Renewal Rates
    # Contracts expiring in the last 30 days
    recent_expired = filtered_df[(filtered_df['days_to_expiry'] < 0) & (filtered_df['days_to_expiry'] >= -30)]
    recent_renewal_rate = len(recent_expired[recent_expired['churn'] == 0]) / len(recent_expired) if len(recent_expired) > 0 else 0.0

    # Contracts that expired 30 to 60 days ago
    older_expired = filtered_df[(filtered_df['days_to_expiry'] < -30) & (filtered_df['days_to_expiry'] >= -60)]
    older_renewal_rate = len(older_expired[older_expired['churn'] == 0]) / len(older_expired) if len(older_expired) > 0 else 0.0
    
    delta_rate = recent_renewal_rate - older_renewal_rate
    
    col1.metric("Contracts in View", f"{total_contracts:,}")
    col2.metric("Revenue at Risk", f"₹ {revenue_at_risk:,.0f}")
    col3.metric("Critical Flight Risks", f"{critical_count}")
    col4.metric("Avg Equipment Age", f"{filtered_df['equipment_age_years'].mean():.1f} yrs")
    
    with col5:
        st.metric("Renewal Rate (Last 30d)", f"{recent_renewal_rate:.0%}", f"{delta_rate:+.1%}")
        
        # Build Trend Sparkline
        spark_df = filtered_df.copy()
        spark_df['month'] = pd.to_datetime(spark_df['contract_end_date']).dt.to_period('M').astype(str)
        monthly_rates = spark_df.groupby('month').apply(lambda x: len(x[x['churn']==0])/len(x) if len(x)>0 else 0).reset_index()
        monthly_rates.columns = ['Month', 'Renewal Rate']
        monthly_rates = monthly_rates.sort_values('Month').tail(6) # Last 6 months for sparkline
        
        sparkline = alt.Chart(monthly_rates).mark_area(
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#3b82f6', offset=0), alt.GradientStop(color='white', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            ),
            line={'color': '#2563eb', 'strokeWidth': 2}
        ).encode(
            x=alt.X('Month:N', axis=None),
            y=alt.Y('Renewal Rate:Q', axis=None, scale=alt.Scale(zero=False)),
            tooltip=['Month', alt.Tooltip('Renewal Rate:Q', format='.0%')]
        ).properties(height=60).configure_view(strokeWidth=0)
        
        st.altair_chart(sparkline, use_container_width=True)
    
    st.markdown("---")
    st.subheader(f"High-Risk Customers ({len(at_risk)}) - Total Revenue at Risk: ₹ {revenue_at_risk:,.0f}")
    
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
    st.subheader("Customer Deep Dive")
    
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
        
        st.success(f"Prescribed Action: {customer_row['intervention']}")
        st.info(f"Outreach Channel: {customer_row['outreach_action']}")
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### Risk Factors")
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
            if customer_row['renewal_rate'] < 0.5:
                reasons.append(f"This customer has renewed only **{customer_row['previous_renewals']} of {customer_row['contracts_due_historical']} times** (Historical Renewal Rate: {customer_row['renewal_rate']:.0%}).")
            elif customer_row['previous_renewals'] == 0:
                reasons.append("First-year customer, no established loyalty yet.")
            
            if not reasons:
                reasons.append("Mix of minor factors (age, contract value, timing).")
                
            for i, r in enumerate(reasons):
                st.markdown(f"{i+1}. {r}")
                
        with col_right:
            st.markdown("#### Feature Impact (SHAP)")
            X_customer = pd.DataFrame([customer_row[feature_names]])
            for col in feature_names:
                X_customer[col] = pd.to_numeric(X_customer[col], errors='coerce')

            shap_values = explainer.shap_values(X_customer)[0]
            
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Impact': shap_values,
                'Absolute Impact': np.abs(shap_values)
            }).sort_values('Absolute Impact', ascending=False).head(8)
            
            shap_df['Direction'] = shap_df['Impact'].apply(lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
            
            chart = alt.Chart(shap_df).mark_bar().encode(
                x=alt.X('Impact:Q', title='Impact on Churn Probability', axis=alt.Axis(format='%')),
                y=alt.Y('Feature:N', sort='-x', title=''),
                color=alt.Color('Direction:N', scale=alt.Scale(domain=['Increases Risk', 'Decreases Risk'], range=['#de350b', '#00875a']), legend=alt.Legend(title="Effect")),
                tooltip=['Feature', alt.Tooltip('Impact:Q', format='.2%')]
            ).properties(height=350).configure_axis(grid=False).configure_view(strokeWidth=0)
            
            st.altair_chart(chart, use_container_width=True)

# ---- Tab 3: Field Ops Planner ----
with tab3:
    st.subheader("Field Technician Workload Balancer")
    st.caption("Prioritize physical visits for high-value customers at immediate risk of churn due to service failures.")
    
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
    st.subheader("Model Insights")
    st.markdown("""
    This system runs on two specialized models:
    1. **XGBoost Classifier**: Imbalanced learning (SMOTE/class weights) to predict binary churn probability.
    2. **Cox Proportional Hazards (Survival Analysis)**: Used by top telecom/SaaS companies to predict *time-to-churn*.
    """)
    
    importance = xgb_model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(15)
    
    chart = alt.Chart(imp_df).mark_bar(color='#0052cc').encode(
        x=alt.X('Importance:Q', title='Relative Importance (Gain)'),
        y=alt.Y('Feature:N', sort='-x', title=''),
        tooltip=['Feature', 'Importance']
    ).properties(height=500, title='Global Feature Importance (Top 15)').configure_axis(grid=False).configure_view(strokeWidth=0)
    
    st.altair_chart(chart, use_container_width=True)

# ---- Tab 5: Retention Letter Generator ----
with tab5:
    st.subheader("Retention Outreach Generator")
    st.caption("Generate personalized retention messages for at-risk customers using Gemini 2.5 Flash.")
    st.markdown("---")

    mode = st.radio("Outreach Mode", ["Individual Customer", "Renewal Campaign (Bulk)"], horizontal=True)

    if mode == "Individual Customer":
        ret_customers = filtered_df.sort_values('churn_probability', ascending=False)['customer_id'].tolist()

        if not ret_customers:
            st.warning("No customers match the current filters. Adjust the sidebar filters to see customers.")
        else:
            col_sel1, col_sel2 = st.columns([2, 1])
            with col_sel1:
                ret_selected = st.selectbox("Select Customer", ret_customers, key="ret_customer_select",
                                            format_func=lambda cid: f"{cid} — {filtered_df[filtered_df['customer_id']==cid].iloc[0]['customer_name']} (Risk: {filtered_df[filtered_df['customer_id']==cid].iloc[0]['churn_probability']:.0%})")
            with col_sel2:
                ret_channel = st.selectbox("Outreach Channel", ["WhatsApp", "Email", "Formal Letter"], key="ret_channel_select")

            ret_row = filtered_df[filtered_df['customer_id'] == ret_selected].iloc[0]

            # Show a compact customer summary card
            st.write("**Customer Profile**")
            sum_cols = st.columns(4)
            sum_cols[0].write(f"**Name:** {ret_row['customer_name']}\n\n**City:** {ret_row['city']}")
            sum_cols[1].write(f"**Brand:** {ret_row['equipment_brand']}\n\n**Age:** {int(ret_row['equipment_age_years'])} yrs")
            sum_cols[2].write(f"**Contract:** ₹{ret_row['contract_value_inr']:,.0f} ({ret_row['contract_tier']})\n\n**Expiry:** {ret_row['contract_end_date']}")
            sum_cols[3].write(f"**Risk:** {ret_row['churn_probability']:.0%}\n\n**Renewal Rate:** {ret_row['renewal_rate']:.0%}")
            
            st.markdown("<br>", unsafe_allow_html=True)

            # Generate button
            if st.button("Generate Message", type="primary", use_container_width=True):
                with st.spinner(f"Generating {ret_channel} message..."):
                    message = generate_retention_message(ret_row.to_dict(), ret_channel)
                    st.session_state['generated_message'] = message
                    st.session_state['generated_channel'] = ret_channel

            # Display result
            if 'generated_message' in st.session_state and st.session_state['generated_message']:
                st.markdown(f"#### Generated {st.session_state.get('generated_channel', '')} Message")
                st.text_area("Output", value=st.session_state['generated_message'], height=300, key="ret_output_area")

    else:
        st.write("#### 🎯 Bulk Renewal Campaign")
        st.write("Targeting contracts expiring in < 30 days with < 50% historical renewal rate.")
        target_df = filtered_df[(filtered_df['days_to_expiry'] < 30) & (filtered_df['renewal_rate'] < 0.5)]
        st.metric("Eligible Customers for Campaign", len(target_df))
        
        if len(target_df) > 0:
            st.dataframe(target_df[['customer_id', 'customer_name', 'renewal_rate', 'days_to_expiry', 'churn_probability']].style.format({'renewal_rate': '{:.0%}', 'churn_probability': '{:.0%}'}))
            campaign_channel = st.selectbox("Campaign Channel", ["WhatsApp", "Email"], key="camp_chan")
            if st.button("Generate Campaign Drafts (Top 3 as Demo)", type="primary"):
                with st.spinner("Generating bulk messages..."):
                    for _, row in target_df.head(3).iterrows():
                        msg = generate_retention_message(row.to_dict(), campaign_channel)
                        st.markdown(f"**To: {row['customer_name']}** (`{row['customer_id']}`)")
                        st.info(msg)

# ---- Tab 6: Renewal Funnel ----
with tab6:
    st.subheader("Renewal Pipeline Funnel")
    st.caption("Conversion metrics for contracts expiring within the next 60 days.")
    
    funnel_df = filtered_df[filtered_df['days_to_expiry'] < 60].copy()
    
    stage1 = len(funnel_df) # Contracts Due
    stage2 = len(funnel_df[funnel_df['renewal_reminder_sent'] == 1]) # Contacted
    stage3 = len(funnel_df[funnel_df['churn'] == 0]) # Renewed
    stage4 = len(funnel_df[funnel_df['churn'] == 1]) # Churned
    
    if stage1 > 0:
        funnel_data = pd.DataFrame({
            'Stage': ['1. Contracts Due', '2. Contacted', '3. Renewed', '4. Churned'],
            'Count': [stage1, stage2, stage3, stage4]
        })
        
        # Altair bar chart as funnel
        chart = alt.Chart(funnel_data).mark_bar(color='#0f172a').encode(
            y=alt.Y('Stage:N', sort=None, title=''),
            x=alt.X('Count:Q', title='Number of Customers'),
            tooltip=['Stage', 'Count']
        ).properties(height=250).configure_axis(grid=False).configure_view(strokeWidth=0)
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No contracts expiring within the next 60 days matching the current filters.")
    
    st.markdown("---")
    st.subheader("Customer Segmentation by Renewal Rate")
    
    def bucket_rate(r):
        if r <= 0.4: return "0-40% (Low)"
        elif r <= 0.7: return "40-70% (Medium)"
        else: return "70-100% (High)"
        
    filtered_df['Renewal Segment'] = filtered_df['renewal_rate'].apply(bucket_rate)
    seg_df = filtered_df.groupby('Renewal Segment').agg(
        Customers=('customer_id', 'count'),
        Avg_Churn_Risk=('churn_probability', 'mean')
    ).reset_index()
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.dataframe(seg_df.style.format({'Avg_Churn_Risk': '{:.1%}'}), hide_index=True)
    with col_s2:
        seg_chart = alt.Chart(seg_df).mark_bar(color='#2563eb').encode(
            x='Renewal Segment:N',
            y=alt.Y('Avg_Churn_Risk:Q', axis=alt.Axis(format='%')),
            tooltip=['Renewal Segment', alt.Tooltip('Avg_Churn_Risk:Q', format='.1%')]
        ).properties(height=200)
        st.altair_chart(seg_chart, use_container_width=True)
