import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import altair as alt
import os
from google import genai

# --- Page Config ---
st.set_page_config(page_title="AMC Retention Dashboard", layout="wide", page_icon="🛡️")

# --- Custom CRM CSS ---
st.markdown("""
<style>
    /* Professional CRM Theme */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #0052cc;
    }
    div[data-testid="stMetricValue"] {
        color: #172b4d;
        font-size: 28px;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #5e6c84;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        border-bottom: 2px solid #ebecf0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 10px 15px;
        color: #5e6c84;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f4f5f7;
        border-bottom: 3px solid #0052cc !important;
        color: #0052cc;
    }
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        overflow: hidden;
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

# --- Gemini Retention Message Generator ---
def generate_retention_message(customer: dict, channel: str) -> str:
    """Calls Google Gemini to generate a personalized retention outreach message."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "ERROR: GEMINI_API_KEY environment variable is not set. Please set it and restart the app."

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
- Churn Risk Score: {customer.get('churn_probability', 0):.0%}

Channel: {channel}
{channel_instructions.get(channel, channel_instructions['Email'])}

Rules:
1. Address the customer by their actual name.
2. Mention their specific equipment brand and age naturally in the message.
3. Reference their contract expiry date or days remaining.
4. If they have unresolved complaints or missed visits, acknowledge and apologize.
5. Include a soft, non-pushy call to action (e.g., schedule a call, renew online, reply to this message).
6. Do NOT invent any data not provided above.
7. Output ONLY the message — no commentary, no notes, no meta-text.
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
    api_key = os.environ.get("GEMINI_API_KEY")
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

# --- Natural Language Search Bar ---
st.markdown("""<div style='background:#ffffff; border:1px solid #dfe1e6; border-radius:8px; padding:16px 20px; margin-bottom:20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);'>
    <span style='color:#172b4d; font-weight:600; font-size:15px;'>🧠 AI-Powered Natural Language Search</span>
    <span style='color:#5e6c84; font-size:13px; margin-left:12px;'>Ask questions about your customer data in plain English</span>
</div>""", unsafe_allow_html=True)

nl_query = st.text_input(
    "Ask a question",
    placeholder='e.g. "Show me premium customers in Mumbai with churn risk above 70%"',
    label_visibility="collapsed",
    key="nl_search_bar"
)

# Process NL query
if nl_query:
    with st.spinner("🧠 Translating your query with Gemini AI..."):
        expression = nl_to_pandas_filter(nl_query)
        nl_result, nl_expr, nl_error = apply_nl_filter(df, expression)

    if nl_error:
        st.error(f"❌ {nl_error}")
        if nl_expr and not nl_expr.startswith("ERROR:"):
            st.caption(f"Generated expression: `{nl_expr}`")
    else:
        st.caption(f"🔗 Pandas filter: `{nl_expr}`")
        st.success(f"✅ Found **{len(nl_result)}** matching customers")
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
st.sidebar.markdown("## 🔎 Global Filters")
st.sidebar.markdown("<span style='color:#5e6c84; font-size: 14px;'>Leave empty to select all</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")

selected_city = st.sidebar.multiselect("📍 City", options=sorted(df['city'].unique()))
selected_brand = st.sidebar.multiselect("🏷️ Equipment Brand", options=sorted(df['equipment_brand'].unique()))
selected_tier = st.sidebar.multiselect("⭐ Contract Tier", options=sorted(df['contract_tier'].unique()))
risk_filter = st.sidebar.multiselect("⚠️ Risk Level", options=["🔴 Critical", "🟡 High", "🟠 Medium", "🟢 Low"])

# Filter logic
mask = pd.Series(True, index=df.index)
if selected_city: mask = mask & df['city'].isin(selected_city)
if selected_brand: mask = mask & df['equipment_brand'].isin(selected_brand)
if selected_tier: mask = mask & df['contract_tier'].isin(selected_tier)
if risk_filter: mask = mask & df['risk_level'].isin(risk_filter)

filtered_df = df[mask].copy()

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Revenue at Risk", "👥 Explainability", "🔧 Field Ops", "📈 Model Insights", "✉️ Retention Outreach"])

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
            st.markdown("#### 📊 Key Churn Drivers (SHAP Impact)")
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
    }).sort_values('Importance', ascending=False).head(15)
    
    chart = alt.Chart(imp_df).mark_bar(color='#0052cc').encode(
        x=alt.X('Importance:Q', title='Relative Importance (Gain)'),
        y=alt.Y('Feature:N', sort='-x', title=''),
        tooltip=['Feature', 'Importance']
    ).properties(height=500, title='Global Feature Importance (Top 15)').configure_axis(grid=False).configure_view(strokeWidth=0)
    
    st.altair_chart(chart, use_container_width=True)

# ---- Tab 5: Retention Letter Generator ----
with tab5:
    st.markdown("### ✉️ AI-Powered Retention Outreach Generator")
    st.markdown("Generate personalized retention messages for at-risk customers using **Google Gemini AI**. "
                "Messages are crafted from each customer's real data — equipment details, contract status, and service history.")
    st.markdown("---")

    ret_customers = filtered_df.sort_values('churn_probability', ascending=False)['customer_id'].tolist()

    if not ret_customers:
        st.warning("No customers match the current filters. Adjust the sidebar filters to see customers.")
    else:
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            ret_selected = st.selectbox("Select Customer", ret_customers, key="ret_customer_select",
                                        format_func=lambda cid: f"{cid} — {df[df['customer_id']==cid].iloc[0]['customer_name']} (Risk: {df[df['customer_id']==cid].iloc[0]['churn_probability']:.0%})")
        with col_sel2:
            ret_channel = st.selectbox("Outreach Channel", ["WhatsApp", "Email", "Formal Letter"], key="ret_channel_select")

        ret_row = df[df['customer_id'] == ret_selected].iloc[0]

        # Show a compact customer summary card
        st.markdown(f"""
        <div style='background:#f4f5f7; border:1px solid #dfe1e6; border-radius:8px; padding:16px 20px; margin:12px 0;'>
            <div style='display:flex; gap:40px; flex-wrap:wrap;'>
                <div><strong style='color:#5e6c84;'>NAME</strong><br>{ret_row['customer_name']}</div>
                <div><strong style='color:#5e6c84;'>CITY</strong><br>{ret_row['city']}</div>
                <div><strong style='color:#5e6c84;'>BRAND</strong><br>{ret_row['equipment_brand']}</div>
                <div><strong style='color:#5e6c84;'>AGE</strong><br>{int(ret_row['equipment_age_years'])} yrs</div>
                <div><strong style='color:#5e6c84;'>CONTRACT</strong><br>₹{ret_row['contract_value_inr']:,.0f} ({ret_row['contract_tier']})</div>
                <div><strong style='color:#5e6c84;'>EXPIRY</strong><br>{ret_row['contract_end_date']} ({int(ret_row['days_to_expiry'])}d left)</div>
                <div><strong style='color:#5e6c84;'>RISK</strong><br>{ret_row['churn_probability']:.0%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Generate button
        if st.button("🚀 Generate Retention Message", type="primary", use_container_width=True):
            with st.spinner(f"Generating {ret_channel} message via Gemini AI..."):
                message = generate_retention_message(ret_row.to_dict(), ret_channel)
                st.session_state['generated_message'] = message
                st.session_state['generated_channel'] = ret_channel

        # Display result
        if 'generated_message' in st.session_state and st.session_state['generated_message']:
            st.markdown(f"#### Generated {st.session_state.get('generated_channel', '')} Message")
            st.text_area("Output", value=st.session_state['generated_message'], height=300, key="ret_output_area")

            # Copy button using st.code as fallback + JS clipboard
            st.markdown(f"""
            <textarea id="clipboardText" style="position:absolute;left:-9999px;">{st.session_state['generated_message']}</textarea>
            <button onclick="
                var t = document.getElementById('clipboardText');
                t.style.position='static';
                t.select();
                document.execCommand('copy');
                t.style.position='absolute';
                t.style.left='-9999px';
                this.innerText='✅ Copied!';
                setTimeout(()=>this.innerText='📋 Copy to Clipboard', 2000);
            " style="
                background:#0052cc; color:white; border:none; padding:10px 24px;
                border-radius:6px; font-size:14px; font-weight:600; cursor:pointer;
                margin-top:8px; transition: background 0.2s;
            " onmouseover="this.style.background='#0065ff'" onmouseout="this.style.background='#0052cc'">
                📋 Copy to Clipboard
            </button>
            """, unsafe_allow_html=True)
