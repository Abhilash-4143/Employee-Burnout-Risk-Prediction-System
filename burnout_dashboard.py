"""Professional Employee Burnout Risk Analytics Dashboard.

Enterprise-grade HR analytics platform with clean, professional UI.
Run with: streamlit run burnout_dashboard.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.business_insights import HR_RECOMMENDATIONS, build_insights
from src.data_preprocessing import detect_target_column, load_data
from src.feature_engineering import (
    add_engineered_features,
    feature_importance_by_correlation,
)
from src.model_training import DEFAULT_MODEL_PATH, load_model, predict

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Employee Burnout Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Color Palette
COLORS = {
    "primary": "#1f2937",      # Dark gray
    "secondary": "#4f46e5",    # Indigo
    "accent": "#06b6d4",       # Cyan
    "success": "#10b981",      # Green
    "warning": "#f59e0b",      # Amber
    "danger": "#ef4444",       # Red
    "bg_light": "#f9fafb",     # Light gray
    "bg_card": "#ffffff",      # White
    "text": "#111827",         # Dark text
    "text_muted": "#6b7280",   # Muted text
    "border": "#e5e7eb",       # Light border
}

# ============================================================================
# CUSTOM CSS - CLEAN & PROFESSIONAL
# ============================================================================
CUSTOM_CSS = f"""
<style>
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}

    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                     'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
                     sans-serif;
        background: {COLORS['bg_light']};
        color: {COLORS['text']};
    }}

    .stApp {{
        background: {COLORS['bg_light']};
    }}

    [data-testid="stAppViewContainer"] {{
        background: {COLORS['bg_light']};
        padding: 2rem 0;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
        border-bottom: 1px solid {COLORS['border']};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {COLORS['bg_card']};
        border-right: 1px solid {COLORS['border']};
    }}

    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        padding: 2rem 1.5rem;
    }}

    /* Main Container */
    .main {{
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 2rem;
    }}

    /* Headers */
    h1 {{
        color: {COLORS['text']};
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }}

    h2 {{
        color: {COLORS['text']};
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        letter-spacing: -0.25px;
    }}

    h3 {{
        color: {COLORS['text']};
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }}

    /* Text */
    p, span {{
        color: {COLORS['text_muted']};
        line-height: 1.6;
    }}

    /* Cards */
    [data-testid="stMetricContainer"] {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }}

    [data-testid="stMetricContainer"]:hover {{
        border-color: {COLORS['secondary']};
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.08);
    }}

    /* KPI Cards */
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {COLORS['secondary']};
        margin: 0.5rem 0;
    }}

    .metric-label {{
        font-size: 0.9rem;
        font-weight: 500;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Buttons */
    .stButton > button {{
        background: {COLORS['secondary']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }}

    .stButton > button:hover {{
        background: #4338ca;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
        transform: translateY(-1px);
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        color: {COLORS['text']};
        padding: 0.75rem;
        font-size: 0.95rem;
    }}

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {{
        border-color: {COLORS['secondary']};
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Expander */
    [data-testid="stExpander"] {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
    }}

    /* Tabs */
    [data-testid="stTabs"] {{
        gap: 0;
    }}

    [data-testid="stTabs"] button {{
        color: {COLORS['text_muted']};
        border-bottom: 2px solid transparent;
        border-radius: 0;
    }}

    [data-testid="stTabs"] button[data-selected="true"] {{
        color: {COLORS['secondary']};
        border-bottom-color: {COLORS['secondary']};
    }}

    /* Success/Warning/Error */
    [data-testid="stAlert"] {{
        border-radius: 8px;
        border: 1px solid;
    }}

    [data-testid="stAlert"][type="success"] {{
        background: rgba(16, 185, 129, 0.05);
        border-color: rgba(16, 185, 129, 0.2);
        color: #065f46;
    }}

    [data-testid="stAlert"][type="warning"] {{
        background: rgba(245, 158, 11, 0.05);
        border-color: rgba(245, 158, 11, 0.2);
        color: #78350f;
    }}

    [data-testid="stAlert"][type="error"] {{
        background: rgba(239, 68, 68, 0.05);
        border-color: rgba(239, 68, 68, 0.2);
        color: #7f1d1d;
    }}

    /* Divider */
    hr {{
        border: none;
        height: 1px;
        background: {COLORS['border']};
        margin: 2rem 0;
    }}

    /* Columns */
    [data-testid="column"] {{
        padding: 0 1rem;
    }}

    [data-testid="column"]:first-child {{
        padding-left: 0;
    }}

    [data-testid="column"]:last-child {{
        padding-right: 0;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: {COLORS['bg_light']};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {COLORS['border']};
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['text_muted']};
    }}

    /* Subtitle */
    .subtitle {{
        color: {COLORS['text_muted']};
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }}

    /* Stats Grid */
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }}

    /* Chart Container */
    .chart-container {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }}

    /* Section */
    .section {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
    }}

    /* Table */
    table {{
        width: 100%;
        border-collapse: collapse;
    }}

    th {{
        background: {COLORS['bg_light']};
        color: {COLORS['text']};
        font-weight: 600;
        border-bottom: 2px solid {COLORS['border']};
        padding: 1rem;
        text-align: left;
    }}

    td {{
        border-bottom: 1px solid {COLORS['border']};
        padding: 1rem;
    }}

    tr:hover {{
        background: {COLORS['bg_light']};
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_data_cached():
    """Load data with caching."""
    try:
        df = load_data()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def load_model_cached():
    """Load trained model with caching."""
    try:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            st.warning("Model not found. Please run main.py first.")
            return None
        return load_model(DEFAULT_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def create_kpi_card(title, value, unit="", color="secondary"):
    """Create a professional KPI card."""
    col_color = COLORS.get(color, COLORS["secondary"])
    
    with st.container():
        st.markdown(
            f"""
            <div style="
                background: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 1.5rem;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                <div style="
                    font-size: 0.9rem;
                    font-weight: 500;
                    color: {COLORS['text_muted']};
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    margin-bottom: 0.5rem;
                ">{title}</div>
                <div style="
                    font-size: 2.5rem;
                    font-weight: 700;
                    color: {col_color};
                    margin-bottom: 0.25rem;
                ">{value}</div>
                <div style="
                    font-size: 0.85rem;
                    color: {COLORS['text_muted']};
                ">{unit}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================================
# SIDEBAR & NAVIGATION
# ============================================================================

st.sidebar.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, {COLORS['secondary']} 0%, {COLORS['accent']} 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
        <div style="font-size: 1.3rem; font-weight: 700; margin-bottom: 0.25rem;">
            Burnout Analytics
        </div>
        <div style="font-size: 0.9rem; opacity: 0.9;">
            HR Risk Assessment
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Overview", "Analysis", "Prediction", "Risk Assessment", "Model Performance", "Recommendations", "About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="
        color: {COLORS['text_muted']};
        font-size: 0.85rem;
        line-height: 1.6;
    ">
        <strong>Platform</strong><br>
        Real-time employee burnout risk assessment and analytics system.
        <br><br>
        <strong>Model</strong><br>
        Linear Regression with 12 engineered features.
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

if page == "Dashboard":
    st.title("Dashboard")
    st.markdown('<p class="subtitle">Executive overview of employee wellbeing metrics</p>', unsafe_allow_html=True)
    
    df = load_data_cached()
    if df is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            create_kpi_card("Total Employees", f"{len(df):,}", "")
        with col2:
            avg_burnout = df["burnout_risk_score"].mean()
            create_kpi_card("Avg Burnout", f"{avg_burnout:.1f}", "out of 100")
        with col3:
            high_risk = (df["burnout_risk_score"] > 70).sum()
            create_kpi_card("High Risk", f"{high_risk}", "employees")
        with col4:
            avg_stress = df["stress_level"].mean()
            create_kpi_card("Avg Stress", f"{avg_stress:.1f}", "")
        with col5:
            avg_productivity = df["productivity_score"].mean()
            create_kpi_card("Avg Productivity", f"{avg_productivity:.1f}", "")

        st.markdown("---")

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Burnout Distribution")
            fig = px.histogram(df, x="burnout_risk_score", nbins=30, color_discrete_sequence=[COLORS["secondary"]])
            fig.update_layout(template="plotly_white", height=400, showlegend=False, xaxis_title="Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Risk Breakdown")
            low = (df["burnout_risk_score"] < 35).sum()
            medium = ((df["burnout_risk_score"] >= 35) & (df["burnout_risk_score"] <= 70)).sum()
            high = (df["burnout_risk_score"] > 70).sum()
            
            fig = px.pie(values=[low, medium, high], names=["Low", "Medium", "High"],
                        color_discrete_map={"Low": COLORS["success"], "Medium": COLORS["warning"], "High": COLORS["danger"]})
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: DATA OVERVIEW
# ============================================================================

elif page == "Data Overview":
    st.title("Data Overview")
    st.markdown('<p class="subtitle">Dataset summary and quality metrics</p>', unsafe_allow_html=True)

    df = load_data_cached()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            create_kpi_card("Records", f"{len(df):,}", "rows")
        with col2:
            create_kpi_card("Features", f"{len(df.columns)}", "columns")
        with col3:
            missing = df.isnull().sum().sum()
            create_kpi_card("Missing", f"{missing}", "cells")
        with col4:
            duplicates = df.duplicated().sum()
            create_kpi_card("Duplicates", f"{duplicates}", "rows")

        st.markdown("---")
        st.markdown("### Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

# ============================================================================
# PAGE: ANALYSIS
# ============================================================================

elif page == "Analysis":
    st.title("Data Analysis")
    st.markdown('<p class="subtitle">Feature relationships and correlations</p>', unsafe_allow_html=True)

    df = load_data_cached()
    if df is not None:
        df = add_engineered_features(df)
        
        st.markdown("### Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:12]
        fig = px.imshow(df[numeric_cols].corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: PREDICTION
# ============================================================================

elif page == "Prediction":
    st.title("Risk Prediction")
    st.markdown('<p class="subtitle">Predict individual burnout risk</p>', unsafe_allow_html=True)

    model = load_model_cached()
    if model is not None:
        st.markdown("### Employee Information")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Age", 18, 70, 35)
        with col2:
            experience = st.number_input("Years Experience", 0, 50, 5)
        with col3:
            work_hours = st.number_input("Weekly Hours", 0, 80, 40)
        with col4:
            meetings = st.number_input("Meetings/Week", 0, 50, 10)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            emails = st.number_input("Emails/Day", 0, 500, 50)
        with col2:
            projects = st.number_input("Projects", 0, 20, 3)
        with col3:
            remote = st.number_input("Remote Days/Month", 0, 30, 10)
        with col4:
            sleep = st.number_input("Sleep Hours", 1.0, 12.0, 7.0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stress = st.slider("Stress (1-10)", 1, 10, 5)
        with col2:
            exercise = st.number_input("Exercise Hours/Week", 0, 20, 3)
        with col3:
            sick_leaves = st.number_input("Sick Days/Year", 0, 50, 3)
        with col4:
            productivity = st.slider("Productivity (1-10)", 1, 10, 7)

        if st.button("Calculate Risk", use_container_width=True):
            # Build a DataFrame with the raw feature names so add_engineered_features
            # can compute workload_index, health_index, productivity_efficiency.
            raw_input = pd.DataFrame([{
                "age": age,
                "years_experience": experience,
                "weekly_work_hours": work_hours,
                "meetings_per_week": meetings,
                "emails_sent_per_day": emails,
                "projects_handled": projects,
                "remote_days_per_month": remote,
                "sleep_hours": sleep,
                "stress_level": stress,
                "exercise_hours_week": exercise,
                "sick_leaves_year": sick_leaves,
                "productivity_score": productivity,
            }])
            engineered_input = add_engineered_features(raw_input)
            # Select features in the exact order the scaler was trained on.
            feature_names = model["feature_names"]
            input_data = engineered_input[feature_names].values
            burnout_risk = predict(model, input_data)
            score = burnout_risk[0]
            
            st.markdown("---")
            
            if score < 35:
                risk = "Low Risk"
                color = COLORS["success"]
            elif score < 70:
                risk = "Medium Risk"
                color = COLORS["warning"]
            else:
                risk = "High Risk"
                color = COLORS["danger"]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                <div style="background: {color}15; border: 2px solid {color}; border-radius: 8px; padding: 2rem; text-align: center;">
                    <div style="font-size: 2.5rem; font-weight: 700; color: {color};">{score:.1f}%</div>
                    <div style="font-size: 1rem; color: {COLORS['text']}; margin-top: 0.5rem;">{risk}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Recommendations")
                if score >= 70:
                    st.error("⚠️ High-risk employee. Schedule immediate wellness intervention.")
                elif score >= 40:
                    st.warning("Monitor closely. Provide flexible work arrangements and support.")
                else:
                    st.success("✓ Employee is doing well. Continue regular engagement.")

# ============================================================================
# PAGE: RISK ASSESSMENT
# ============================================================================

elif page == "Risk Assessment":
    st.title("Risk Assessment")
    st.markdown('<p class="subtitle">Employee-level risk analysis</p>', unsafe_allow_html=True)

    df = load_data_cached()
    if df is not None:
        col1, col2, col3 = st.columns(3)
        
        low_risk = (df["burnout_risk_score"] < 35).sum()
        medium = ((df["burnout_risk_score"] >= 35) & (df["burnout_risk_score"] <= 70)).sum()
        high_risk = (df["burnout_risk_score"] > 70).sum()
        
        with col1:
            create_kpi_card("Low Risk", low_risk, f"{low_risk/len(df)*100:.1f}%", "success")
        with col2:
            create_kpi_card("Medium", medium, f"{medium/len(df)*100:.1f}%", "warning")
        with col3:
            create_kpi_card("High Risk", high_risk, f"{high_risk/len(df)*100:.1f}%", "danger")

        st.markdown("---")
        st.markdown("### High-Risk Employees")
        high_risk_df = df[df["burnout_risk_score"] > 70].sort_values("burnout_risk_score", ascending=False).head(20)
        st.dataframe(high_risk_df[["age", "stress_level", "weekly_work_hours", "sleep_hours", "productivity_score", "burnout_risk_score"]], use_container_width=True)

# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================

elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown('<p class="subtitle">Machine learning model metrics</p>', unsafe_allow_html=True)

    report_path = os.path.join("reports", "model_report.txt")
    if os.path.exists(report_path):
        with open(report_path) as f:
            st.code(f.read(), "text")
    else:
        st.info("Run main.py to generate reports.")

# ============================================================================
# PAGE: RECOMMENDATIONS
# ============================================================================

elif page == "Recommendations":
    st.title("Business Recommendations")
    st.markdown('<p class="subtitle">Evidence-based HR strategies</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Organizational Initiatives")
        for i, rec in enumerate(HR_RECOMMENDATIONS[:9], 1):
            st.markdown(f"**{i}. {rec}**")
    
    with col2:
        st.markdown("### Individual Wellness")
        for i, rec in enumerate(HR_RECOMMENDATIONS[9:], 1):
            st.markdown(f"**{i}. {rec}**")

# ============================================================================
# PAGE: ABOUT
# ============================================================================

elif page == "About":
    st.title("About")
    st.markdown('<p class="subtitle">Project information and technical details</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Overview")
        st.markdown("""
Machine learning system that predicts employee burnout risk and provides actionable recommendations.

**Key Features:**
- Real-time burnout assessment
- Predictive modeling
- Data-driven insights
- Personalized recommendations
""")
    
    with col2:
        st.markdown("### Technology")
        st.markdown("""
**Backend:** Python, Scikit-Learn
**Frontend:** Streamlit
**Visualization:** Plotly
**Model:** Linear Regression
**Data:** CSV format
""")
    
    st.markdown("---")
    st.markdown("### Dataset Features (13 Total)")
    features_list = [
        "Age", "Years Experience", "Weekly Work Hours", "Meetings per Week",
        "Emails Sent Daily", "Projects Handled", "Remote Days Monthly",
        "Sleep Hours", "Stress Level", "Exercise Hours Weekly",
        "Sick Leaves Yearly", "Productivity Score", "Burnout Risk Score"
    ]
    
    cols = st.columns(3)
    for i, feature in enumerate(features_list):
        with cols[i % 3]:
            st.markdown(f"- {feature}")
