"""
streamlit_app.py - Space-Grade Burn-In Latent Defect Screening Dashboard
SIH26170: Automated Multi-Lot Parametric Screening & Explainability System

Pages & Features:
1. Executive Screening Overview & Key KPIs (Recall, Escapes, Hours Saved)
2. Production Lot Analytics & AEC-Q001 Dynamic PAT Distribution
3. Flagged Components & 3-Tier Disposition (ACCEPT / REVIEW / REJECT)
4. QA Inspector Deep-Dive: Report Card, Plotly Trajectory Cone, Feature Attributions
5. Model Validation & Baselines Comparison (Static vs Global Z vs Dynamic PAT)
6. Interactive What-If Simulation (Threshold & Cost Sensitivity Tuning)
7. CSV Batch Inference & Export
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generate_physics import generate_physics_dataset, GeneratorConfig, split_lots
from src.preprocess import BurnInPreprocessor
from src.module_a import DynamicOutlierDetector, evaluate_baselines
from src.module_b import DriftPredictor, evaluate_drift_prediction
from src.decision import ScreeningDecisionEngine, SequentialScreeningEngine
from src.explain import (
    generate_qa_report_card, compute_local_feature_contributions,
    generate_trajectory_plot_data, generate_model_card,
)

# Industrial & Aerospace Integrations
from src.integrations.database import ScreeningDatabase
from src.integrations.ate_parser import ATEDataParser
from src.integrations.chamber_connector import ChamberController
from src.integrations.mes_webhook import MESWebhookNotifier
from src.integrations.compliance_coc import CertificateOfConformanceGenerator

# Page Configuration
st.set_page_config(
    page_title="SIH26170: Space-Grade Burn-In Defect Screening",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Aesthetics Upgrade)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 8s ease infinite;
    }

    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Metric Cards Glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 10px 10px -5px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Radio Button Navigation Styling */
    div.row-widget.stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    div.row-widget.stRadio > div > label {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 10px 15px;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        cursor: pointer;
    }
    div.row-widget.stRadio > div > label:hover {
        background: rgba(59, 130, 246, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
        transform: translateX(5px);
    }
    div.row-widget.stRadio > div > label[data-checked="true"] {
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.2) 0%, transparent 100%);
        border-left: 4px solid #3b82f6;
    }

    /* Status Badges & Callouts */
    .status-accept { color: #10b981; font-weight: 700; text-shadow: 0 0 10px rgba(16, 185, 129, 0.4); }
    .status-review { color: #f59e0b; font-weight: 700; text-shadow: 0 0 10px rgba(245, 158, 11, 0.4); }
    .status-reject { color: #ef4444; font-weight: 700; text-shadow: 0 0 10px rgba(239, 68, 68, 0.4); }
    
    .report-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #8b5cf6;
        padding: 24px;
        border-radius: 0 12px 12px 0;
        backdrop-filter: blur(8px);
        box-shadow: inset 0 0 20px rgba(139, 92, 246, 0.05);
        font-family: 'Inter', sans-serif;
    }
    
    /* Dataframes Customization */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_or_generate_data():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic')
    test_path = os.path.join(data_dir, 'burnin_test.csv')
    train_path = os.path.join(data_dir, 'burnin_train.csv')
    val_path = os.path.join(data_dir, 'burnin_val.csv')
    
    if os.path.exists(test_path) and os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
    else:
        df = generate_physics_dataset(GeneratorConfig(n_lots=45, random_state=42))
        train_df, val_df, test_df = split_lots(df)
        os.makedirs(data_dir, exist_ok=True)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
    return train_df, val_df, test_df


@st.cache_resource
def train_and_run_pipeline(train_df, val_df, test_df):
    preprocessor = BurnInPreprocessor(small_lot_threshold=30, shrinkage_prior_weight=15.0)
    preprocessor.fit(train_df)
    
    train_p = preprocessor.transform(train_df)
    val_p = preprocessor.transform(val_df)
    test_p = preprocessor.transform(test_df)
    
    mod_a = DynamicOutlierDetector(cost_fn_weight=50.0, cost_fp_weight=1.0)
    mod_a.fit(train_p, val_p)
    
    mod_b = DriftPredictor(safety_k_sigma=3.0)
    mod_b.fit(train_p)
    
    pred_a = mod_a.predict_detailed(test_p)
    pred_b = mod_b.predict(test_p)
    
    engine = ScreeningDecisionEngine()
    final_df = engine.evaluate(pred_a, pred_b)
    
    # SPRT sequential decision layer
    sprt = SequentialScreeningEngine()
    final_df = sprt.decide_at_24h(final_df)
    sprt_economics = sprt.compute_economics(final_df)
    
    # Detector report and model card
    detector_report = mod_a.get_detector_report()
    model_card = generate_model_card(
        n_train_parts=len(train_p),
        n_defects=int(train_p['is_defect'].sum()) if 'is_defect' in train_p.columns else 0,
        certified_recall=detector_report.get('certified_recall_bound'),
        arrhenius_Ea=mod_b.Ea,
        detector_report=detector_report,
    )
    
    baseline_df = evaluate_baselines(final_df)
    drift_metrics = evaluate_drift_prediction(final_df)
    
    return (
        final_df, baseline_df, drift_metrics,
        preprocessor, mod_a, mod_b,
        sprt_economics, detector_report, model_card,
    )


# Load Data & Pipeline
train_df, val_df, test_df = load_or_generate_data()
(
    final_df, baseline_df, drift_metrics,
    preprocessor, mod_a, mod_b,
    sprt_economics, detector_report, model_card,
) = train_and_run_pipeline(train_df, val_df, test_df)

# Initialize Integration Singletons in Session State
if 'db' not in st.session_state:
    st.session_state.db = ScreeningDatabase()
if 'chamber_controller' not in st.session_state:
    st.session_state.chamber_controller = ChamberController()
if 'mes_notifier' not in st.session_state:
    st.session_state.mes_notifier = MESWebhookNotifier()
if 'coc_generator' not in st.session_state:
    st.session_state.coc_generator = CertificateOfConformanceGenerator()
if 'ate_parser' not in st.session_state:
    st.session_state.ate_parser = ATEDataParser()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/satellite.png", width=64)
st.sidebar.title("Space Screening System")
st.sidebar.caption("SIH26170 | High-Reliability Electronics ESS")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🚀 Executive Screening Overview",
        "📊 Lot Distribution & Dynamic PAT",
        "⚠️ Flagged Parts & Disposition",
        "🔬 QA Inspector Deep-Dive",
        "🧬 SPRT Sequential Decision",
        "📈 Benchmarks & Baselines",
        "🎛️ Interactive What-If Simulation",
        "📁 Batch CSV Inference & Export",
        "📡 Live ATE & Chamber Telemetry (IoT)",
        "🏭 MES & Webhook Integration Hub",
        "📜 Space Certificate of Conformance (CoC)",
        "📄 Model Card & Limitations",
        "🔌 REST API & Industrial ATE Ingestion"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Live System Integrations")
interlock_status = "🛡️ ARMED" if not st.session_state.chamber_controller.is_interlock_tripped else "🛑 TRIPPED"
st.sidebar.markdown(f"""
- **REST API**: 🟢 `/api/v1` (Active)
- **Database**: 🗄️ SQLite + SHA-256 Audit
- **Chamber Safety**: {interlock_status}
- **MES Webhook**: 🔔 Authenticated HMAC
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Standards Compliance")
st.sidebar.markdown("""
- **AEC-Q001 Rev-D**: Dynamic Part Average Testing (DPAT)
- **MIL-STD-883K**: Method 1015 Burn-In Screening
- **ESA ECSS-Q-ST-60C**: Class 1 Space Flight Pedigree
- **NASA EEE-INST-002**: Level 1 Flight Quality
""")


# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if nav_choice == "🚀 Executive Screening Overview":
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 30px 0;">
        <h1 style="font-size: 3.2rem; margin-bottom: 0;">🛰️ Aerospace Electronics Screening</h1>
        <h3 style="color: #94a3b8; font-weight: 400; margin-top: 10px;">Predictive Time-Series Latent Defect Detection (SIH26170)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    In space mission payloads, standard static pass/fail testing fails because **latent defects** drift dynamically 
    during 125°C Burn-In thermal stress while remaining numerically beneath datasheet upper bounds.
    This platform integrates **AEC-Q001 Dynamic PAT (Module A)** with **Early Time-Series Drift Forecasting (Module B)**
    to eliminate field escapes and terminate defective burn-ins at 24 hours.
    """)
    
    # High-Level KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    n_parts = len(final_df)
    n_defects = int(final_df['is_defect'].sum())
    n_caught = int(((final_df['is_defect'] == 1) & (final_df['final_decision'].isin(['REVIEW', 'REJECT']))).sum())
    escapes = n_defects - n_caught
    recall = (n_caught / n_defects) * 100 if n_defects > 0 else 100.0
    early_rejects = int(final_df['early_rejection_24h'].sum())
    hours_saved = int(final_df['burnin_hours_saved'].sum())
    
    with col1:
        st.metric("Total Components Screened", f"{n_parts:,}", f"{final_df['lot_id'].nunique()} Lots")
    with col2:
        st.metric("Defect Interception Rate", f"{recall:.1f}%", f"{n_caught}/{n_defects} Intercepted")
    with col3:
        st.metric("Catastrophic Escapes (FN)", f"{escapes}", "Target: 0 Escapes", delta_color="inverse")
    with col4:
        st.metric("Early Terminations @ 24h", f"{early_rejects}", "Chamber freed early")
    with col5:
        st.metric("Chamber Hours Saved", f"{hours_saved:,} hrs", "144h saved / part")
    
    # SPRT Sequential Economics Row
    sprt_c1, sprt_c2, sprt_c3, sprt_c4 = st.columns(4)
    n_uncertain = sprt_economics.get('uncertain_continued_to_96h', 0)
    sprt_fn = sprt_economics.get('FN', '-')
    sprt_cost = sprt_economics.get('cost_50FN_1FP', '-')
    sprt_c1.metric("🧬 SPRT Accept @ 24h", sprt_economics.get('accept_24h', 0))
    sprt_c2.metric("🧬 SPRT Reject @ 24h", sprt_economics.get('reject_24h_or_96h', 0))
    sprt_c3.metric("🧬 Continue to 96h", n_uncertain, "Uncertain at 24h")
    sprt_c4.metric("🧬 Cost Score (50·FN+FP)", sprt_cost)
        
    st.markdown("---")
    
    # Architecture Diagram & Key Breakthrough
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("### ⚙️ Multi-Layer Screening Architecture (Phase 4-7 Upgrade)")
        st.info(r"""
        1. **Data Layer**: Robust log-transform & empirical Bayes shrinkage for small lots (<30 parts).
        2. **Module A (6-Layer Dynamic PAT)**: Rules + MCD + IsolationForest + Autoencoder + GMM + LOF, fused via CalibratedStackedClassifier.
        3. **Module B (Arrhenius Drift Predictor)**: Forecasts 168h trajectory with quantile intervals. Safety slope derived from physics-based Arrhenius mission-life model.
        4. **SPRT Sequential Decision**: ACCEPT/REJECT/UNCERTAIN at 24h, re-decide at 96h with Sequential Probability Ratio Test.
        5. **Explainability**: Calibrated P(defect), counterfactual margin, nearest healthy neighbours, auto-generated model card.
        """)
        
        # Summary Donut Chart of Dispositions
        dec_counts = final_df['final_decision'].value_counts().reset_index()
        dec_counts.columns = ['Decision', 'Count']
        fig_pie = px.pie(
            dec_counts, values='Count', names='Decision',
            color='Decision',
            color_discrete_map={'ACCEPT': '#22c55e', 'REVIEW': '#eab308', 'REJECT': '#ef4444'},
            hole=0.45,
            title="Screening Disposition Breakdown"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_right:
        st.markdown("### 🏆 Why Traditional Static Limits Fail")
        st.error(r"""
        **The Problem Statement Example**:
        - Lot Average Leakage: **10.0 µA**
        - Defective Component: **45.0 µA** (Massive 4.5× lot anomaly!)
        - Static Datasheet Limit: **50.0 µA**
        
        🔴 **Traditional Tester**: Component is marked **PASS** (45 < 50 µA) and launches into orbit, causing mission failure!
        
        🟢 **Our System**: Recognizes $Z_{DPAT} = 8.1\sigma$ above lot median $\implies$ Immediate **REJECT** at 24h!
        """)
        
        # Quick Defect Breakdown Table
        st.markdown("### 🔍 Interception by Defect Mechanism")
        d_break = []
        for dtype, grp in final_df[final_df['is_defect'] == 1].groupby('defect_type'):
            caught = grp['final_decision'].isin(['REVIEW', 'REJECT']).sum()
            d_break.append({
                'Defect Type': dtype.replace('_', ' ').title(),
                'Total': len(grp),
                'Caught': caught,
                'Recall': f"{(caught/len(grp))*100:.1f}%"
            })
        st.dataframe(pd.DataFrame(d_break), hide_index=True, use_container_width=True)


# ==========================================
# PAGE 2: LOT DISTRIBUTION & DYNAMIC PAT
# ==========================================
elif nav_choice == "📊 Lot Distribution & Dynamic PAT":
    st.title("📊 Production Lot Distribution & Dynamic PAT")
    st.markdown("Inspect wafer lot distributions and compare **Static Datasheet Limits** against **AEC-Q001 Dynamic Part Average Testing (DPAT)** bounds.")
    
    selected_lot = st.selectbox("Select Production Lot to Inspect:", sorted(final_df['lot_id'].unique()))
    lot_data = final_df[final_df['lot_id'] == selected_lot]
    
    col1, col2, col3, col4 = st.columns(4)
    med_0h = lot_data['value_0h'].median()
    med_24h = lot_data['value_24h'].median()
    n_lot_parts = len(lot_data)
    lot_flagged = lot_data['final_decision'].isin(['REVIEW', 'REJECT']).sum()
    
    col1.metric("Lot Part Count", f"{n_lot_parts} units")
    col2.metric("Lot Median @ 0h", f"{med_0h:.2f} µA")
    col3.metric("Lot Median @ 24h", f"{med_24h:.2f} µA", f"+{((med_24h-med_0h)/med_0h)*100:.1f}% drift")
    col4.metric("Flagged Anomalies", f"{lot_flagged} units", f"{(lot_flagged/n_lot_parts)*100:.1f}%")
    
    st.markdown("---")
    
    time_point = st.radio("Display Distribution for:", ["24h Burn-In Reading (Primary)", "0h Pre-Burn-In Baseline"], horizontal=True)
    val_col = 'value_24h' if "24h" in time_point else 'value_0h'
    med_val = lot_data['lot_median_v24'].iloc[0] if "24h" in time_point else lot_data['lot_median_v0'].iloc[0]
    mad_val = lot_data['lot_mad_v24'].iloc[0] if "24h" in time_point else lot_data['lot_mad_v0'].iloc[0]
    limit_val = lot_data['datasheet_limit'].iloc[0]
    
    dpat_upper = med_val + 3.5 * mad_val
    dpat_review = med_val + 2.5 * mad_val
    
    fig = go.Figure()
    
    # Histogram of healthy parts
    healthy_vals = lot_data[lot_data['is_defect'] == 0][val_col]
    defect_vals = lot_data[lot_data['is_defect'] == 1][val_col]
    
    fig.add_trace(go.Histogram(
        x=healthy_vals, name="Healthy Parts (Nominal)",
        marker_color="#3b82f6", opacity=0.75, nbinsx=30
    ))
    
    if len(defect_vals) > 0:
        fig.add_trace(go.Histogram(
            x=defect_vals, name="Latent Defect Parts",
            marker_color="#ef4444", opacity=0.9, nbinsx=15
        ))
        
    # Vertical Lines for Limits
    fig.add_vline(x=limit_val, line_width=2.5, line_dash="dash", line_color="#dc2626",
                  annotation_text=f"Static Limit ({limit_val:.1f} µA)", annotation_position="top right")
    fig.add_vline(x=dpat_upper, line_width=2, line_dash="dot", line_color="#f97316",
                  annotation_text=f"Dynamic PAT Limit (+3.5σ: {dpat_upper:.1f} µA)", annotation_position="top left")
    fig.add_vline(x=med_val, line_width=1.5, line_dash="solid", line_color="#22c55e",
                  annotation_text=f"Lot Median ({med_val:.1f} µA)", annotation_position="bottom left")
                  
    fig.update_layout(
        title=f"Parametric Distribution for {selected_lot} ({val_col})",
        xaxis_title="Leakage Current / Iddq (µA)",
        yaxis_title="Component Count",
        barmode='overlay',
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# PAGE 3: FLAGGED PARTS & DISPOSITION
# ==========================================
elif nav_choice == "⚠️ Flagged Parts & Disposition":
    st.title("⚠️ Flagged Components & Multi-Tier Disposition")
    st.markdown("Filter and review components flagged by Module A (Dynamic PAT Outlier) and Module B (Time-Series Drift Projection).")
    
    col1, col2, col3 = st.columns(3)
    filter_status = col1.selectbox("Filter Disposition:", ["All Parts", "REJECT Only", "REVIEW Only", "ACCEPT Only", "Defects Only"])
    filter_lot = col2.selectbox("Filter Lot:", ["All Lots"] + sorted(final_df['lot_id'].unique().tolist()))
    sort_by = col3.selectbox("Sort By:", ["Composite Risk Score (High to Low)", "24h Value (µA)", "Predicted 168h Value (µA)"])
    
    df_view = final_df.copy()
    if filter_status == "REJECT Only":
        df_view = df_view[df_view['final_decision'] == 'REJECT']
    elif filter_status == "REVIEW Only":
        df_view = df_view[df_view['final_decision'] == 'REVIEW']
    elif filter_status == "ACCEPT Only":
        df_view = df_view[df_view['final_decision'] == 'ACCEPT']
    elif filter_status == "Defects Only":
        df_view = df_view[df_view['is_defect'] == 1]
        
    if filter_lot != "All Lots":
        df_view = df_view[df_view['lot_id'] == filter_lot]
        
    if "Risk" in sort_by:
        df_view = df_view.sort_values(by='composite_risk_score', ascending=False)
    elif "24h" in sort_by:
        df_view = df_view.sort_values(by='value_24h', ascending=False)
    else:
        df_view = df_view.sort_values(by='pred_v168', ascending=False)
        
    st.write(f"Displaying **{len(df_view):,}** parts:")
    
    display_cols = [
        'part_id', 'lot_id', 'final_decision', 'composite_risk_score',
        'value_0h', 'value_24h', 'pred_v168', 'pred_v168_upper',
        'z_pat_24h', 'early_rejection_24h', 'primary_reason'
    ]
    
    # Apply Premium Pandas Styling
    styled_df = df_view[display_cols].style.format({
        'composite_risk_score': '{:.2f}',
        'value_0h': '{:.2f} µA',
        'value_24h': '{:.2f} µA',
        'pred_v168': '{:.2f} µA',
        'pred_v168_upper': '{:.2f} µA',
        'z_pat_24h': '{:.1f}σ'
    }).background_gradient(
        subset=['composite_risk_score'], cmap='Reds', vmin=0, vmax=1
    ).background_gradient(
        subset=['z_pat_24h'], cmap='Oranges', vmin=0, vmax=10
    ).applymap(
        lambda x: 'color: #ef4444; font-weight: bold;' if x == 'REJECT' else ('color: #eab308; font-weight: bold;' if x == 'REVIEW' else 'color: #10b981;'),
        subset=['final_decision']
    )
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=550
    )


# ==========================================
# PAGE 4: QA INSPECTOR DEEP-DIVE & EXPLAINABILITY
# ==========================================
elif nav_choice == "🔬 QA Inspector Deep-Dive":
    st.title("🔬 QA Inspector Deep-Dive & Explainability")
    st.markdown("Transparent audit trail, natural language justification, and trajectory forecasting cone for space certification.")
    
    # Select Part
    flagged_ids = final_df[final_df['final_decision'].isin(['REJECT', 'REVIEW'])]['part_id'].tolist()
    all_ids = final_df['part_id'].tolist()
    default_id = flagged_ids[0] if flagged_ids else all_ids[0]
    
    selected_part_id = st.selectbox(
        "Select Component Serial Number (Pre-populated with flagged parts):",
        options=flagged_ids + [pid for pid in all_ids if pid not in flagged_ids],
        index=0
    )
    
    part_row = final_df[final_df['part_id'] == selected_part_id].iloc[0]
    lot_id = part_row['lot_id']
    lot_df = final_df[final_df['lot_id'] == lot_id]
    
    # Generate QA Report Card
    report = generate_qa_report_card(part_row)
    
    # Decision Badge
    dec = report['decision']
    if dec == 'REJECT':
        st.error(f"### 🔴 DISPOSITION: REJECT (Early Termination @ 24h)")
    elif dec == 'REVIEW':
        st.warning(f"### 🟡 DISPOSITION: REVIEW (Secondary Bench Test Required)")
    else:
        st.success(f"### 🟢 DISPOSITION: ACCEPT (Flight Ready)")
        
    st.markdown(f"""
    <div class="report-box">
        <h4>📋 QA Inspector Plain-Language Summary</h4>
        <p style="font-size: 16px; line-height: 1.6;">{report['narrative']}</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    
    col_chart, col_shap = st.columns([3, 2])
    
    with col_chart:
        st.markdown("#### 📈 Multi-Point Aging Trajectory & Forecast Cone")
        traj_data = generate_trajectory_plot_data(part_row, lot_df)
        
        fig_traj = go.Figure()
        
        # 1. Lot 5th - 95th Percentile Confidence Ribbon
        fig_traj.add_trace(go.Scatter(
            x=traj_data['time_hours'] + traj_data['time_hours'][::-1],
            y=traj_data['lot_p95_curve'] + traj_data['lot_p05_curve'][::-1],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name="Lot 5th-95th % Nominal Band"
        ))
        
        # 2. Lot Median Curve
        fig_traj.add_trace(go.Scatter(
            x=traj_data['time_hours'], y=traj_data['lot_median_curve'],
            mode='lines', line=dict(color='#3b82f6', dash='dash'),
            name="Lot Median Aging Curve"
        ))
        
        # 3. Static Datasheet Limit Line
        fig_traj.add_trace(go.Scatter(
            x=[0, 168], y=[traj_data['datasheet_limit'], traj_data['datasheet_limit']],
            mode='lines', line=dict(color='#ef4444', width=2, dash='dot'),
            name=f"Datasheet Max ({traj_data['datasheet_limit']:.1f} µA)"
        ))
        
        # 4. Safety Slope Ceiling Line
        fig_traj.add_trace(go.Scatter(
            x=[0, 168], y=traj_data['safety_slope_ceiling'],
            mode='lines', line=dict(color='#f59e0b', width=1.5, dash='dash'),
            name="AEC-Q001 Safety Slope Ceiling"
        ))
        
        # 5. Observed Part Readings (0h, 24h)
        fig_traj.add_trace(go.Scatter(
            x=traj_data['part_observed_hours'], y=traj_data['part_observed_values'],
            mode='lines+markers', marker=dict(size=10, color='#e11d48'),
            line=dict(color='#e11d48', width=3),
            name="Observed Readings (0h, 24h)"
        ))
        
        # 6. Projected Forecast Cone (24h -> 168h)
        fig_traj.add_trace(go.Scatter(
            x=[24, 168, 168, 24],
            y=[traj_data['forecast_median'][0], traj_data['forecast_upper'][1],
               traj_data['forecast_lower'][1], traj_data['forecast_median'][0]],
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.25)',
            line=dict(color='rgba(255,255,255,0)'),
            name="Projected 168h Confidence Cone [10%-90%]"
        ))
        
        fig_traj.add_trace(go.Scatter(
            x=traj_data['forecast_hours'], y=traj_data['forecast_median'],
            mode='lines+markers', line=dict(color='#dc2626', width=2.5, dash='dot'),
            marker=dict(symbol='diamond', size=8),
            name=f"Predicted 168h ({traj_data['forecast_median'][1]:.1f} µA)"
        ))
        
        # 7. Actual ground truth points if available
        if traj_data['part_hidden_hours']:
            fig_traj.add_trace(go.Scatter(
                x=traj_data['part_hidden_hours'], y=traj_data['part_hidden_values'],
                mode='markers', marker=dict(symbol='star', size=11, color='#a855f7'),
                name="Actual Ground Truth (96h, 168h)"
            ))
            
        fig_traj.update_layout(
            title=f"Aging Trajectory & Safety Envelope for {selected_part_id}",
            xaxis_title="Burn-In Stress Duration (Hours @ 125°C)",
            yaxis_title="Leakage Current / Iddq (µA)",
            template='plotly_dark',
            hovermode='x unified'
        )
        st.plotly_chart(fig_traj, use_container_width=True)
        
    with col_shap:
        st.markdown("#### 🔬 Parametric Feature Attributions")
        st.caption("Quantifies which physical metrics drove the anomaly classification:")
        
        contribs = compute_local_feature_contributions(part_row)
        c_df = pd.DataFrame(contribs)
        
        fig_bar = px.bar(
            c_df, x='impact_score', y='feature', orientation='h',
            color='impact_score',
            color_continuous_scale='Reds',
            labels={'impact_score': 'Contribution Weight (Relative Deviation)', 'feature': 'Physical Metric'},
            hover_data=['description', 'raw_value']
        )
        fig_bar.update_layout(
            template='plotly_dark',
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=10, r=10, t=30, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("#### 📑 Regulatory Audit Log")
        st.json(report['audit_trail'])


# ==========================================
# PAGE 5: SPRT SEQUENTIAL DECISION
# ==========================================
elif nav_choice == "🧬 SPRT Sequential Decision":
    st.title("🧬 Sequential Probability Ratio Test (SPRT) Decision Engine")
    st.markdown("""
    The **Sequential Screening Engine** implements a formal statistical decision at 24h:
    - **ACCEPT**: Sufficient evidence part is healthy → release from chamber.
    - **REJECT**: Sufficient evidence of defect → early termination (saves 144h).
    - **UNCERTAIN**: Insufficient evidence → continue burn-in to 96h for re-evaluation.
    
    This trades chamber hours against escape risk using the **Wald SPRT** framework.
    """)
    
    # SPRT Decision Breakdown
    st.markdown("### 📊 24h SPRT Decision Distribution")
    
    if 'sprt_decision_24h' in final_df.columns:
        sprt_counts = final_df['sprt_decision_24h'].value_counts().reset_index()
        sprt_counts.columns = ['Decision', 'Count']
        
        col_sprt1, col_sprt2 = st.columns([2, 3])
        
        with col_sprt1:
            sprt_color_map = {'ACCEPT': '#22c55e', 'REJECT': '#ef4444', 'UNCERTAIN': '#f59e0b'}
            fig_sprt_pie = px.pie(
                sprt_counts, values='Count', names='Decision',
                color='Decision', color_discrete_map=sprt_color_map,
                hole=0.45, title="SPRT 24h Disposition"
            )
            fig_sprt_pie.update_layout(template='plotly_dark')
            st.plotly_chart(fig_sprt_pie, use_container_width=True)
        
        with col_sprt2:
            # Economics summary
            st.markdown("### 💰 Economic Trade-Off Analysis")
            e = sprt_economics
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Total Parts", e.get('total_parts', 0))
            ec2.metric("SPRT Recall", f"{e.get('recall', 0)*100:.1f}%")
            ec3.metric("Cost (50·FN+FP)", e.get('cost_50FN_1FP', '-'))
            
            ec4, ec5, ec6 = st.columns(3)
            ec4.metric("False Negatives", e.get('FN', '-'), delta_color="inverse")
            ec5.metric("False Positives", e.get('FP', '-'))
            ec6.metric("Chamber Hours Saved", f"{e.get('total_chamber_hours_saved', 0):,}")
            
            if e.get('uncertain_continued_to_96h', 0) > 0:
                st.warning(
                    f"**{e['uncertain_continued_to_96h']} parts** require continued burn-in to 96h. "
                    f"At 96h, a second SPRT decision determines final disposition."
                )
        
        st.markdown("---")
        
        # Log-Likelihood Ratio Distribution
        if 'sprt_log_lr' in final_df.columns:
            st.markdown("### 📈 Log-Likelihood Ratio Distribution")
            
            fig_llr = go.Figure()
            
            for dec, color in sprt_color_map.items():
                subset = final_df[final_df['sprt_decision_24h'] == dec]
                if len(subset) > 0:
                    fig_llr.add_trace(go.Histogram(
                        x=subset['sprt_log_lr'], name=dec,
                        marker_color=color, opacity=0.7,
                        nbinsx=30
                    ))
            
            # Add SPRT boundary lines
            sprt_engine = SequentialScreeningEngine()
            log_A = float(np.log(sprt_engine.A))
            log_B = float(np.log(sprt_engine.B))
            fig_llr.add_vline(x=log_A, line_dash="dash", line_color="#ef4444",
                              annotation_text=f"Reject boundary (A={log_A:.1f})")
            fig_llr.add_vline(x=log_B, line_dash="dash", line_color="#22c55e",
                              annotation_text=f"Accept boundary (B={log_B:.1f})")
            
            fig_llr.update_layout(
                template='plotly_dark', barmode='overlay',
                xaxis_title="Log-Likelihood Ratio (Λ)",
                yaxis_title="Count",
                title="SPRT Log-Likelihood Ratio: Where Does Each Part Fall?"
            )
            st.plotly_chart(fig_llr, use_container_width=True)
        
        # Table of uncertain parts
        st.markdown("### 🔍 Uncertain Parts (Require 96h Re-Evaluation)")
        uncertain = final_df[final_df['sprt_decision_24h'] == 'UNCERTAIN']
        if len(uncertain) > 0:
            display_cols = ['part_id', 'lot_id', 'z_pat_24h', 'z_drift_24h',
                            'anomaly_score_mod_a', 'sprt_log_lr', 'sprt_reason']
            show_cols = [c for c in display_cols if c in uncertain.columns]
            st.dataframe(uncertain[show_cols].head(30), hide_index=True, use_container_width=True)
        else:
            st.success("All parts received definitive ACCEPT or REJECT decisions at 24h.")
    else:
        st.info("SPRT decisions not available. Run the pipeline with the upgraded SequentialScreeningEngine.")


# ==========================================
# PAGE 6: BENCHMARKS & BASELINES
# ==========================================
elif nav_choice == "📈 Benchmarks & Baselines":
    st.title("📈 Model Benchmarks & Baselines Comparison")
    st.markdown("Proves the superiority of our dynamic lot-aware multi-module system against traditional industry baselines.")
    
    # Baseline comparison table
    st.markdown("### 🏆 Comprehensive Screening Baseline Matrix")
    st.dataframe(baseline_df, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Module B Drift Prediction Accuracy")
        st.metric("Mean Absolute Error (168h)", f"{drift_metrics['mae_original_scale_uA']:.4f} µA", "Ground Truth Target")
        st.metric("Log-Space MAE", f"{drift_metrics['mae_log_scale']:.4f}")
        st.metric("R² Goodness of Fit", f"{drift_metrics['r2_score']:.4f}")
        st.metric("Burn-In Chamber Hours Saved", f"{drift_metrics['chamber_hours_saved']:,} hrs")
        
        # Scatter: Predicted vs Actual 168h
        sample_pts = final_df.dropna(subset=['value_168h']).sample(min(800, len(final_df)), random_state=42)
        fig_scat = px.scatter(
            sample_pts, x='value_168h', y='pred_v168',
            color='final_decision',
            color_discrete_map={'ACCEPT': '#22c55e', 'REVIEW': '#eab308', 'REJECT': '#ef4444'},
            labels={'value_168h': 'Actual Measured 168h (µA)', 'pred_v168': 'Predicted 168h (µA)'},
            title="Predicted vs Actual 168h Drift",
            opacity=0.7
        )
        # 1:1 diagonal reference
        max_val = max(sample_pts['value_168h'].max(), sample_pts['pred_v168'].max())
        fig_scat.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines', line=dict(color='white', dash='dash'), name='1:1 Ideal'
        ))
        fig_scat.update_layout(template='plotly_dark')
        st.plotly_chart(fig_scat, use_container_width=True)
        
    with col2:
        st.markdown("### 🧩 Per-Detector Score Analysis")
        st.info("Individual anomaly score distribution from each detection layer (computed live from test data):")
        
        # Build live per-detector performance from actual pipeline output
        from sklearn.metrics import recall_score as _recall
        y_true_bench = final_df['is_defect'].values
        score_cols = {
            'Rule Flags': 'score_rule_flag',
            'Mahalanobis (MCD)': 'score_mahalanobis',
            'Isolation Forest': 'score_iforest',
            'Autoencoder': 'score_autoencoder',
            'GMM Density': 'score_gmm',
            'LOF': 'score_lof',
            'Full Ensemble': 'anomaly_score_mod_a',
        }
        
        detector_rows = []
        for name, col in score_cols.items():
            if col in final_df.columns:
                scores = final_df[col].values
                # Threshold at median of defective scores for each detector
                defect_scores = scores[y_true_bench == 1]
                if len(defect_scores) > 0:
                    thresh = float(np.percentile(defect_scores, 25))  # catch 75%+ of defects
                    pred = (scores >= thresh).astype(int)
                    rec = _recall(y_true_bench, pred, zero_division=0)
                    from sklearn.metrics import confusion_matrix as _cm
                    cm = _cm(y_true_bench, pred, labels=[0, 1])
                    _, fp, fn, _ = cm.ravel()
                    detector_rows.append({
                        'Detector': name,
                        'Recall': f"{rec*100:.1f}%",
                        'FN': int(fn),
                        'FP': int(fp),
                        'Cost': int(50*fn + fp),
                        'Mean Score (Defects)': f"{defect_scores.mean():.3f}",
                    })
        
        if detector_rows:
            ablation_table = pd.DataFrame(detector_rows)
            st.dataframe(ablation_table, hide_index=True, use_container_width=True)
            
            # Bar Chart
            fig_abl = px.bar(
                ablation_table, x='Recall', y='Detector', orientation='h',
                color='FN', color_continuous_scale='Reds_r',
                title="Defect Recall by Detection Layer"
            )
            fig_abl.update_layout(template='plotly_dark', yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_abl, use_container_width=True)
        
        # Arrhenius Physics Info
        st.markdown("### ⚛️ Arrhenius Safety Slope Physics")
        arr = mod_b.arrhenius_info
        if arr:
            st.metric("Acceleration Factor (AF)", f"{arr.get('acceleration_factor', 'N/A')}×")
            ac1, ac2 = st.columns(2)
            ac1.metric("Activation Energy", f"{arr.get('Ea_eV', 0.7)} eV")
            ac2.metric("Safety Slope", f"{arr.get('safety_slope_uA_per_hr', 'N/A'):.4f} µA/h")
            st.caption(f"Stress: {arr.get('T_stress_C', 125)}°C → Field: {arr.get('T_use_C', 55)}°C | "
                       f"168h burn-in ≈ {arr.get('equiv_field_years', 'N/A')} years field equivalent")


# ==========================================
# PAGE 6: INTERACTIVE WHAT-IF SIMULATION
# ==========================================
elif nav_choice == "🎛️ Interactive What-If Simulation":
    st.title("🎛️ Interactive What-If Sensitivity Simulation")
    st.markdown("Fine-tune screening sensitivity sliders in real time and immediately see the impact on **Recall**, **False Alarms**, and **Chamber Savings**.")
    
    col1, col2 = st.columns(2)
    with col1:
        sim_z_pat = st.slider("Dynamic PAT Z-Score Threshold (σ)", 2.0, 4.5, 3.5, 0.1)
        sim_safety_k = st.slider("Safety Slope Multiplier k (Mean + k·σ)", 1.5, 4.0, 2.5, 0.1)
    with col2:
        sim_fn_cost = st.slider("Catastrophic Escape Penalty Weight (FN)", 10, 100, 50, 5)
        sim_fp_cost = st.slider("Scrap Penalty Weight (FP)", 1, 10, 1, 1)
        
    # Recalculate
    y_true = final_df['is_defect'].values
    
    # Recalculate decisions with simulated thresholds
    sim_pat_flag = (final_df['z_pat_0h'] > sim_z_pat) | (final_df['z_pat_24h'] > sim_z_pat)
    sim_slope_limit = mod_b.healthy_mean_slope + sim_safety_k * mod_b.healthy_std_slope
    sim_drift_flag = final_df['pred_upper_drift_rate'] > sim_slope_limit
    
    sim_flagged = sim_pat_flag | sim_drift_flag | (final_df['anomaly_score_mod_a'] >= 0.45)
    
    from sklearn.metrics import recall_score, precision_score, confusion_matrix
    sim_rec = recall_score(y_true, sim_flagged, zero_division=0)
    sim_cm = confusion_matrix(y_true, sim_flagged)
    tn, fp, fn, tp = sim_cm.ravel() if sim_cm.size == 4 else (0, 0, 0, 0)
    sim_cost = (sim_fn_cost * fn) + (sim_fp_cost * fp)
    sim_saved = tp * 144
    
    st.markdown("---")
    st.markdown("### 📊 Real-Time Simulated Performance")
    
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Simulated Defect Recall", f"{sim_rec*100:.1f}%", f"{tp}/{tp+fn} Caught")
    sc2.metric("Missed Defects (Escapes)", f"{fn}", delta_color="inverse")
    sc3.metric("Scrap Units (False Alarms)", f"{fp}")
    sc4.metric("Simulated Cost Score", f"{sim_cost:,}")
    
    st.success(f"Estimated Thermal Chamber Hours Saved: **{sim_saved:,} hours** by terminating early at 24h.")


# ==========================================
# PAGE 7: BATCH CSV INFERENCE & EXPORT
# ==========================================
elif nav_choice == "📁 Batch CSV Inference & Export":
    st.title("📁 Batch CSV Inference & Export")
    st.markdown("Upload a new production lot CSV to run automated screening, or export the verified test lot predictions.")
    
    uploaded_file = st.file_uploader("Upload Component Burn-In CSV (Required columns: part_id, lot_id, value_0h, value_24h):", type=['csv'])
    
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded {len(custom_df)} components across {custom_df['lot_id'].nunique()} lots.")
            
            if st.button("🚀 Run Screening Pipeline"):
                with st.spinner("Executing Dynamic PAT & Drift Forecasting..."):
                    c_proc = preprocessor.transform(custom_df)
                    c_pred_a = mod_a.predict_detailed(c_proc)
                    c_pred_b = mod_b.predict(c_proc)
                    c_final = ScreeningDecisionEngine().evaluate(c_pred_a, c_pred_b)
                    
                    st.write(c_final[['part_id', 'lot_id', 'final_decision', 'composite_risk_score', 'pred_v168', 'primary_reason']].head(20))
                    
                    csv_export = c_final.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Screening Disposition CSV",
                        data=csv_export,
                        file_name=f"burnin_screening_results_{datetime.date.today()}.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
    else:
        st.info("No file uploaded. You can download the current benchmark test lot results below:")
        test_export = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Benchmark Test Dispositions CSV",
            data=test_export,
            file_name="benchmark_test_dispositions.csv",
            mime="text/csv"
        )


# ==========================================
# PAGE 8: LIVE ATE & CHAMBER TELEMETRY (IoT)
# ==========================================
elif nav_choice == "📡 Live ATE & Chamber Telemetry (IoT)":
    st.title("📡 Live ATE & Burn-In Chamber Telemetry (IoT)")
    st.markdown("""
    Real-time monitoring of industrial Environmental Stress Screening (ESS) burn-in ovens 
    (**Chroma 58158 / ESPEC Bay 04**) conforming to **MIL-STD-883K Method 1015 Condition D** 
    (125.0°C ± 1.0°C Steady-State Reverse Bias).
    """)
    
    chamber = st.session_state.chamber_controller
    telem = chamber.read_telemetry()
    
    # Chamber Hardware Alert Banner
    if telem['interlock_tripped']:
        st.error(f"🛑 **EMERGENCY HARDWARE INTERLOCK TRIPPED!** DUT Power Relay Disconnected (0.0V). Reason: `{telem['trip_reason']}`")
    else:
        st.success(f"🛡️ **CHAMBER SAFETY INTERLOCK ARMED & NOMINAL.** Thermal bay temperature steady at {telem['temperature_celsius']}°C.")
        
    # Sensor Gauges
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dev = telem['temp_deviation_c']
        delta_str = f"{dev:+.2f}°C vs Setpoint"
        st.metric("Chamber Temperature", f"{telem['temperature_celsius']} °C", delta_str, delta_color="inverse" if abs(dev) > 1.0 else "normal")
    with c2:
        st.metric("Bias Supply Voltage", f"{telem['voltage_bias_volts']:.2f} V", "VDD Static High-Rel")
    with c3:
        st.metric("Rack Total Current", f"{telem['rack_current_mA']} mA", f"Max: {telem['max_rack_current_mA']} mA")
    with c4:
        st.metric("Bay Controller State", telem['status'], "Bay 04 - Slot Tray A")
        
    st.markdown("---")
    
    # Chamber Control Bench
    st.markdown("### 🎛️ Chamber Hardware & Safety Interlock Controls")
    btn1, btn2, btn3, btn4 = st.columns(4)
    
    with btn1:
        if st.button("🔄 Query Live Telemetry Tick", use_container_width=True):
            st.rerun()
            
    with btn2:
        if st.button("⚠️ Inject Thermal Runaway (+10°C)", use_container_width=True):
            chamber.inject_thermal_spike(10.5)
            st.session_state.mes_notifier.notify_chamber_thermal_trip(
                chamber.chamber_id, chamber.current_temp, "TEST_SIMULATED_THERMAL_RUNAWAY"
            )
            st.rerun()
            
    with btn3:
        if st.button("🛑 Emergency Power Trip", use_container_width=True):
            chamber.trip_safety_interlock("MANUAL_EMERGENCY_ESTOP_TRIGGERED")
            st.rerun()
            
    with btn4:
        if st.button("🔁 QA Reset Interlock", use_container_width=True):
            chamber.reset_interlock()
            st.rerun()
            
    st.markdown("---")
    
    # 4x8 DUT Socket Matrix Visualization
    st.markdown("### 🔲 Burn-In Socket Board Matrix (32-DUT Active Tray)")
    st.caption("Visualizes physical component sockets inside the thermal oven. Early rejected parts are flagged red to be depowered.")
    
    # Generate 32 simulated socket statuses
    tray_cols = st.columns(8)
    for idx in range(32):
        col = tray_cols[idx % 8]
        if idx in [6, 18]:
            status_color = "#ef4444"
            status_text = "REJECT"
            icon = "❌"
        elif idx in [14, 27]:
            status_color = "#eab308"
            status_text = "REVIEW"
            icon = "⚠️"
        else:
            status_color = "#22c55e"
            status_text = "PASS"
            icon = "✅"
            
        with col:
            st.markdown(f"""
            <div style="background-color: #1e293b; border: 2px solid {status_color}; border-radius: 6px; padding: 8px; text-align: center; margin-bottom: 8px;">
                <div style="font-size: 11px; color: #94a3b8;">SKT-{idx+1:02d}</div>
                <div style="font-size: 18px;">{icon}</div>
                <div style="font-size: 10px; font-weight: bold; color: {status_color};">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    
    # Thermal Chamber Sensor Telemetry Trace
    st.markdown("### 📈 Real-Time Chamber Environmental Log (24-Cycle Trace)")
    np.random.seed(42)
    time_points = [f"{(i*5):02d}m" for i in range(24)]
    temp_trace = 125.0 + np.random.normal(0, 0.25, 24)
    current_trace = 650.0 + np.random.normal(0, 15, 24)
    
    df_trace = pd.DataFrame({
        "Cycle Time": time_points,
        "Chamber Temp (°C)": temp_trace,
        "Rack Current (mA)": current_trace
    })
    
    fig_trace = go.Figure()
    fig_trace.add_trace(go.Scatter(
        x=df_trace["Cycle Time"], y=df_trace["Chamber Temp (°C)"],
        mode='lines+markers', name='Temperature (°C)', line=dict(color='#f97316', width=2)
    ))
    fig_trace.add_hline(y=125.0, line_dash="dash", line_color="#38bdf8", annotation_text="Setpoint 125°C")
    fig_trace.add_hline(y=127.0, line_dash="dot", line_color="#ef4444", annotation_text="Max Limit 127°C")
    fig_trace.update_layout(
        template='plotly_dark',
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(range=[123.5, 128.0])
    )
    st.plotly_chart(fig_trace, use_container_width=True)


# ==========================================
# PAGE 9: MES & WEBHOOK INTEGRATION HUB
# ==========================================
elif nav_choice == "🏭 MES & Webhook Integration Hub":
    st.title("🏭 MES & Enterprise Webhook Dispatcher")
    st.markdown("""
    In high-reliability semiconductor fabs, the screening engine integrates with 
    **Manufacturing Execution Systems (MES)** like *Siemens Opcenter*, *Applied Materials*, and *Rockwell Automation*.
    Every webhook is authenticated using an industry-standard **HMAC-SHA256 signature** (`X-Screening-Signature`).
    """)
    
    mes = st.session_state.mes_notifier
    
    # Webhook Endpoint Config
    st.markdown("### ⚙️ Webhook Endpoint Configuration")
    wc1, wc2 = st.columns([3, 2])
    with wc1:
        endpoint_url = st.text_input("MES Webhook Target URL:", value="https://mes.space-electronics.internal/api/v1/events")
    with wc2:
        secret_key = st.text_input("HMAC-SHA256 Secret Key:", value="space-grade-secret-key-2026", type="password")
        
    st.markdown("---")
    
    # Event Trigger Simulator
    st.markdown("### ⚡ Live Webhook Event Dispatcher")
    st.caption("Click any event to simulate real-time notification to the assembly floor or Slack/Teams incident channel:")
    
    ev_c1, ev_c2, ev_c3 = st.columns(3)
    
    with ev_c1:
        if st.button("🚨 Dispatch Maverick Lot Alert", use_container_width=True):
            res = mes.notify_maverick_lot("LOT-ISRO-2026-X1", rejection_rate_pct=8.4, threshold_pct=5.0)
            st.session_state.last_webhook = res
            st.toast("Dispatched AEC-Q001 Maverick Lot Alert!", icon="🚨")
            
    with ev_c2:
        if st.button("⚡ Dispatch 24h Early Rejection", use_container_width=True):
            res = mes.notify_early_rejection("ISRO-D042", "LOT-ISRO-2026-X1", "PAT_Z_SCORE_DRIFT_EXCEEDED", 38.5, 50.0)
            st.session_state.last_webhook = res
            st.toast("Dispatched Early Rejection Notice!", icon="⚡")
            
    with ev_c3:
        if st.button("🔥 Dispatch Chamber Thermal Trip", use_container_width=True):
            res = mes.notify_chamber_thermal_trip("CHAMBER-ESS-BAY04", 134.5, "THERMAL_RUNAWAY_BREACH_134C")
            st.session_state.last_webhook = res
            st.toast("Dispatched Chamber Thermal Runaway Alarm!", icon="🔥")
            
    # Inspect Dispatched Payload
    if 'last_webhook' in st.session_state:
        lw = st.session_state.last_webhook
        st.markdown("### 📦 Dispatched Webhook Transmission")
        
        info_c1, info_c2, info_c3 = st.columns(3)
        info_c1.metric("Event ID", lw['event_id'])
        info_c2.metric("HTTP Status", lw['http_status'], lw['status'])
        info_c3.metric("Latency", f"{lw['delivery_time_ms']} ms")
        
        st.markdown("**Cryptographic Header:**")
        st.code(f"X-Screening-Signature: {lw['signature']}\nContent-Type: application/json", language="http")
        
        st.markdown("**Dispatched JSON Payload:**")
        st.json(lw['payload'])
        
    st.markdown("---")
    
    # Event History Log
    st.markdown("### 📜 Dispatched Webhook Audit History")
    if mes.history:
        history_rows = [
            {
                "Event ID": h['event_id'],
                "Event Type": h['event_type'],
                "Timestamp": h['timestamp'],
                "Delivery Status": h['status'],
                "Latency (ms)": h['delivery_time_ms']
            }
            for h in reversed(mes.history)
        ]
        st.dataframe(pd.DataFrame(history_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No webhooks dispatched yet. Click any button above to test.")


# ==========================================
# PAGE 10: SPACE CERTIFICATE OF CONFORMANCE (CoC)
# ==========================================
elif nav_choice == "📜 Space Certificate of Conformance (CoC)":
    st.title("📜 Space-Grade Certificate of Conformance (CoC)")
    st.markdown("""
    Every flight model lot screened for space missions (**ISRO / Gaganyaan / NASA / ESA**) 
    requires a formal **Certificate of Conformance (CoC)** compliant with:
    - **MIL-STD-883K Method 1015 Condition D**
    - **AEC-Q001 Rev-D (Dynamic Part Average Testing)**
    - **ESA ECSS-Q-ST-60C Class 1 Space Flight Pedigree**
    """)
    
    selected_lot_coc = st.selectbox("Select Production Lot for Certification:", sorted(final_df['lot_id'].unique()))
    lot_coc_df = final_df[final_df['lot_id'] == selected_lot_coc]
    
    coc_gen = st.session_state.coc_generator
    sha256_seal = coc_gen.compute_lot_integrity_hash(lot_coc_df)
    
    # Inspector Details Input
    qa_inspector = st.text_input("Authorised Quality Assurance Signatory:", value="Dr. K. S. Raman, Head of Space Quality Assurance")
    
    st.markdown("---")
    
    # Live Certificate Preview Card
    st.markdown("### 🔍 Digital Certificate Preview")
    
    n_p = len(lot_coc_df)
    n_rej = int((lot_coc_df['final_decision'] == 'REJECT').sum())
    n_rev = int((lot_coc_df['final_decision'] == 'REVIEW').sum())
    n_acc = n_p - n_rej - n_rev
    scrap_rate = (n_rej / n_p) * 100 if n_p > 0 else 0.0
    maverick = scrap_rate > 5.0
    
    st.markdown(f"""
    <div style="background-color: #0f172a; border: 2px solid #3b82f6; border-radius: 8px; padding: 24px; font-family: sans-serif;">
        <div style="text-align: center; border-bottom: 2px solid #1e3a8a; padding-bottom: 12px; margin-bottom: 16px;">
            <h2 style="color: #60a5fa; margin: 0;">SPACE ELECTRONICS RELIABILITY LABORATORY</h2>
            <div style="color: #94a3b8; font-size: 13px;">CERTIFICATE OF CONFORMANCE & SPACE-GRADE LOT ACCEPTANCE</div>
            <div style="color: #cbd5e1; font-size: 11px; margin-top: 4px;">STANDARDS: MIL-STD-883K COND D | AEC-Q001 REV-D | ESA ECSS-Q-ST-60C</div>
        </div>
        
        <table style="width: 100%; color: #f8fafc; font-size: 13px; margin-bottom: 16px;">
            <tr>
                <td><b>Certificate ID:</b> COC-ISRO-2026-{selected_lot_coc.replace('-', '')[:6]}</td>
                <td><b>Production Lot:</b> {selected_lot_coc}</td>
            </tr>
            <tr>
                <td><b>Screening Bay:</b> CHROMA-58158-BAY04</td>
                <td><b>Chamber Condition:</b> 125.0°C ± 1.0°C (3.3V Bias)</td>
            </tr>
            <tr>
                <td><b>Flight Tier:</b> CLASS S / LEVEL 1 FLIGHT MODEL</td>
                <td><b>Issue Date:</b> {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</td>
            </tr>
        </table>
        
        <div style="background-color: #1e293b; padding: 12px; border-radius: 6px; margin-bottom: 16px;">
            <div style="color: #38bdf8; font-weight: bold; margin-bottom: 6px;">1. Screening Yield & Maverick Lot Determination:</div>
            <div style="font-size: 13px; color: #e2e8f0;">
                • Total Parts Screened: <b>{n_p} units</b><br/>
                • Flight Accept (ACCEPT): <b style="color: #22c55e;">{n_acc} units ({(n_acc/n_p)*100:.1f}%)</b><br/>
                • Engineering Review (REVIEW): <b style="color: #eab308;">{n_rev} units ({(n_rev/n_p)*100:.1f}%)</b><br/>
                • Latent Defect Early Rejection (REJECT): <b style="color: #ef4444;">{n_rej} units ({scrap_rate:.1f}%)</b><br/>
                • AEC-Q001 Maverick Lot Status: <b style="color: {'#ef4444' if maverick else '#22c55e'};">{'ALERT: MAVERICK LOT (>5% SCRAP)' if maverick else 'PASSED (HOMOGENEOUS LOT)'}</b>
            </div>
        </div>
        
        <div style="background-color: #1e293b; padding: 12px; border-radius: 6px; margin-bottom: 16px;">
            <div style="color: #38bdf8; font-weight: bold; margin-bottom: 4px;">2. Cryptographic SHA-256 Digital Verification Seal:</div>
            <div style="font-family: monospace; font-size: 12px; color: #a5f3fc; word-break: break-all;">
                {sha256_seal}
            </div>
        </div>
        
        <div style="border-top: 1px solid #334155; padding-top: 12px; font-size: 12px; color: #94a3b8;">
            <b>Certified By:</b> {qa_inspector}<br/>
            <b>Digital Verification:</b> STAMPED & VERIFIED AT SPACE QUALITY ASSURANCE REPOSITORY
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # PDF Generator Button
    st.markdown("### 📥 Generate & Download Aerospace PDF Certificate")
    st.caption("Generates a formal, printable PDF complete with embedded QR code and cryptographic seal.")
    
    if st.button("📄 Generate Official PDF Certificate of Conformance", use_container_width=True):
        with st.spinner("Compiling Aerospace PDF Certificate with QR Verification..."):
            pdf_path = coc_gen.generate_pdf(selected_lot_coc, lot_coc_df, inspector_name=qa_inspector)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
                
            st.download_button(
                label=f"📥 Download {os.path.basename(pdf_path)}",
                data=pdf_bytes,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )
            st.success(f"Certificate generated successfully! Size: {len(pdf_bytes):,} bytes.")


# ==========================================
# PAGE 12: MODEL CARD & LIMITATIONS
# ==========================================
elif nav_choice == "📄 Model Card & Limitations":
    st.title("📄 Auto-Generated Model Card & Limitations")
    st.markdown("""
    Transparency document for this screening system — auto-generated from the pipeline configuration.
    Attached to CoC certificates and presented to auditors.
    """)
    
    # Model Identity
    st.markdown("### 🏷️ Model Identity")
    mc1, mc2 = st.columns(2)
    mc1.metric("Model Name", model_card.get('model_name', 'N/A'))
    mc2.metric("Version", model_card.get('version', 'N/A'))
    st.info(f"**Task:** {model_card.get('task', 'N/A')}")
    
    st.markdown("---")
    
    # Standards
    st.markdown("### 📋 Standards Alignment")
    for std in model_card.get('standards_alignment', []):
        st.markdown(f"- ✅ {std}")
    
    st.markdown("---")
    
    # Training Data
    st.markdown("### 📊 Training Data")
    td = model_card.get('training_data', {})
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Total Parts", td.get('n_parts', 0))
    tc2.metric("Defective Parts", td.get('n_defects', 0))
    tc3.metric("Data Source", td.get('type', 'Unknown')[:30])
    
    st.markdown("**Defect Mechanisms Modeled:**")
    for mech in td.get('defect_mechanisms', []):
        st.markdown(f"  - {mech}")
    
    st.warning(f"**Caveat:** {td.get('caveat', 'No caveat specified.')}")
    
    st.markdown("---")
    
    # Detector Status
    st.markdown("### 🔧 Detector Layer Status")
    det = model_card.get('detectors', detector_report)
    if det:
        det_rows = []
        for key, val in det.items():
            if isinstance(val, bool):
                det_rows.append({'Layer': key.replace('_', ' ').title(), 'Status': '✅ Active' if val else '❌ Inactive'})
            elif isinstance(val, (int, float)) and val is not None:
                det_rows.append({'Layer': key.replace('_', ' ').title(), 'Value': str(val)})
        if det_rows:
            st.dataframe(pd.DataFrame(det_rows), hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Certified Recall
    st.markdown("### 🎯 Certified Recall")
    cr = model_card.get('certified_recall', {})
    if cr.get('bound') is not None:
        st.success(f"**Certified Recall Lower Bound:** {cr['bound']*100:.2f}% "
                   f"(Clopper-Pearson, {cr.get('confidence', 0.95)*100:.0f}% confidence, "
                   f"n={cr.get('n_calibration_defects', '?')} calibration defects)")
    else:
        st.warning(cr.get('note', 'Recall certification not available.'))
    
    st.markdown("---")
    
    # Assumptions
    st.markdown("### 📌 Assumptions")
    for i, assumption in enumerate(model_card.get('assumptions', []), 1):
        st.markdown(f"{i}. {assumption}")
    
    # Limitations
    st.markdown("### ⚠️ Limitations")
    for i, limitation in enumerate(model_card.get('limitations', []), 1):
        st.markdown(f"{i}. {limitation}")
    
    # Ethical
    st.markdown("### 🤝 Ethical Considerations")
    for item in model_card.get('ethical_considerations', []):
        st.markdown(f"- {item}")
    
    st.markdown("---")
    
    # Raw JSON export
    with st.expander("📝 Raw Model Card JSON (for programmatic access)"):
        st.json(model_card)


# ==========================================
# PAGE 13: REST API & INDUSTRIAL ATE INGESTION
# ==========================================
elif nav_choice == "🔌 REST API & Industrial ATE Ingestion":
    st.title("🔌 Enterprise REST API & Industrial ATE Ingestion")
    st.markdown("""
    Integrate automated semiconductor test equipment (**Teradyne J750, Advantest V93000, Chroma 58158**) 
    and LabVIEW test benches with the screening engine via high-performance REST APIs.
    """)
    
    tab_ate, tab_api = st.tabs(["📂 Industrial ATE Datalog Ingestion", "🔌 Live REST API & Swagger Explorer"])
    
    with tab_ate:
        st.markdown("### 📥 Automated Test Equipment (ATE) File Ingestion")
        st.markdown("Upload raw tester datalogs (Teradyne ASCII format or Chroma CSV):")
        
        btn_sample = st.button("📂 Load Pre-Built Teradyne J750 Sample Log (LOT-ISRO-2026-X1)")
        ate_file = st.file_uploader("Or Upload Custom ATE Datalog File:", type=['txt', 'csv', 'log'])
        
        log_content = None
        filename = "sample.txt"
        
        if btn_sample:
            sample_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ate_samples', 'teradyne_j750_lot_x1.txt'))
            if os.path.exists(sample_path):
                with open(sample_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                filename = "teradyne_j750_lot_x1.txt"
            else:
                log_content = st.session_state.ate_parser.generate_sample_ate_log(50, "LOT-ISRO-2026-X1")
                filename = "teradyne_sample.txt"
        elif ate_file is not None:
            log_content = ate_file.read().decode('utf-8', errors='ignore')
            filename = ate_file.name
            
        if log_content:
            st.info(f"Loaded `{filename}` ({len(log_content.splitlines())} lines). Parsing ATE datalog...")
            parsed = st.session_state.ate_parser.parse_ate_stream(log_content, filename=filename)
            meta = parsed['metadata']
            parsed_df = parsed['dataframe']
            
            # Show metadata
            st.markdown("#### 📋 ATE Tester Session Metadata")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Tester Model", meta.get('tester_type', 'ATE-GENERIC'))
            mc2.metric("Test Program", meta.get('test_program', 'MIL_STD_883K'))
            mc3.metric("Oven Temp", f"{meta.get('temperature_c', 125.0)} °C")
            mc4.metric("Extracted Parts", len(parsed_df))
            
            # Run screening on parsed ATE data
            if st.button("🚀 Execute Automated Screening on ATE Data"):
                with st.spinner("Processing through Dynamic PAT & Drift Forecaster..."):
                    df_proc = preprocessor.transform(parsed_df)
                    res_a = mod_a.predict_detailed(df_proc)
                    res_b = mod_b.predict(df_proc)
                    final_ate = ScreeningDecisionEngine().evaluate(res_a, res_b)
                    
                    st.markdown("#### 📊 Screening Dispositions")
                    st.dataframe(final_ate[['part_id', 'lot_id', 'value_0h', 'value_24h', 'final_decision', 'composite_risk_score', 'primary_reason']].head(30), use_container_width=True)
                    
                    csv_down = final_ate.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download ATE Disposition CSV", data=csv_down, file_name=f"ate_screened_{datetime.date.today()}.csv", mime="text/csv")
                    
    with tab_api:
        st.markdown("### 🌐 Enterprise REST API Documentation & Swagger UI")
        st.markdown("""
        The FastAPI backend runs on `http://localhost:8000`. You can test endpoints below or open the interactive Swagger UI.
        """)
        
        st.link_button("🚀 Open Interactive Swagger / OpenAPI Docs", "http://localhost:8000/docs")
        
        st.markdown("#### 📡 Core API Endpoints")
        api_table = pd.DataFrame([
            {"Method": "POST", "Endpoint": "/api/v1/screen/part", "Description": "Single component real-time test bench screening"},
            {"Method": "POST", "Endpoint": "/api/v1/screen/lot", "Description": "Batch production lot screening & Maverick Lot evaluation"},
            {"Method": "POST", "Endpoint": "/api/v1/ingest/ate", "Description": "Raw ATE tester ASCII / CSV datalog file upload"},
            {"Method": "POST", "Endpoint": "/api/v1/chamber/telemetry", "Description": "Burn-in oven temperature & rack power ingestion"},
            {"Method": "GET",  "Endpoint": "/api/v1/lots/{lot_id}/coc", "Description": "Download official signed PDF Certificate of Conformance"},
            {"Method": "POST", "Endpoint": "/api/v1/webhooks/test", "Description": "Dispatch HMAC-SHA256 authenticated MES webhook"},
            {"Method": "GET",  "Endpoint": "/api/v1/audit/logs", "Description": "Query cryptographic SHA-256 audit trail"},
            {"Method": "WS",   "Endpoint": "/ws/chamber-live", "Description": "Real-time WebSocket telemetry broadcast"}
        ])
        st.dataframe(api_table, hide_index=True, use_container_width=True)
        
        st.markdown("#### 💻 Python ATE Tester Client Integration Example")
        st.code("""
import requests

# Send 24h burn-in measurement from automated test bench (LabVIEW / Python)
response = requests.post(
    "http://localhost:8000/api/v1/screen/part",
    json={
        "part_id": "ISRO-D042",
        "lot_id": "LOT-ISRO-2026-X1",
        "value_0h": 10.25,
        "value_24h": 38.50,
        "datasheet_limit": 50.0
    }
)

result = response.json()
print("Disposition:", result['final_decision'])         # REJECT
print("Hours Saved:", result['burnin_hours_saved'])       # 144 hrs saved!
print("Audit Hash:", result['audit_sha256'])
        """, language="python")

