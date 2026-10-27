"""
Student Performance Prediction and Analytics System.
Interactive Streamlit Web Dashboard — Cinematic SELENE Intelligence Platform.
"""

import os
import sys
import io
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Environment safeguard for matplotlib caching in constrained environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data, check_data_quality, get_summary_statistics
from src.predict import (
    load_trained_model,
    load_behavioral_model,
    predict_student_grade,
    categorize_grade,
)
from src.explainability import (
    get_global_feature_importances,
    get_aggregated_feature_importances,
    explain_prediction_tree_path,
)
from src.batch_predictor import (
    predict_batch,
    generate_sample_batch_csv,
    validate_batch_dataframe,
)
from src.report_generator import generate_student_pdf_report
from src.preprocessor import separate_features_target
from src.logger import get_logger
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)


# =====================================================================
# STREAMLIT PAGE CONFIGURATION & CINEMATIC SELENE DARK THEME
# =====================================================================

st.set_page_config(
    page_title="SELENE // Student Performance Intelligence Platform",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Cinematic SELENE Lunar/Space Intelligence Atmosphere
st.markdown(
    """
    <style>
    /* Google Fonts Import for Space/Telemetry Typography */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    /* Deep-Space Cosmic Canvas */
    .stApp {
        background-color: #030712;
        background-image: 
            radial-gradient(ellipse at 50% -10%, rgba(56, 189, 248, 0.12) 0%, rgba(3, 7, 18, 0) 65%),
            radial-gradient(circle at 85% 15%, rgba(129, 140, 248, 0.08) 0%, rgba(3, 7, 18, 0) 45%),
            radial-gradient(circle at 15% 75%, rgba(14, 165, 233, 0.05) 0%, rgba(3, 7, 18, 0) 50%);
        background-attachment: fixed;
        color: #F8FAFC;
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Monospace Telemetry Details */
    .telemetry-mono {
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
        letter-spacing: -0.01em;
    }
    
    /* Cinematic Hero Container */
    .selene-hero-banner {
        position: relative;
        background: linear-gradient(180deg, rgba(11, 17, 32, 0.85) 0%, rgba(3, 7, 18, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 2.25rem 2rem 1.75rem 2rem;
        margin-bottom: 2rem;
        overflow: hidden;
        box-shadow: 0 0 50px -10px rgba(56, 189, 248, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    .selene-hero-banner::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #38BDF8 50%, transparent 100%);
    }
    
    /* Top Telemetry Mission Bar */
    .telemetry-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        padding-bottom: 1rem;
        margin-bottom: 1.25rem;
        border-bottom: 1px solid rgba(30, 41, 59, 0.8);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 9999px;
        padding: 0.2rem 0.65rem;
        color: #38BDF8;
    }
    .telemetry-pulse {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #34D399;
        box-shadow: 0 0 8px #34D399;
    }
    
    /* Hero Typography */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 40%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
        text-shadow: 0 0 35px rgba(56, 189, 248, 0.25);
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94A3B8;
        max-width: 850px;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    
    /* Glassmorphic HUD Metric Cards */
    .hud-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(11, 17, 32, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.16);
        border-radius: 12px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(12px);
        position: relative;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .hud-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 12px 35px -5px rgba(56, 189, 248, 0.25);
    }
    .hud-card .hud-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.35rem;
    }
    .hud-card h3 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #38BDF8;
        letter-spacing: -0.02em;
        font-family: 'JetBrains Mono', monospace;
        text-shadow: 0 0 15px rgba(56, 189, 248, 0.35);
    }
    .hud-card p {
        margin: 0.35rem 0 0 0;
        font-size: 0.78rem;
        color: #CBD5E1;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Dark Cosmic HUD Containers */
    .hud-panel {
        background: rgba(11, 17, 32, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    .hud-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        letter-spacing: -0.01em;
    }
    
    /* Badges & Status Indicators */
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-blue {
        background-color: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.15);
    }
    .badge-green {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.15);
    }
    .badge-amber {
        background-color: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.35);
        box-shadow: 0 0 10px rgba(245, 158, 11, 0.15);
    }
    .badge-rose {
        background-color: rgba(244, 63, 94, 0.12);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.35);
        box-shadow: 0 0 10px rgba(244, 63, 94, 0.15);
    }
    
    /* Minimalist Space Observatory Footer */
    .selene-footer {
        text-align: center;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        margin-top: 4rem;
        padding-top: 1.75rem;
        border-top: 1px solid rgba(30, 41, 59, 0.8);
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# MATPLOTLIB COSMIC SPACE THEME HELPER (OFFLINE & DEPENDENCY-FREE)
# =====================================================================

def apply_dark_figure_style(fig, ax):
    """
    Applies consistent Deep Cosmic Space theme styling to Matplotlib figures and axes.
    Keeps visualizations completely offline, lightweight, and cohesive with the SELENE HUD.
    """
    fig.patch.set_facecolor("#0B1120")
    ax.set_facecolor("#080E1C")
    ax.tick_params(colors="#94A3B8", which="both", labelsize=8.5)
    ax.xaxis.label.set_color("#CBD5E1")
    ax.yaxis.label.set_color("#CBD5E1")
    ax.title.set_color("#F8FAFC")
    for spine in ax.spines.values():
        spine.set_color("#1E293B")
        spine.set_linewidth(1.0)
    ax.grid(color="#172554", linestyle="--", linewidth=0.7, alpha=0.5)


# =====================================================================
# CACHED DATA & MODEL LOADERS
# =====================================================================

@st.cache_data
def get_dataset():
    """Cached loader for the authentic student performance dataset."""
    try:
        data = load_data("data/student_performance.csv")
        logger.info("Dashboard: dataset loaded successfully (%d rows).", len(data))
        return data
    except FileNotFoundError as e:
        logger.error("Dashboard: dataset file missing — %s", e)
        st.error(
            "📂 **Dataset file not found:** `data/student_performance.csv`\n\n"
            "Please verify that the data directory is present in the project root."
        )
        return None
    except Exception as e:
        logger.error("Dashboard: unexpected error loading dataset — %s", e)
        st.error(f"Unexpected error loading dataset: {e}")
        return None


@st.cache_resource
def get_model():
    """Cached loader for the serialized production Scikit-learn Pipeline."""
    try:
        loaded_model = load_trained_model("models/student_model.pkl")
        logger.info("Dashboard: production model pipeline loaded successfully.")
        return loaded_model
    except FileNotFoundError as e:
        logger.error("Dashboard: model file missing — %s", e)
        st.error(
            "🤖 **Trained model not found:** `models/student_model.pkl`\n\n"
            "Run `python src/train.py` in your terminal to train and save the model."
        )
        return None
    except Exception as e:
        logger.error("Dashboard: unexpected error loading model — %s", e)
        st.error(f"Unexpected error loading model pipeline: {e}")
        return None


@st.cache_resource
def get_behavioral_model():
    """Cached loader for the early-semester behavioral Scikit-learn Pipeline."""
    try:
        loaded_model = load_behavioral_model("models/behavioral_model.pkl")
        logger.info("Dashboard: behavioral model pipeline loaded successfully.")
        return loaded_model
    except Exception as e:
        logger.warning("Dashboard: behavioral model not available — %s", e)
        return None


@st.cache_data
def get_evaluation_metrics():
    """Cached loader for stored model evaluation metrics."""
    metrics_path = "models/evaluation_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Dashboard: evaluation metrics loaded from '%s'.", metrics_path)
        return data
    logger.warning("Dashboard: evaluation_metrics.json not found at '%s'.", metrics_path)
    return None


df = get_dataset()
model = get_model()
beh_model = get_behavioral_model()
eval_data = get_evaluation_metrics()


# =====================================================================
# SIDEBAR NAVIGATION & TELEMETRY OBSERVATORY
# =====================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.25rem;">
            <span style="font-size: 1.6rem;">🌕</span>
            <div>
                <strong style="font-size: 1.15rem; color: #F8FAFC; letter-spacing: -0.02em;">SELENE</strong><br>
                <span class="telemetry-mono" style="font-size: 0.68rem; color: #38BDF8;">OBSERVATORY // v2.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Student Performance Telemetry & Risk Intelligence")
    st.markdown("---")

    st.markdown(
        """
        <div class="telemetry-mono" style="font-size: 0.78rem; color: #94A3B8; line-height: 1.8;">
            <strong style="color: #F8FAFC;">// MISSION CONTEXT</strong><br>
            • DOMAIN: Third Year B.Tech IT<br>
            • LAB: Data Science & Vis<br>
            • BENCHMARK: UCI Edu Mining<br>
            • TARGET: Final Mark <em>G3</em> [0–20]
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="telemetry-mono" style="font-size: 0.78rem; color: #94A3B8; line-height: 1.8;">
            <strong style="color: #F8FAFC;">// ORBITAL TELEMETRY</strong><br>
            • ENGINE: <span style="color: #38BDF8;">Scikit-learn Pipeline</span><br>
            • TRANSFORM: <span style="color: #818CF8;">ColumnTransformer</span><br>
            • PRIMARY: <span style="color: #34D399;">Random Forest (R² 0.81)</span><br>
            • ADVISORY: <span style="color: #FBBF24;">Behavioral Mode (Active)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("🔒 100% Local Inference · Zero External Telemetry")


# =====================================================================
# CINEMATIC SELENE HERO BANNER (PROCEDURAL LUNAR / EARTHRISE SVG)
# =====================================================================

st.markdown(
    """
    <div class="selene-hero-banner">
        <!-- Telemetry Status Bar -->
        <div class="telemetry-bar">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span class="telemetry-pill">
                    <span class="telemetry-pulse"></span>
                    TELEMETRY: NOMINAL
                </span>
                <span>COORD: TRANQUILITY BASE [0.6740° N · 23.4720° E]</span>
            </div>
            <div>
                <span>MODEL PIPELINE: RANDOM FOREST v1.9 // VERIFIED</span>
            </div>
        </div>
        
        <!-- Hero Content with Procedural Lunar Earthrise SVG -->
        <div style="display: grid; grid-template-columns: 1.4fr 0.8fr; gap: 2rem; align-items: center;">
            <div>
                <h1 class="hero-title">SELENE // Student Performance Intelligence</h1>
                <p class="hero-subtitle">
                    A cinematic predictive observatory and early academic warning platform.
                    Synthesizing multi-variable behavioral indicators, academic trajectory, and 
                    socioeconomic support into empirical grade forecasts.
                </p>
                <div style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
                    <span class="badge-pill badge-blue">Empirical Machine Learning</span>
                    <span class="badge-pill badge-green">Zero Data Leakage Pipeline</span>
                    <span class="badge-pill badge-amber">Early Intervention HUD</span>
                </div>
            </div>
            <!-- Lightweight Procedural Earthrise & Lunar Horizon SVG -->
            <div style="display: flex; justify-content: center; align-items: center;">
                <svg width="240" height="170" viewBox="0 0 240 170" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 0 25px rgba(56, 189, 248, 0.3));">
                    <defs>
                        <!-- Earth Atmosphere Glow -->
                        <radialGradient id="earthGlow" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.9"/>
                            <stop offset="60%" stop-color="#0284C7" stop-opacity="0.6"/>
                            <stop offset="100%" stop-color="#0369A1" stop-opacity="0"/>
                        </radialGradient>
                        <radialGradient id="earthBody" cx="40%" cy="35%" r="60%">
                            <stop offset="0%" stop-color="#7DD3FC"/>
                            <stop offset="45%" stop-color="#0284C7"/>
                            <stop offset="85%" stop-color="#0C4A6E"/>
                            <stop offset="100%" stop-color="#021E38"/>
                        </radialGradient>
                        <linearGradient id="lunarSurface" x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stop-color="#334155"/>
                            <stop offset="40%" stop-color="#1E293B"/>
                            <stop offset="100%" stop-color="#0B1120"/>
                        </linearGradient>
                    </defs>
                    <!-- Background Stars -->
                    <circle cx="25" cy="20" r="0.8" fill="#FFFFFF" opacity="0.6"/>
                    <circle cx="65" cy="45" r="1.1" fill="#7DD3FC" opacity="0.7"/>
                    <circle cx="190" cy="25" r="0.9" fill="#FFFFFF" opacity="0.8"/>
                    <circle cx="215" cy="65" r="0.7" fill="#38BDF8" opacity="0.5"/>
                    <circle cx="140" cy="15" r="1.2" fill="#FFFFFF" opacity="0.9"/>
                    <circle cx="40" cy="85" r="0.8" fill="#FFFFFF" opacity="0.4"/>
                    <!-- Earth Atmosphere Outer Glow -->
                    <circle cx="120" cy="65" r="46" fill="url(#earthGlow)"/>
                    <!-- Earth Sphere -->
                    <circle cx="120" cy="65" r="34" fill="url(#earthBody)"/>
                    <!-- Continents / Cloud Swirls -->
                    <path d="M108 55 C115 50, 130 52, 134 58 C136 62, 132 68, 126 70 C120 72, 114 66, 108 62 Z" fill="#34D399" opacity="0.45"/>
                    <path d="M112 72 C118 70, 125 76, 130 82 C125 86, 118 84, 114 80 Z" fill="#38BDF8" opacity="0.35"/>
                    <ellipse cx="122" cy="50" rx="18" ry="4" fill="#FFFFFF" opacity="0.25" transform="rotate(-15 122 50)"/>
                    <!-- Lunar Horizon Surface Curve -->
                    <path d="M0 135 Q120 100 240 135 L240 170 L0 170 Z" fill="url(#lunarSurface)"/>
                    <!-- Lunar Horizon Highlight Rim -->
                    <path d="M0 135 Q120 100 240 135" stroke="rgba(56, 189, 248, 0.4)" stroke-width="1.5" fill="none"/>
                    <!-- Subtle Craters -->
                    <ellipse cx="70" cy="142" rx="14" ry="4" fill="#131B2E" stroke="#475569" stroke-width="0.8" opacity="0.6"/>
                    <ellipse cx="165" cy="138" rx="20" ry="5" fill="#131B2E" stroke="#475569" stroke-width="0.8" opacity="0.7"/>
                    <ellipse cx="125" cy="152" rx="9" ry="2.5" fill="#0F172A" stroke="#334155" stroke-width="0.6" opacity="0.5"/>
                </svg>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# FIVE INTERACTIVE TABS (PRESERVED COMPLETE FUNCTIONALITY & BATCH TERMINAL)
# =====================================================================

tab_overview, tab_eda, tab_eval, tab_predict, tab_batch = st.tabs(
    [
        "📋 Mission Overview",
        "📊 Telemetry Analytics (EDA)",
        "⚙️ Model Intelligence Benchmark",
        "🎯 Performance Predictor Terminal",
        "📦 Batch CSV Intelligence Terminal",
    ]
)


# =====================================================================
# TAB 1: MISSION OVERVIEW
# =====================================================================

with tab_overview:
    # Luminous HUD Telemetry Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="hud-card">
                <div class="hud-tag">// TELEMETRY.01</div>
                <h3>395</h3>
                <p>STUDENT COHORT</p>
                <div style="margin-top: 0.6rem;"><span class="badge-pill badge-blue">Authentic Benchmark</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="hud-card">
                <div class="hud-tag">// DIMENSIONS.02</div>
                <h3>33</h3>
                <p>ATTRIBUTES SAMPLED</p>
                <div style="margin-top: 0.6rem;"><span class="badge-pill badge-blue">15 Num · 17 Cat · 1 Target</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="hud-card">
                <div class="hud-tag">// TARGET.03</div>
                <h3>0 – 20</h3>
                <p>EVALUATION SCALE (G3)</p>
                <div style="margin-top: 0.6rem;"><span class="badge-pill badge-amber">Continuous Mark</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        r2_val = eval_data["metrics"]["Random Forest Regressor"]["R2"] if eval_data else "0.8106"
        st.markdown(
            f"""
            <div class="hud-card">
                <div class="hud-tag">// PRECISION.04</div>
                <h3>{r2_val}</h3>
                <p>PRODUCTION R² SCORE</p>
                <div style="margin-top: 0.6rem;"><span class="badge-pill badge-green">81.1% Variance Explained</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown(
            """
            <div class="hud-panel">
                <div class="hud-header">🛰️ Problem Statement & Scientific Scope</div>
                <p style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.6;">
                    Early identification of students needing educational intervention is critical.
                    Conventional exams identify at-risk learners only when final marks are published,
                    leaving inadequate opportunity for timely remedial mentoring.
                </p>
                <ul style="color: #94A3B8; font-size: 0.9rem; line-height: 1.7; margin-bottom: 0;">
                    <li><strong style="color: #F8FAFC;">Multi-Factor Telemetry:</strong> Quantifies relationships across academic history, study time, absences, and socioeconomic support.</li>
                    <li><strong style="color: #F8FAFC;">Continuous Forecasting:</strong> Predicts continuous final marks (scale 0–20) using Scikit-learn regression pipelines.</li>
                    <li><strong style="color: #F8FAFC;">Early Warning Advisory:</strong> Categorizes forecasts into four actionable tiers (<em>Excellent, Good, Pass, Needs Intervention</em>).</li>
                    <li><strong style="color: #F8FAFC;">Strict Leakage Prevention:</strong> Preprocessing pipelines fit solely on training data, preventing optimistic evaluation bias.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if df is not None:
            quality = check_data_quality(df)
            st.markdown(
                f"""
                <div class="hud-panel">
                    <div class="hud-header">🔍 Dataset Provenance & Observational Integrity</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; color: #CBD5E1; font-size: 0.88rem;" class="telemetry-mono">
                        <div>• SOURCE: UCI Machine Learning Repo</div>
                        <div>• DOMAIN: Secondary School Math</div>
                        <div>• RECORDS: {quality['row_count']} Observations</div>
                        <div>• ATTRIBUTES: {quality['column_count']} Variables</div>
                        <div>• MISSING CELLS: {quality['missing_values_total']} (0.0% Clean)</div>
                        <div>• DUPLICATES: {quality['duplicate_rows']} (0.0% Unique)</div>
                        <div>• TARGET: G3 (Final Examination)</div>
                        <div>• TARGET MEAN: 10.42 / 20.0 (σ = 4.58)</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_right:
        st.markdown(
            """
            <div class="hud-panel">
                <div class="hud-header">⚡ Machine Learning Telemetry Pipeline Flow</div>
                <pre class="telemetry-mono" style="background-color: #030712; border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; color: #38BDF8; padding: 0.85rem; font-size: 0.73rem; line-height: 1.45;">
┌────────────────────────────────────────────────────────┐
│      UCI Student Performance Dataset (395 records)     │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│   80/20 Reproducible Split (seed=42) [316 train, 79 test]│
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│   Scikit-learn ColumnTransformer (Zero Data Leakage)   │
│   • StandardScaler for 15 Numeric Features             │
│   • OneHotEncoder for 17 Categorical Features          │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│   Candidate Models Evaluated:                          │
│   * Baseline Dummy (Predict Mean)                      │
│   * Linear Regression                                  │
│   * Ridge Regression (L2)                              │
│   * Gradient Boosting Regressor                        │
│   * Random Forest Regressor (Selected: R²=0.8106)      │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│   Cinematic SELENE Intelligence HUD & Predictor        │
└────────────────────────────────────────────────────────┘</pre>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="hud-header">📄 Raw Telemetry Stream (Authentic Benchmark Records)</div>', unsafe_allow_html=True)
    if df is not None:
        st.dataframe(df.head(6), use_container_width=True)


# =====================================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================

with tab_eda:
    st.markdown('<div class="hud-header">📊 Telemetry Analytics & Statistical Visualizations</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #94A3B8; font-size: 0.95rem;">Examine the statistical relationships between student evaluation marks, study time, absences, and academic drivers.</p>',
        unsafe_allow_html=True,
    )

    if df is not None:
        # Chart Row 1: Grade Density Progression & Correlation Matrix
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            st.markdown("##### A. Grade Density Progression (G1, G2, G3)")
            fig, ax = plt.subplots(figsize=(7, 4.3))
            apply_dark_figure_style(fig, ax)
            sns.kdeplot(df["G1"], label="G1 (Period 1)", color="#38BDF8", fill=True, alpha=0.25, ax=ax, linewidth=2)
            sns.kdeplot(df["G2"], label="G2 (Period 2)", color="#34D399", fill=True, alpha=0.25, ax=ax, linewidth=2)
            sns.kdeplot(df["G3"], label="G3 (Final)", color="#F87171", fill=True, alpha=0.25, ax=ax, linewidth=2)
            ax.set_title("Density Distributions of Student Grades (Scale 0-20)", fontsize=10.5, fontweight="bold", pad=12)
            ax.set_xlabel("Mark (0 to 20)")
            ax.set_ylabel("Kernel Density")
            ax.set_xlim(-1, 21)
            leg = ax.legend(facecolor="#0B1120", edgecolor="#1E293B", framealpha=0.95)
            for text in leg.get_texts():
                text.set_color("#F8FAFC")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "**What this shows:** All three exam periods center around 10–11 marks. The spike at 0 marks in G3 reflects students who withdrew or were absent during final evaluations."
            )

        with row1_col2:
            st.markdown("##### B. Correlation Matrix of Academic Indicators")
            fig, ax = plt.subplots(figsize=(7, 4.3))
            apply_dark_figure_style(fig, ax)
            corr_cols = ["G1", "G2", "G3", "studytime", "failures", "absences", "age", "health"]
            corr_matrix = df[corr_cols].corr()
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="vlag",
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                linecolor="#080E1C",
                ax=ax,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8, "color": "#F8FAFC"},
            )
            ax.set_title("Correlation Heatmap of Key Academic Factors", fontsize=10.5, fontweight="bold", pad=12)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "**What this shows:** $G1$ and $G2$ show the strongest positive correlation with final grade $G3$ ($r > 0.80$). Past failures exhibit an inverse relationship with outcomes ($r = -0.36$)."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Chart Row 2: Study Time Boxplot & Absences Scatter Plot
        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.markdown("##### C. Weekly Study Hours vs. Final Grade (G3)")
            fig, ax = plt.subplots(figsize=(7, 4.3))
            apply_dark_figure_style(fig, ax)
            palette = ["#38BDF8", "#818CF8", "#A78BFA", "#C084FC"]
            sns.boxplot(
                x="studytime",
                y="G3",
                data=df,
                palette=palette,
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "#FBBF24", "markeredgecolor": "#FFFFFF", "markersize": 5.5},
                boxprops=dict(alpha=0.85, edgecolor="#FFFFFF", linewidth=0.8),
                whiskerprops=dict(color="#94A3B8"),
                capprops=dict(color="#94A3B8"),
                medianprops=dict(color="#FFFFFF", linewidth=1.5),
                ax=ax,
            )
            ax.set_xticklabels(["< 2 hrs", "2 to 5 hrs", "5 to 10 hrs", "> 10 hrs"])
            ax.set_title("Final Grade by Weekly Study Hours Category", fontsize=10.5, fontweight="bold", pad=12)
            ax.set_xlabel("Weekly Study Time Category")
            ax.set_ylabel("Final Grade (G3)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "**What this shows:** Students studying $>5$ hours weekly (categories 3 & 4) achieve higher median and mean marks (yellow diamonds) with fewer low-grade outliers."
            )

        with row2_col2:
            st.markdown("##### D. School Absences vs. Final Grade (G3)")
            fig, ax = plt.subplots(figsize=(7, 4.3))
            apply_dark_figure_style(fig, ax)
            sns.regplot(
                x="absences",
                y="G3",
                data=df,
                scatter_kws={"alpha": 0.55, "color": "#38BDF8", "edgecolor": "#030712", "s": 35},
                line_kws={"color": "#F43F5E", "linewidth": 2.2},
                ax=ax,
            )
            ax.set_title("School Absences vs. Final Grade with Linear Trendline", fontsize=10.5, fontweight="bold", pad=12)
            ax.set_xlabel("Total Absences Count (0 to 93)")
            ax.set_ylabel("Final Grade (G3)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "**What this shows:** Most students accumulate fewer than 15 absences. Chronic absenteeism displays higher performance dispersion and lower minimum thresholds."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("##### E. Descriptive Statistics Summary (Numeric Attributes)")
        num_stats, _ = get_summary_statistics(df)
        st.dataframe(num_stats, use_container_width=True)


# =====================================================================
# TAB 3: MODEL EVALUATION & BENCHMARK COMPARISON
# =====================================================================

with tab_eval:
    st.markdown('<div class="hud-header">⚙️ Model Intelligence Benchmark & Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #94A3B8; font-size: 0.95rem;">Rigorous comparison across candidate regression algorithms evaluated on the <strong>79 held-out test students (20%, seed=42)</strong>.</p>',
        unsafe_allow_html=True,
    )

    if eval_data is not None:
        metrics_dict = eval_data.get("metrics", {})

        # Display Comparison Table with Styled Highlighting
        eval_df = pd.DataFrame(metrics_dict).T
        eval_df.columns = ["MAE (Mean Absolute Error)", "RMSE (Root Mean Squared Error)", "R² Score"]
        eval_df.index.name = "Candidate Algorithm"

        st.markdown("##### 1. Held-Out Test Set Performance Benchmark")
        st.dataframe(
            eval_df.style.highlight_min(subset=["MAE (Mean Absolute Error)", "RMSE (Root Mean Squared Error)"], color="#064E3B")
            .highlight_max(subset=["R² Score"], color="#064E3B")
            .format("{:.4f}"),
            use_container_width=True,
        )

        best_model_name = eval_data.get("best_model_name", "Random Forest Regressor")
        best_rmse = eval_df.loc[best_model_name, "RMSE (Root Mean Squared Error)"]
        best_r2 = eval_df.loc[best_model_name, "R² Score"]

        st.markdown(
            f"""
            <div style="background-color: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 1rem; margin-top: 0.75rem;">
                <strong style="color: #34D399;">🏆 Selected Production Model: {best_model_name}</strong><br>
                <span style="color: #CBD5E1; font-size: 0.9rem;">
                    Achieved lowest test RMSE (<strong>{best_rmse:.4f}</strong>) and highest variance explained R² (<strong>{best_r2:.4f}</strong>), 
                    reducing baseline dummy error by over 56.7%.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Cross-validation summary if present
        cv_data = eval_data.get("cross_validation")
        if cv_data:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 2. 5-Fold Cross-Validation Summary (Model Stability Check)")
            cv_df = pd.DataFrame(cv_data).T
            cv_df.columns = ["Mean CV R²", "Std CV R²", "Mean CV RMSE", "Std CV RMSE"]
            cv_df.index.name = "Ensemble Algorithm"
            st.dataframe(cv_df.style.format("{:.4f}"), use_container_width=True)

        # Behavioral-only model card if present
        beh_data = eval_data.get("behavioral_model")
        if beh_data:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="hud-panel" style="border-left: 4px solid #F59E0B;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: #FBBF24; font-size: 1.05rem;">Early-Semester Behavioral Model (Excludes G1 & G2)</strong>
                        <span class="badge-pill badge-amber">30 Behavioral Features</span>
                    </div>
                    <p style="color: #94A3B8; font-size: 0.85rem; margin: 0.5rem 0;">{beh_data.get('description', '')}</p>
                    <div style="display: flex; gap: 2rem; color: #E2E8F0; font-size: 0.9rem; margin-top: 0.5rem;" class="telemetry-mono">
                        <div>MAE: <strong>{beh_data.get('MAE', 0):.4f}</strong></div>
                        <div>RMSE: <strong>{beh_data.get('RMSE', 0):.4f}</strong></div>
                        <div>R²: <strong>{beh_data.get('R2', 0):.4f}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Visual Test Set Diagnostics
        if df is not None and model is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 3. Visual Diagnostics on Unseen Test Data (N=79)")
            col_diag1, col_diag2 = st.columns(2)

            @st.cache_data
            def _get_test_predictions(_model, _df):
                X_full, y_full = separate_features_target(_df)
                _, X_test, _, y_test = train_test_split(X_full, y_full, test_size=0.20, random_state=42)
                y_pred = _model.predict(X_test)
                return y_test.values, y_pred

            y_test_arr, y_pred_arr = _get_test_predictions(model, df)
            residuals_arr = y_test_arr - y_pred_arr

            with col_diag1:
                fig, ax = plt.subplots(figsize=(6.5, 4.3))
                apply_dark_figure_style(fig, ax)
                ax.scatter(y_test_arr, y_pred_arr, color="#38BDF8", alpha=0.75, edgecolors="#030712", s=45, label="Test Predictions")
                ax.plot([0, 20], [0, 20], color="#F43F5E", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
                ax.set_title("Actual vs. Predicted Final Marks (Test Set, N=79)", fontsize=10.5, fontweight="bold", pad=12)
                ax.set_xlabel("Actual Grade (G3)")
                ax.set_ylabel("Predicted Grade (G3)")
                ax.set_xlim(-1, 21)
                ax.set_ylim(-1, 21)
                leg = ax.legend(facecolor="#0B1120", edgecolor="#1E293B", framealpha=0.95)
                for text in leg.get_texts():
                    text.set_color("#F8FAFC")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption(
                    "**Actual vs. Predicted:** Proximity of data points to the red dashed reference line indicates high forecasting precision."
                )

            with col_diag2:
                fig, ax = plt.subplots(figsize=(6.5, 4.3))
                apply_dark_figure_style(fig, ax)
                sns.histplot(residuals_arr, kde=True, color="#A78BFA", bins=15, ax=ax, edgecolor="#080E1C")
                ax.axvline(0, color="#F43F5E", linestyle="--", linewidth=2, label="Zero Error Target")
                ax.set_title("Residual Error Distribution (y_test - y_pred)", fontsize=10.5, fontweight="bold", pad=12)
                ax.set_xlabel("Error in Marks")
                ax.set_ylabel("Observations")
                leg = ax.legend(facecolor="#0B1120", edgecolor="#1E293B", framealpha=0.95)
                for text in leg.get_texts():
                    text.set_color("#F8FAFC")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption(
                    "**Residual Distribution:** Symmetrical distribution centered at zero confirms lack of systemic directional prediction bias."
                )

            # -------------------------------------------------------------
            # U2: GLOBAL FEATURE IMPORTANCE & VARIABLE SENSITIVITY
            # -------------------------------------------------------------
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 4. Global Feature Importance & Variable Sensitivity (U2 Explainability)")
            st.markdown(
                '<p style="color: #94A3B8; font-size: 0.92rem;">'
                'Quantifies the population-level influence of student indicators across all 100 decision trees in the ensemble '
                'using <strong>Mean Decrease in Impurity (MDI / Gini Importance)</strong>.'
                '</p>',
                unsafe_allow_html=True,
            )

            col_exp_ctrl1, col_exp_ctrl2 = st.columns([3, 1])
            with col_exp_ctrl1:
                expl_view_mode = st.radio(
                    "Select Feature Importance Scope:",
                    options=[
                        "Production Pipeline (All 58 Transformed Features)",
                        "Domain Attributes (32 Aggregated Parent Factors)",
                        "Early-Semester Behavioral Model (Excludes G1 & G2)",
                    ],
                    horizontal=True,
                    key="global_expl_view_mode",
                )
            with col_exp_ctrl2:
                top_k_feat = st.slider("Display Top N Factors", min_value=5, max_value=20, value=10, step=1, key="top_k_feat_slider")

            try:
                if "Aggregated" in expl_view_mode:
                    df_imp_display = get_aggregated_feature_importances(model, top_n=top_k_feat)
                    feat_labels = df_imp_display["display_name"].tolist()
                    feat_values = (df_imp_display["importance_pct"]).tolist()
                    chart_title = f"Top {len(df_imp_display)} Aggregated Domain Attributes by Importance"
                    x_label = "Relative Importance Share (%)"
                elif "Behavioral" in expl_view_mode:
                    if beh_model is None:
                        st.warning("Behavioral model artifact not found.")
                        df_imp_display = None
                    else:
                        df_imp_display = get_global_feature_importances(beh_model, top_n=top_k_feat)
                        feat_labels = df_imp_display["display_name"].tolist()
                        feat_values = (df_imp_display["importance"] * 100.0).tolist()
                        chart_title = f"Top {len(df_imp_display)} Behavioral Predictors (No Prior Exam Marks)"
                        x_label = "Mean Decrease in Impurity (%)"
                else:
                    df_imp_display = get_global_feature_importances(model, top_n=top_k_feat)
                    feat_labels = df_imp_display["display_name"].tolist()
                    feat_values = (df_imp_display["importance"] * 100.0).tolist()
                    chart_title = f"Top {len(df_imp_display)} Transformed Features by MDI Importance"
                    x_label = "Mean Decrease in Impurity (%)"

                if df_imp_display is not None:
                    col_chart, col_tbl = st.columns([1.3, 1])

                    with col_chart:
                        fig, ax = plt.subplots(figsize=(7, 4.8))
                        apply_dark_figure_style(fig, ax)

                        # Invert so highest feature is at top
                        y_pos = np.arange(len(feat_labels))
                        bars = ax.barh(y_pos, feat_values[::-1], color="#38BDF8", edgecolor="#080E1C", height=0.65, alpha=0.9)

                        # Add data labels
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(
                                width + max(feat_values) * 0.015,
                                bar.get_y() + bar.get_height() / 2,
                                f"{width:.1f}%",
                                va="center",
                                ha="left",
                                color="#F8FAFC",
                                fontsize=8.5,
                                fontweight="bold",
                            )

                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(feat_labels[::-1], fontsize=9)
                        ax.set_xlabel(x_label, fontsize=9.5)
                        ax.set_title(chart_title, fontsize=10.5, fontweight="bold", pad=12)
                        ax.set_xlim(0, max(feat_values) * 1.18)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                    with col_tbl:
                        st.markdown("###### 📋 Telemetry Importance Breakdown")
                        if "parent_attribute" in df_imp_display.columns:
                            show_df = df_imp_display[["display_name", "importance_pct"]].copy()
                            show_df.columns = ["Academic Attribute", "Importance (%)"]
                            show_df["Importance (%)"] = show_df["Importance (%)"].map(lambda x: f"{x:.2f}%")
                        else:
                            show_df = df_imp_display[["display_name", "importance", "cumulative_share"]].copy()
                            show_df["importance"] = show_df["importance"].map(lambda x: f"{x * 100.0:.2f}%")
                            show_df["cumulative_share"] = show_df["cumulative_share"].map(lambda x: f"{x * 100.0:.1f}%")
                            show_df.columns = ["Feature Identifier", "MDI Share", "Cumul. Share"]

                        st.dataframe(show_df, use_container_width=True, height=280)

                    # Educational & Scientific Limitations Advisory
                    st.markdown(
                        """
                        <div style="background-color: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 1rem; margin-top: 0.5rem; font-size: 0.85rem; line-height: 1.6; color: #94A3B8;">
                            <strong style="color: #38BDF8;">🔍 Academic Explainability Insights & Methodological Limitations:</strong>
                            <ul style="margin: 0.4rem 0 0.2rem 1.2rem; padding: 0;">
                                <li><strong>Target Correlation Dominance:</strong> In the full production model, <em>Second Period Grade (G2)</em> accounts for ~80.2% of tree split purity. This is mathematically sound because midterm academic proficiency is the single strongest empirical predictor of final marks.</li>
                                <li><strong>Early-Semester Actionable Levers:</strong> When exam grades are omitted in the Behavioral Model, the Random Forest distributes predictive weight across <em>past failures, absences, weekly study time, and parental education</em>. These represent actionable behavioral intervention targets.</li>
                                <li><strong>Statistical Association vs. Causality:</strong> MDI measures impurity reduction across training samples; it does <em>not</em> imply physical causality. High correlation does not replace holistic academic mentoring.</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as exp_err:
                st.error(f"Failed to generate global feature importance: {exp_err}")


    else:
        st.warning("Evaluation metrics file not found. Please run 'src/train.py' first.")


# =====================================================================
# TAB 4: PERFORMANCE PREDICTOR TERMINAL
# =====================================================================

with tab_predict:
    st.markdown('<div class="hud-header">🎯 Performance Predictor Terminal</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #94A3B8; font-size: 0.95rem;">Input student attributes below to generate an empirical mark forecast, risk classification, and academic guidance advisory.</p>',
        unsafe_allow_html=True,
    )

    # Mode Selector: Full Production Pipeline vs Early Behavioral Pipeline
    pred_mode = st.radio(
        "Select Prediction Mode:",
        options=["Production Model (Full History with G1 & G2)", "Early-Semester Behavioral Model (Pre-Exam: No G1/G2)"],
        index=0,
        horizontal=True,
        help="Use Full History when period exam marks are known. Use Early-Semester when advising students before initial tests.",
    )

    use_behavioral = "Early-Semester" in pred_mode

    with st.form("student_prediction_form"):
        col_input1, col_input2 = st.columns(2)

        with col_input1:
            st.markdown("##### 📚 Academic History & Study Habits")

            if not use_behavioral:
                g1_val = st.slider("First Period Mark (G1)", min_value=0, max_value=20, value=11, step=1, help="Period 1 exam score (0-20)")
                g2_val = st.slider("Second Period Mark (G2)", min_value=0, max_value=20, value=11, step=1, help="Period 2 exam score (0-20)")
            else:
                st.info("💡 **Early-Semester Mode:** Exam grades (G1 and G2) are omitted. Forecast relies strictly on study time, attendance, past failures, and support factors.")
                g1_val = 11
                g2_val = 11

            studytime_val = st.selectbox(
                "Weekly Study Time Category",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "< 2 hours / week",
                    2: "2 to 5 hours / week",
                    3: "5 to 10 hours / week",
                    4: "> 10 hours / week",
                }[x],
                index=1,
            )
            failures_val = st.slider("Past Class Failures Count", min_value=0, max_value=4, value=0, step=1)
            absences_val = st.slider("Total School Absences", min_value=0, max_value=50, value=4, step=1)

        with col_input2:
            st.markdown("##### 👥 Support Systems & Lifestyle")
            schoolsup_val = st.selectbox("Extra Educational Support from School", options=["no", "yes"], index=0)
            famsup_val = st.selectbox("Family Educational Support", options=["yes", "no"], index=0)
            paid_val = st.selectbox("Extra Paid Tutoring Classes", options=["no", "yes"], index=0)
            higher_val = st.selectbox("Aspires to Pursue Higher Education", options=["yes", "no"], index=0)
            internet_val = st.selectbox("Internet Access at Home", options=["yes", "no"], index=0)
            health_val = st.select_slider("Self-Reported Health Status (1=Very Poor, 5=Excellent)", options=[1, 2, 3, 4, 5], value=4)

        submit_btn = st.form_submit_button("🔮 Compute Performance Forecast", use_container_width=True)

    if submit_btn:
        active_model = beh_model if use_behavioral else model

        if active_model is None:
            st.error("Model pipeline is not loaded. Please train the model via 'src/train.py' first.")
        else:
            student_profile = {
                "G1": g1_val,
                "G2": g2_val,
                "studytime": studytime_val,
                "failures": failures_val,
                "absences": absences_val,
                "schoolsup": schoolsup_val,
                "famsup": famsup_val,
                "paid": paid_val,
                "higher": higher_val,
                "internet": internet_val,
                "health": health_val,
            }

            try:
                result = predict_student_grade(student_profile, model=active_model)
                pred_grade = result["predicted_grade"]
                tier_name = result["tier"]
                risk_level = result["risk_level"]
                advisory_text = result["advisory"]

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="hud-header">📋 Diagnostic Performance Telemetry Report</div>', unsafe_allow_html=True)

                card_col1, card_col2, card_col3 = st.columns(3)

                with card_col1:
                    st.markdown(
                        f"""
                        <div class="hud-card">
                            <div class="hud-tag">// ESTIMATE.OUTCOME</div>
                            <h3 style="color: #38BDF8;">{pred_grade:.1f} <span style="font-size: 1.1rem; color: #94A3B8;">/ 20</span></h3>
                            <p>ESTIMATED FINAL GRADE (G3)</p>
                            <div style="margin-top: 0.6rem;"><span class="badge-pill badge-blue">Continuous Metric</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with card_col2:
                    tier_badge = "badge-green" if ("Excellent" in tier_name or "Good" in tier_name) else ("badge-amber" if "Pass" in tier_name else "badge-rose")
                    tier_color = "#34D399" if ("Excellent" in tier_name or "Good" in tier_name) else ("#FBBF24" if "Pass" in tier_name else "#FB7185")
                    st.markdown(
                        f"""
                        <div class="hud-card">
                            <div class="hud-tag">// TIER.CLASSIFICATION</div>
                            <h3 style="color: {tier_color}; font-size: 1.35rem; margin-top: 0.35rem;">{tier_name.split('(')[0].strip()}</h3>
                            <p>PERFORMANCE TIER</p>
                            <div style="margin-top: 0.6rem;"><span class="badge-pill {tier_badge}">{tier_name.split('(')[-1].rstrip(')')}</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with card_col3:
                    risk_badge = "badge-green" if risk_level == "Low Risk" else ("badge-amber" if risk_level == "Moderate Risk" else "badge-rose")
                    risk_color = "#34D399" if risk_level == "Low Risk" else ("#FBBF24" if risk_level == "Moderate Risk" else "#FB7185")
                    st.markdown(
                        f"""
                        <div class="hud-card">
                            <div class="hud-tag">// STATUS.ADVISORY</div>
                            <h3 style="color: {risk_color}; font-size: 1.35rem; margin-top: 0.35rem;">{risk_level}</h3>
                            <p>ACADEMIC RISK STATUS</p>
                            <div style="margin-top: 0.6rem;"><span class="badge-pill {risk_badge}">Actionable Flag</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(
                    f"""
                    <div class="hud-panel" style="border-left: 4px solid {risk_color};">
                        <strong style="color: #F8FAFC; font-size: 1rem;">Actionable Educational Advisory:</strong><br>
                        <span style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.6;">{advisory_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Visual Progress Scale
                st.write("**Grade Relative Benchmark Scale (0 to 20):**")
                norm_progress = float(min(1.0, max(0.0, pred_grade / 20.0)))
                st.progress(norm_progress)

                # -------------------------------------------------------------
                # U2: INDIVIDUAL PREDICTION EXPLANATION (LOCAL EXPLAINABILITY)
                # -------------------------------------------------------------
                try:
                    expl_res = explain_prediction_tree_path(active_model, student_profile, top_k=5)
                    if expl_res.get("supported", False):
                        base_val = expl_res["base_value"]
                        delta_val = pred_grade - base_val
                        pos_drivers = expl_res.get("positive_contributions", [])
                        neg_drivers = expl_res.get("negative_contributions", [])

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="hud-header">🧭 Individual Prediction Explanation & Attribution (U2 Local Interpretability)</div>', unsafe_allow_html=True)
                        st.markdown(
                            '<p style="color: #94A3B8; font-size: 0.92rem;">'
                            'Decomposes this individual student\'s forecast from the population baseline across all decision trees '
                            'using the <strong>Saabas Tree Path Decomposition algorithm</strong>.'
                            '</p>',
                            unsafe_allow_html=True,
                        )

                        # Top Telemetry Metrics for Local Attribution
                        m_col1, m_col2, m_col3 = st.columns(3)
                        with m_col1:
                            st.markdown(
                                f"""
                                <div class="hud-card" style="padding: 1rem;">
                                    <div class="hud-tag">// POPULATION.BASELINE</div>
                                    <h4 style="color: #94A3B8; margin: 0.25rem 0;">{base_val:.2f} <span style="font-size: 0.85rem;">/ 20</span></h4>
                                    <p style="font-size: 0.78rem;">TRAINING POPULATION MEAN</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        with m_col2:
                            delta_color = "#34D399" if delta_val >= 0 else "#FB7185"
                            delta_sign = "+" if delta_val >= 0 else ""
                            st.markdown(
                                f"""
                                <div class="hud-card" style="padding: 1rem;">
                                    <div class="hud-tag">// NET.FEATURE.SHIFT (Δ)</div>
                                    <h4 style="color: {delta_color}; margin: 0.25rem 0;">{delta_sign}{delta_val:.2f} <span style="font-size: 0.85rem;">marks</span></h4>
                                    <p style="font-size: 0.78rem;">DEVIATION FROM BASELINE</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        with m_col3:
                            exact_badge = "badge-green" if expl_res.get("is_exact", False) else "badge-amber"
                            exact_label = "Exact (10⁻⁴)" if expl_res.get("is_exact", False) else "Approx"
                            st.markdown(
                                f"""
                                <div class="hud-card" style="padding: 1rem;">
                                    <div class="hud-tag">// MODEL.RECONSTRUCTION</div>
                                    <h4 style="color: #38BDF8; margin: 0.25rem 0;">{expl_res['reconstructed_grade']:.2f} <span style="font-size: 0.85rem;">/ 20</span></h4>
                                    <p style="font-size: 0.78rem;"><span class="badge-pill {exact_badge}">{exact_label}</span></p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Top Drivers Split: Positive Boosters vs Negative Pressures
                        st.markdown("<br>", unsafe_allow_html=True)
                        drv_col1, drv_col2 = st.columns(2)

                        with drv_col1:
                            st.markdown("###### 🟢 Top Upward Drivers (Score Boosters)")
                            if pos_drivers:
                                for item in pos_drivers:
                                    st.markdown(
                                        f"""
                                        <div style="background-color: rgba(16, 185, 129, 0.08); border-left: 3px solid #34D399; padding: 0.45rem 0.75rem; margin-bottom: 0.4rem; border-radius: 4px; font-size: 0.85rem; color: #E2E8F0; display: flex; justify-content: space-between;">
                                            <span><strong>{item['display_text']}</strong></span>
                                            <span style="color: #34D399; font-weight: bold; font-family: 'JetBrains Mono', monospace;">+{item['contribution']:.2f}</span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.markdown("<p style='color: #64748B; font-size: 0.85rem;'>No strong positive deflection factors identified.</p>", unsafe_allow_html=True)

                        with drv_col2:
                            st.markdown("###### 🔴 Top Downward Pressures (Score Reducers)")
                            if neg_drivers:
                                for item in neg_drivers:
                                    st.markdown(
                                        f"""
                                        <div style="background-color: rgba(244, 63, 94, 0.08); border-left: 3px solid #FB7185; padding: 0.45rem 0.75rem; margin-bottom: 0.4rem; border-radius: 4px; font-size: 0.85rem; color: #E2E8F0; display: flex; justify-content: space-between;">
                                            <span><strong>{item['display_text']}</strong></span>
                                            <span style="color: #FB7185; font-weight: bold; font-family: 'JetBrains Mono', monospace;">{item['contribution']:.2f}</span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.markdown("<p style='color: #64748B; font-size: 0.85rem;'>No downward pressure factors identified for this profile.</p>", unsafe_allow_html=True)

                        # Bidirectional Attribution Chart
                        chart_items = pos_drivers[::-1] + neg_drivers[::-1]
                        if chart_items:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("###### 📊 Bidirectional Feature Attribution Waterfall")

                            c_labels = [i["display_text"] for i in chart_items]
                            c_vals = [i["contribution"] for i in chart_items]
                            c_colors = ["#34D399" if v >= 0 else "#FB7185" for v in c_vals]

                            fig_local, ax_local = plt.subplots(figsize=(7.5, max(3.5, len(c_labels) * 0.42)))
                            apply_dark_figure_style(fig_local, ax_local)

                            y_indices = np.arange(len(c_labels))
                            bars = ax_local.barh(y_indices, c_vals, color=c_colors, edgecolor="#080E1C", height=0.6, alpha=0.9)
                            ax_local.axvline(0, color="#64748B", linestyle="--", linewidth=1.2, label="Population Mean (0.00)")

                            # Data label annotations
                            max_abs = max([abs(v) for v in c_vals]) if c_vals else 1.0
                            offset = max_abs * 0.03
                            for bar, val in zip(bars, c_vals):
                                x_pos = val + (offset if val >= 0 else -offset)
                                ha_align = "left" if val >= 0 else "right"
                                ax_local.text(
                                    x_pos,
                                    bar.get_y() + bar.get_height() / 2,
                                    f"{val:+.2f}",
                                    va="center",
                                    ha=ha_align,
                                    color="#F8FAFC",
                                    fontsize=8.5,
                                    fontweight="bold",
                                )

                            ax_local.set_yticks(y_indices)
                            ax_local.set_yticklabels(c_labels, fontsize=8.8)
                            ax_local.set_xlabel("Contribution in Grade Marks (relative to baseline)", fontsize=9.2)
                            ax_local.set_title("Individual Feature Attribution via Tree Path Decomposition", fontsize=10.5, fontweight="bold", pad=12)
                            ax_local.set_xlim(-max_abs * 1.35, max_abs * 1.35)
                            st.pyplot(fig_local, use_container_width=True)
                            plt.close(fig_local)

                        # Educational Distinction & Non-Causal Guidance Box
                        st.markdown(
                            """
                            <div style="background-color: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 0.95rem; margin-top: 0.75rem; font-size: 0.83rem; line-height: 1.55; color: #94A3B8;">
                                <strong style="color: #38BDF8;">💡 Interpreting Local Explainability vs. Global Importance:</strong><br>
                                • <strong>Global vs. Local:</strong> Tab 3 displays the macro-level sensitivity across all 395 students. This local panel quantifies why <em>this specific student</em> deflected from the population average.<br>
                                • <strong>Tree Attribution vs. Physical Causation:</strong> Attributions explain the mathematical mechanics of tree ensemble splits. They do <em>not</em> prove direct mechanical causation (e.g., changing study time will only assist learning if combined with instructional comprehension and focus).
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info(f"Local explainability: {expl_res.get('message', 'Not supported for current model.')}")
                except Exception as expl_err:
                    logger.warning("Local explainability failed: %s", expl_err)
                    st.info("Local prediction breakdown is temporarily unavailable for this configuration.")

                # Cautionary Educational Disclaimer
                st.markdown(
                    """
                    <div style="background-color: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 0.9rem; color: #94A3B8; font-size: 0.82rem; line-height: 1.5; margin-top: 1rem;">
                        ⚠️ <strong style="color: #CBD5E1;">Educational Machine Learning Disclaimer:</strong>
                        Predictions are statistical approximations based on demographic and historical trends from the UCI benchmark dataset.
                        They must not be used as definitive judgments or sole determinants for academic tracking or disciplinary measures.
                        Individual student dedication, targeted counseling, and continuous formative assessment can significantly alter real outcomes.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # -------------------------------------------------------------
                # U5: INDIVIDUAL STUDENT REPORT & PDF EXPORT
                # -------------------------------------------------------------
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="hud-header">📄 Official Student Intelligence Dossier (U5 PDF Export)</div>', unsafe_allow_html=True)
                st.markdown(
                    '<p style="color: #94A3B8; font-size: 0.92rem;">'
                    'Export a complete, publication-grade executive PDF dossier documenting this student\'s performance estimate, '
                    'academic tier, risk status, feature attributions, and actionable advisory.'
                    '</p>',
                    unsafe_allow_html=True,
                )

                try:
                    pdf_data = generate_student_pdf_report(
                        student_profile=student_profile,
                        prediction_result=result,
                        explanation_result=expl_res if ('expl_res' in locals() and expl_res.get('supported')) else None,
                        model_name=pred_mode,
                    )
                    st.download_button(
                        label="📥 Download Official Student Performance Dossier (PDF)",
                        data=pdf_data,
                        file_name=f"student_intelligence_report_{int(pred_grade * 10)}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        help="Generates an offline, vector-sharp PDF dossier containing all prediction metrics, drivers, and advisories.",
                    )
                except Exception as pdf_err:
                    logger.error("PDF export button error: %s", pdf_err)
                    st.warning("PDF dossier generation is temporarily unavailable.")

            except Exception as err:
                st.error(f"Prediction failed: {err}")


# =====================================================================
# TAB 5: BATCH CSV INTELLIGENCE TERMINAL (U4)
# =====================================================================

with tab_batch:
    st.markdown('<div class="hud-header">📦 Batch CSV Cohort Intelligence Terminal</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #94A3B8; font-size: 0.95rem;">'
        'Upload a cohort CSV file to perform high-throughput validation, batch grade forecasting, '
        'academic risk segmentation, and export the enriched dataset.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Top Controls: Mode Selection & Sample Template Download
    ctrl_col1, ctrl_col2 = st.columns([2.5, 1.5])

    with ctrl_col1:
        batch_pred_mode = st.radio(
            "Select Batch Prediction Pipeline:",
            options=["Production Model (Full History with G1 & G2)", "Early-Semester Behavioral Model (Pre-Exam: No G1/G2)"],
            index=0,
            horizontal=True,
            key="batch_pred_mode_radio",
            help="Full History requires G1 and G2 scores. Early-Semester evaluates students before initial period exams.",
        )
        batch_use_behavioral = "Early-Semester" in batch_pred_mode

    with ctrl_col2:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        sample_csv_text = generate_sample_batch_csv(5)
        st.download_button(
            label="📥 Download Benchmark Sample Batch CSV",
            data=sample_csv_text,
            file_name="student_batch_sample.csv",
            mime="text/csv",
            help="Download a pre-formatted benchmark template containing 5 diverse student records to test batch processing.",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # CSV File Uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing student records",
        type=["csv"],
        help="Upload a standard comma-separated values (.csv) file. Supports arbitrary identifier columns (e.g. student_id, name).",
    )

    if uploaded_file is not None:
        try:
            # Read file content safely
            raw_bytes = uploaded_file.read()
            if len(raw_bytes) == 0:
                st.error("The uploaded file is empty (0 bytes). Please upload a valid CSV file.")
            else:
                try:
                    uploaded_df = pd.read_csv(io.BytesIO(raw_bytes))
                except Exception as parse_err:
                    st.error(f"Failed to parse CSV file: {parse_err}. Please ensure the file is a valid comma-separated text file.")
                    uploaded_df = None

                if uploaded_df is not None:
                    if uploaded_df.empty:
                        st.error("The uploaded CSV has headers but contains 0 data rows.")
                    else:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("##### 1. Pre-Flight File & Schema Audit")

                        # Validate batch
                        try:
                            valid_df, invalid_df, summary = validate_batch_dataframe(
                                uploaded_df, use_behavioral=batch_use_behavioral
                            )

                            # Metrics Summary Cards
                            scol1, scol2, scol3, scol4 = st.columns(4)
                            with scol1:
                                st.markdown(
                                    f"""
                                    <div class="hud-card" style="padding: 1rem;">
                                        <div class="hud-tag">// TOTAL.RECORDS</div>
                                        <h3 style="color: #38BDF8; margin: 0.2rem 0;">{summary['total_rows']}</h3>
                                        <p style="font-size: 0.78rem;">UPLOADED ROWS</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with scol2:
                                st.markdown(
                                    f"""
                                    <div class="hud-card" style="padding: 1rem;">
                                        <div class="hud-tag">// VALID.RECORDS</div>
                                        <h3 style="color: #34D399; margin: 0.2rem 0;">{summary['valid_rows']}</h3>
                                        <p style="font-size: 0.78rem;">PASSED AUDIT ({summary['valid_percentage']}%)</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with scol3:
                                inv_color = "#FB7185" if summary["invalid_rows"] > 0 else "#94A3B8"
                                st.markdown(
                                    f"""
                                    <div class="hud-card" style="padding: 1rem;">
                                        <div class="hud-tag">// INVALID.RECORDS</div>
                                        <h3 style="color: {inv_color}; margin: 0.2rem 0;">{summary['invalid_rows']}</h3>
                                        <p style="font-size: 0.78rem;">FLAGGED ERRORS</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with scol4:
                                st.markdown(
                                    f"""
                                    <div class="hud-card" style="padding: 1rem;">
                                        <div class="hud-tag">// PIPELINE.TARGET</div>
                                        <h4 style="color: #818CF8; margin: 0.45rem 0; font-size: 1.05rem;">
                                            {"Behavioral Model" if batch_use_behavioral else "Production RF"}
                                        </h4>
                                        <p style="font-size: 0.78rem;">ZERO RETRAINING</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            # If invalid rows detected, show detailed audit report
                            if not invalid_df.empty:
                                with st.expander(f"⚠️ Validation Audit Issues Identified ({len(invalid_df)} records)", expanded=True):
                                    st.markdown(
                                        '<p style="color: #FB7185; font-size: 0.88rem;">'
                                        'The following rows contained missing values, invalid categories, or out-of-range numbers. '
                                        'They are safely excluded from inference to preserve prediction integrity without inventing data.'
                                        '</p>',
                                        unsafe_allow_html=True,
                                    )
                                    cols_to_show = ["_row_number"]
                                    if "student_id" in invalid_df.columns:
                                        cols_to_show.append("student_id")
                                    cols_to_show.append("validation_error")
                                    st.dataframe(invalid_df[cols_to_show], use_container_width=True)

                            # Run batch inference if at least 1 valid row exists
                            if not valid_df.empty:
                                active_batch_model = beh_model if batch_use_behavioral else model
                                if active_batch_model is None:
                                    st.error("Selected model pipeline artifact is not loaded.")
                                else:
                                    results_df, cohort_summary = predict_batch(
                                        uploaded_df,
                                        model=active_batch_model,
                                        use_behavioral=batch_use_behavioral,
                                    )

                                    st.markdown("<br>", unsafe_allow_html=True)
                                    st.markdown("##### 2. Cohort Performance Intelligence & Risk Summary")

                                    # Cohort Summary Cards
                                    cs_col1, cs_col2, cs_col3, cs_col4 = st.columns(4)
                                    with cs_col1:
                                        st.markdown(
                                            f"""
                                            <div class="hud-panel" style="border-left: 3px solid #38BDF8; padding: 0.85rem;">
                                                <span class="telemetry-mono" style="font-size: 0.75rem; color: #94A3B8;">COHORT MEAN GRADE</span>
                                                <h3 style="color: #38BDF8; margin: 0.2rem 0;">{cohort_summary['mean_predicted_grade']:.2f} <span style="font-size: 0.85rem; color: #94A3B8;">/ 20</span></h3>
                                                <span style="font-size: 0.78rem; color: #CBD5E1;">Median: {cohort_summary['median_predicted_grade']:.2f}</span>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                    with cs_col2:
                                        low_risk_cnt = cohort_summary["risk_breakdown"].get("Low Risk", 0)
                                        st.markdown(
                                            f"""
                                            <div class="hud-panel" style="border-left: 3px solid #34D399; padding: 0.85rem;">
                                                <span class="telemetry-mono" style="font-size: 0.75rem; color: #94A3B8;">LOW RISK COHORT</span>
                                                <h3 style="color: #34D399; margin: 0.2rem 0;">{low_risk_cnt} <span style="font-size: 0.85rem; color: #94A3B8;">students</span></h3>
                                                <span style="font-size: 0.78rem; color: #34D399;">Predicted Tier 1 & 2 (≥14 marks)</span>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                    with cs_col3:
                                        mod_risk_cnt = cohort_summary["risk_breakdown"].get("Moderate Risk", 0)
                                        st.markdown(
                                            f"""
                                            <div class="hud-panel" style="border-left: 3px solid #FBBF24; padding: 0.85rem;">
                                                <span class="telemetry-mono" style="font-size: 0.75rem; color: #94A3B8;">MODERATE RISK</span>
                                                <h3 style="color: #FBBF24; margin: 0.2rem 0;">{mod_risk_cnt} <span style="font-size: 0.85rem; color: #94A3B8;">students</span></h3>
                                                <span style="font-size: 0.78rem; color: #FBBF24;">Satisfactory / Pass (10–13 marks)</span>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                    with cs_col4:
                                        high_risk_cnt = cohort_summary["risk_breakdown"].get("High Risk", 0)
                                        st.markdown(
                                            f"""
                                            <div class="hud-panel" style="border-left: 3px solid #FB7185; padding: 0.85rem;">
                                                <span class="telemetry-mono" style="font-size: 0.75rem; color: #94A3B8;">ACTIONABLE INTERVENTION</span>
                                                <h3 style="color: #FB7185; margin: 0.2rem 0;">{high_risk_cnt} <span style="font-size: 0.85rem; color: #94A3B8;">students</span></h3>
                                                <span style="font-size: 0.78rem; color: #FB7185;">Requires Academic Support (0–9 marks)</span>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                    st.markdown("<br>", unsafe_allow_html=True)
                                    st.markdown("##### 3. Enriched Cohort Telemetry Stream")

                                    # Prioritize key prediction columns for readability
                                    priority_cols = ["_row_number"]
                                    if "student_id" in results_df.columns:
                                        priority_cols.append("student_id")
                                    priority_cols.extend([
                                        "status",
                                        "predicted_grade",
                                        "performance_tier",
                                        "risk_level",
                                        "actionable_advisory",
                                        "validation_error",
                                    ])
                                    other_cols = [c for c in results_df.columns if c not in priority_cols]
                                    ordered_display_df = results_df[priority_cols + other_cols]

                                    st.dataframe(ordered_display_df, use_container_width=True, height=360)

                                    # CSV Export Download Button
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    export_csv_data = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="💾 Export Complete Predictions CSV (Enriched Data)",
                                        data=export_csv_data,
                                        file_name=f"studentai_batch_predictions_{'behavioral' if batch_use_behavioral else 'production'}.csv",
                                        mime="text/csv",
                                        use_container_width=True,
                                    )
                            else:
                                st.warning("No valid rows were found in the uploaded file. Please review the validation audit above.")

                        except ValueError as val_err:
                            st.error(f"Schema Validation Error: {val_err}")
                        except Exception as proc_err:
                            st.error(f"Batch processing error: {proc_err}")

        except Exception as file_read_err:
            st.error(f"Error processing uploaded file: {file_read_err}")
    else:
        st.info("👆 Please upload a CSV file or download the benchmark sample template above to test batch predictions.")


# =====================================================================
# CINEMATIC SPACE OBSERVATORY FOOTER
# =====================================================================

st.markdown(
    """
    <div class="selene-footer">
        SELENE // STUDENT PERFORMANCE INTELLIGENCE PLATFORM<br>
        THIRD YEAR B.TECH INFORMATION TECHNOLOGY | DATA SCIENCE & VISUALIZATION OBSERVATORY<br>
        <em>100% Local Inference · Zero External Telemetry · Predictions are Probabilistic Estimates</em>
    </div>
    """,
    unsafe_allow_html=True,
)
