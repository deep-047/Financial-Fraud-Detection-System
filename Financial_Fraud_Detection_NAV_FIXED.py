import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "bank_fraud.csv"
MODELS_DIR = ROOT / "models"
MODEL_RESULTS_PATH = MODELS_DIR / "model_comparison.csv"


# =========================================================
# MODEL ARTIFACT CONFIGURATION
# =========================================================

MODEL_CANDIDATES = [
    {
        "name": "LightGBM",
        "model": MODELS_DIR / "lightgbm.pkl",
        "features": MODELS_DIR / "lgbm_features.pkl",
        "threshold": MODELS_DIR / "lgbm_threshold.pkl",
    },
    {
        "name": "XGBoost",
        "model": MODELS_DIR / "xgboost.pkl",
        "features": MODELS_DIR / "xgb_features.pkl",
        "threshold": MODELS_DIR / "xgb_threshold.pkl",
    },
    {
        "name": "Balanced Random Forest",
        "model": MODELS_DIR / "balanced_random_forest.pkl",
        "features": MODELS_DIR / "rf_features.pkl",
        "threshold": MODELS_DIR / "rf_threshold.pkl",
    },
    {
        "name": "Logistic Regression",
        "model": MODELS_DIR / "logistic_regression.pkl",
        "features":  None,
        "threshold": MODELS_DIR / "threshold.pkl",
    },
]


# =========================================================
# CUSTOM CSS — PREMIUM FINTECH DESIGN (VISUAL ONLY)
# =========================================================

st.markdown(
    """
    <style>
    :root {
      --bg:#060d18; --panel:#0d1b2b; --panel2:#12263b; --line:rgba(100,190,225,.20);
      --cyan:#42e3f5; --blue:#568cff; --text:#f1f7ff; --muted:#91a8c0;
    }
    html, body, [class*="css"] {font-family:Inter,"Segoe UI",Arial,sans-serif;}
    .stApp {background:radial-gradient(900px 500px at 82% -8%,rgba(0,190,230,.14),transparent 62%),radial-gradient(700px 480px at -8% 30%,rgba(63,104,220,.12),transparent 65%),linear-gradient(145deg,#050c16,#081626 55%,#091a2c)!important;color:var(--text);}
    [data-testid="stAppViewContainer"],[data-testid="stMain"]{background:transparent!important;}
    [data-testid="stHeader"]{background:rgba(5,13,24,.94)!important;border-bottom:1px solid rgba(69,190,225,.12);height:2.7rem;}
    [data-testid="stToolbar"]{background:transparent!important;}
    [data-testid="stMainBlockContainer"]{max-width:1600px!important;padding:1.15rem clamp(1.1rem,3vw,2.7rem) 3rem!important;}

    /* Premium navigation rail */
    [data-testid="stSidebar"]{background:radial-gradient(ellipse at 12% 0%,rgba(0,190,220,.17),transparent 38%),linear-gradient(180deg,#061321 0%,#091a2c 58%,#0b2035 100%)!important;border-right:1px solid rgba(57,213,232,.22);box-shadow:12px 0 42px rgba(0,0,0,.25);}
    [data-testid="stSidebar"]>div:first-child{padding-top:1.25rem;}
    [data-testid="stSidebar"] *{color:#dceafa;}
    [data-testid="stSidebar"] h1{font-size:1.34rem!important;font-weight:900!important;letter-spacing:.7px;line-height:1.25;text-transform:uppercase;}
    [data-testid="stSidebar"] [data-testid="stRadio"]>label{color:#779bb9!important;text-transform:uppercase;letter-spacing:2px;font-size:.67rem;font-weight:900;}
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"]{gap:8px;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label{padding:.78rem .9rem;border:1px solid transparent;border-radius:13px;background:rgba(255,255,255,.018);transition:all .2s ease;font-weight:650;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:rgba(57,213,232,.09);border-color:rgba(57,213,232,.25);transform:translateX(2px);}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:linear-gradient(105deg,rgba(8,164,199,.28),rgba(63,115,205,.15))!important;border:1px solid rgba(57,213,232,.46)!important;box-shadow:inset 3px 0 var(--cyan),0 9px 24px rgba(0,0,0,.22);color:#fff!important;}

    /* Strong typographic hierarchy */
    h1,h2,h3,h4,h5,h6{color:#f2f8ff!important;letter-spacing:-.45px;}
    h1{font-size:1.95rem!important;font-weight:900!important;}
    h2{font-size:1.42rem!important;font-weight:850!important;margin-top:.55rem!important;padding-bottom:.5rem;border-bottom:1px solid rgba(86,158,194,.17);}
    h3{font-size:1.08rem!important;font-weight:800!important;}
    p,li,label{color:#d5e4f3;}
    .main-title{display:block;position:relative;padding:1.15rem 1.55rem 1rem;margin:.1rem 0 .35rem;border:1px solid rgba(73,195,225,.30);border-radius:20px;background:radial-gradient(ellipse at 92% 0%,rgba(43,196,225,.20),transparent 42%),linear-gradient(115deg,#142e49 0%,#0d2239 56%,#091a2d 100%);box-shadow:0 16px 42px rgba(0,0,0,.27),inset 0 1px rgba(255,255,255,.07);color:#f5fbff;font-size:clamp(1.55rem,2.45vw,2.2rem);line-height:1.2;font-weight:950;letter-spacing:-1px;text-shadow:0 0 25px rgba(57,213,232,.16);}
    .main-title:after{content:"FRAUD INTELLIGENCE  /  REAL-TIME TRANSACTION RISK";display:block;margin-top:.62rem;color:#6fe1ef;font-size:.60rem;line-height:1.4;font-weight:900;letter-spacing:2.15px;}
    .section-caption{color:#91abc4!important;margin:.3rem 0 1rem;font-size:.91rem;letter-spacing:.2px;}

    /* KPI cards: premium glass + individual accent rails */
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]){gap:1rem!important;}
    [data-testid="stMetric"]{position:relative;min-height:128px;padding:1.18rem 1.25rem 1.05rem;border-radius:18px;border:1px solid rgba(83,178,215,.27);background:radial-gradient(ellipse at 100% 0%,rgba(40,174,213,.17),transparent 48%),linear-gradient(145deg,#142e49 0%,#0b1e32 100%);box-shadow:0 13px 30px rgba(0,0,0,.24),inset 0 1px rgba(255,255,255,.055);transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;overflow:hidden;}
    [data-testid="stMetric"]:before{content:"";position:absolute;left:18px;right:18px;top:0;height:2px;background:linear-gradient(90deg,transparent,var(--cyan),var(--blue),transparent);opacity:.95;}
    [data-testid="stMetric"]:hover{transform:translateY(-4px);border-color:rgba(57,213,232,.58);box-shadow:0 18px 38px rgba(0,0,0,.30),0 0 24px rgba(57,213,232,.08);}
    [data-testid="stMetricLabel"] p{color:#9bb4cb!important;font-size:.70rem!important;font-weight:900!important;letter-spacing:1px;text-transform:uppercase;}
    [data-testid="stMetricValue"]{color:#f5fbff!important;font-size:clamp(1.35rem,2vw,1.9rem)!important;font-weight:900!important;letter-spacing:-.8px;}
    [data-testid="stMetricDelta"]{color:var(--cyan)!important;}

    /* Chart surfaces and tables */
    [data-testid="stPlotlyChart"]{margin:.35rem 0 1.1rem;padding:.75rem .8rem .35rem;border-radius:18px;border:1px solid rgba(76,174,215,.23);background:linear-gradient(145deg,rgba(17,39,62,.97),rgba(8,22,38,.97));box-shadow:0 13px 30px rgba(0,0,0,.22),inset 0 1px rgba(255,255,255,.035);}
    [data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid rgba(76,174,215,.24);border-radius:15px;background:#0d2034;padding:4px;box-shadow:0 10px 26px rgba(0,0,0,.18);overflow:hidden;}
    [data-testid="stExpander"],[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid rgba(76,174,215,.20)!important;border-radius:16px!important;background:rgba(14,32,52,.78)!important;}
    [data-testid="stTabs"] button{color:#a6bdd2;font-weight:750;}
    [data-testid="stTabs"] button[aria-selected="true"]{color:var(--cyan)!important;border-bottom:2px solid var(--cyan)!important;}

    /* Controls */
    [data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stDateInput"] input,[data-testid="stSelectbox"] div[data-baseweb="select"],[data-testid="stMultiSelect"] div[data-baseweb="select"],[data-baseweb="select"]>div,input,textarea{background:#10243a!important;color:#edf7ff!important;border-color:#294761!important;border-radius:10px!important;}
    [data-baseweb="popover"],[data-baseweb="menu"]{background:#10243a!important;border:1px solid #294761!important;}
    [data-testid="stSlider"] [role="slider"]{background:var(--cyan);}
    .stButton>button,.stDownloadButton>button{min-height:43px;padding:0 18px;border-radius:12px!important;border:1px solid rgba(57,213,232,.46)!important;background:linear-gradient(105deg,#087f9f,#12b8d2 55%,#397fff)!important;color:#041321!important;font-weight:900!important;box-shadow:0 8px 22px rgba(5,157,193,.20);transition:all .18s ease;}
    .stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-2px);border-color:#82efff!important;box-shadow:0 11px 28px rgba(37,217,242,.26);}
    [data-testid="stAlert"]{border-radius:14px!important;border:1px solid rgba(87,172,210,.24)!important;background:rgba(15,36,58,.90)!important;box-shadow:0 8px 22px rgba(0,0,0,.14);}
    hr{border-color:rgba(94,151,187,.19)!important;}
    .stCaption,[data-testid="stCaptionContainer"]{color:#91a9c1!important;}
    a{color:#51d9ec!important;} footer{visibility:hidden;}
    ::-webkit-scrollbar{width:9px;height:9px;} ::-webkit-scrollbar-track{background:#071322;} ::-webkit-scrollbar-thumb{background:#294963;border-radius:10px;} ::-webkit-scrollbar-thumb:hover{background:#0c9dbb;}
    @media(max-width:900px){[data-testid="stMainBlockContainer"]{padding:1rem 1rem 2rem!important;}.main-title{font-size:1.5rem;padding:1rem;}[data-testid="stMetric"]{min-height:108px;padding:14px;}[data-testid="stMetricValue"]{font-size:1.3rem!important;}}


    /* FINAL REFINEMENT: spacing, hierarchy and executive dashboard polish */
    [data-testid="stHeader"] {
        height: 3.15rem !important;
        min-height: 3.15rem !important;
        background: rgba(5, 13, 24, .98) !important;
        backdrop-filter: blur(14px);
        z-index: 999990 !important;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 3.25rem !important;
        padding-bottom: 3.5rem !important;
    }
    [data-testid="stMain"] > div:first-child {
        padding-top: .35rem !important;
    }
    .main-title {
        min-height: 112px;
        display: flex !important;
        flex-direction: column;
        justify-content: center;
        padding: 1.15rem 1.65rem 1.05rem !important;
        margin: .15rem 0 .55rem !important;
        border-radius: 20px !important;
        border: 1px solid rgba(75, 205, 235, .32) !important;
        background:
            radial-gradient(ellipse at 88% 8%, rgba(43, 211, 239, .20), transparent 36%),
            linear-gradient(112deg, #142f4b 0%, #10263f 48%, #0a1b2e 100%) !important;
        box-shadow: 0 18px 42px rgba(0,0,0,.25), inset 0 1px rgba(255,255,255,.07) !important;
        font-size: clamp(1.5rem, 2.35vw, 2.12rem) !important;
        line-height: 1.18 !important;
        letter-spacing: -.85px !important;
        overflow: visible !important;
    }
    .main-title:after {
        margin-top: .58rem !important;
        font-size: .58rem !important;
        letter-spacing: 2.35px !important;
        color: #73e3f0 !important;
    }
    .section-caption {
        margin: .45rem 0 1.35rem !important;
        padding-left: .3rem;
        font-size: .9rem !important;
        color: #9ab3ca !important;
    }
    /* Give the four overview KPIs a subtle individual identity */
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stMetric"] {
        min-height: 136px !important;
        padding: 1.25rem 1.3rem 1.1rem !important;
        border-radius: 17px !important;
        background: linear-gradient(145deg, rgba(20,46,71,.98), rgba(10,27,45,.98)) !important;
    }
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stMetric"]:nth-child(1) {
        border-top: 2px solid #35d8ef !important;
    }
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stMetric"]:nth-child(2) {
        border-top: 2px solid #568cff !important;
    }
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stMetric"]:nth-child(3) {
        border-top: 2px solid #39d6a0 !important;
    }
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stMetric"]:nth-child(4) {
        border-top: 2px solid #9b8cff !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: .69rem !important;
        letter-spacing: 1.15px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: clamp(1.35rem, 1.9vw, 1.85rem) !important;
        letter-spacing: -.9px !important;
    }
    h1, h2, h3 { text-shadow: 0 2px 18px rgba(0,0,0,.15); }
    h2 {
        margin-top: .8rem !important;
        margin-bottom: .45rem !important;
        padding-bottom: .62rem !important;
        border-bottom: 1px solid rgba(92,166,202,.18) !important;
    }
    [data-testid="stPlotlyChart"] {
        padding: .9rem .95rem .45rem !important;
        border-radius: 18px !important;
        background: linear-gradient(145deg, rgba(17,39,62,.94), rgba(8,22,38,.96)) !important;
        border: 1px solid rgba(76,174,215,.24) !important;
        box-shadow: 0 14px 32px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.035) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.65rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        min-height: 48px;
        display: flex;
        align-items: center;
    }
    [data-testid="stDataFrame"] {
        border-radius: 15px !important;
        border: 1px solid rgba(76,174,215,.25) !important;
        box-shadow: 0 12px 28px rgba(0,0,0,.19) !important;
    }
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] { padding-top: 2.2rem !important; }
        .main-title { min-height: 96px; }
    }

    /* NAVIGATION FIX: keep Streamlit's sidebar expand/collapse control visible */
    header[data-testid="stHeader"] {
        display: flex !important;
        visibility: visible !important;
        height: 2.7rem !important;
        min-height: 2.7rem !important;
        background: rgba(5,13,24,.94) !important;
        border-bottom: 1px solid rgba(69,190,225,.12) !important;
    }
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.85rem !important;
        padding-bottom: 3.5rem !important;
    }
    [data-testid="stMain"] > div:first-child {
        padding-top: 0 !important;
    }
    h1 a, h2 a, h3 a, h4 a {
        display: none !important;
    }
    .main-title {
        margin-top: .25rem !important;
        min-height: 118px !important;
    }
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] { padding-top: 1.2rem !important; }
        .main-title { min-height: 98px !important; }
    }
    

    /* NAVIGATION UPGRADE — premium fintech rail only */
    [data-testid="stSidebar"] {
        background:
          radial-gradient(ellipse at 18% 0%, rgba(48,211,239,.18), transparent 34%),
          radial-gradient(ellipse at 100% 55%, rgba(75,117,255,.10), transparent 42%),
          linear-gradient(180deg, #071525 0%, #091a2c 52%, #0b2037 100%) !important;
        border-right: 1px solid rgba(70,205,235,.25) !important;
        box-shadow: 16px 0 44px rgba(0,0,0,.28) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.6rem 1.05rem 1.4rem !important;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 1.22rem !important;
        line-height: 1.22 !important;
        letter-spacing: 1.25px !important;
        font-weight: 950 !important;
        padding: .35rem .15rem .15rem !important;
        text-shadow: 0 0 22px rgba(47,211,239,.16);
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #86a6c1 !important;
        font-size: .78rem !important;
        letter-spacing: .25px !important;
    }
    [data-testid="stSidebar"] hr {
        margin: 1.15rem 0 !important;
        border-color: rgba(92,170,205,.16) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        padding: .78rem .9rem !important;
        margin: 0 0 .55rem !important;
        border-radius: 14px !important;
        border: 1px solid rgba(118,174,207,.09) !important;
        background: linear-gradient(110deg, rgba(255,255,255,.045), rgba(255,255,255,.012)) !important;
        color: #91abc3 !important;
        font-size: .88rem !important;
        font-weight: 720 !important;
        letter-spacing: .1px !important;
        transition: background .2s ease, border-color .2s ease, transform .2s ease, box-shadow .2s ease !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label:hover {
        transform: translateX(3px) !important;
        color: #eafaff !important;
        background: linear-gradient(110deg, rgba(35,199,226,.13), rgba(63,116,221,.08)) !important;
        border-color: rgba(61,207,233,.28) !important;
        box-shadow: 0 7px 20px rgba(0,0,0,.16) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label:has(input:checked) {
        color: #f4fcff !important;
        background: linear-gradient(105deg, rgba(16,166,197,.30), rgba(48,103,185,.20)) !important;
        border: 1px solid rgba(59,218,239,.52) !important;
        box-shadow: inset 3px 0 0 #39dff2, 0 9px 24px rgba(0,0,0,.22), 0 0 18px rgba(35,193,222,.08) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: .16rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color: inherit !important;
        font-weight: inherit !important;
        font-size: inherit !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label input {
        accent-color: #35d9ef !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label:first-child {
        color: #7fa7c5 !important;
        font-size: .66rem !important;
        font-weight: 900 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        background: linear-gradient(110deg, rgba(48,91,126,.20), rgba(255,255,255,.025)) !important;
        border: 1px solid rgba(105,172,207,.12) !important;
        box-shadow: inset 0 1px rgba(255,255,255,.035) !important;
        margin-bottom: .7rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label:first-child:hover {
        transform: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label:not(:first-child) {
        min-height: 49px !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #8da9c2;
    }

    /* ICON-LED PREMIUM NAVIGATION */
    .nav-brand {
        position: relative;
        text-align: center;
        padding: 1.1rem .55rem 1.15rem;
        margin: .05rem 0 1.1rem;
        border: 1px solid rgba(55,204,237,.18);
        border-radius: 18px;
        background: radial-gradient(circle at 50% 0%, rgba(37,217,242,.17), transparent 65%),
                    linear-gradient(145deg, rgba(16,43,69,.88), rgba(7,22,39,.72));
        box-shadow: inset 0 1px rgba(255,255,255,.045), 0 12px 28px rgba(0,0,0,.18);
    }
    .nav-brand-icon {
        width: 58px; height: 58px; margin: 0 auto .65rem;
        display: grid; place-items: center;
        border-radius: 18px;
        color: #42e5f6; font-size: 2.15rem; font-weight: 900;
        background: linear-gradient(145deg, rgba(29,191,222,.22), rgba(74,112,255,.16));
        border: 1px solid rgba(58,216,242,.42);
        box-shadow: 0 0 24px rgba(28,197,230,.15), inset 0 1px rgba(255,255,255,.12);
        text-shadow: 0 0 16px rgba(45,222,245,.55);
    }
    .nav-brand-title { color: #f0f8ff; font-size: 1.05rem; font-weight: 900; letter-spacing: 1.1px; }
    .nav-brand-subtitle { color: #8caec8; font-size: .61rem; font-weight: 800; letter-spacing: 1.45px; margin-top: .3rem; }
    .nav-brand-status {
        display: inline-flex; align-items: center; gap: .42rem;
        margin-top: .85rem; padding: .32rem .65rem;
        border-radius: 999px; color: #8debd9; font-size: .59rem;
        font-weight: 850; letter-spacing: 1px;
        border: 1px solid rgba(49,211,174,.22); background: rgba(22,171,135,.08);
    }
    .nav-brand-status span, .nav-footer-dot {
        width: 7px; height: 7px; display: inline-block; border-radius: 50%;
        background: #2ce0ad; box-shadow: 0 0 10px rgba(44,224,173,.8);
    }
    .nav-section-label {
        color: #6f95b4; font-size: .64rem; font-weight: 900;
        letter-spacing: 2px; padding: .25rem .65rem .55rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] { gap: .42rem !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:not(:first-child) {
        min-height: 53px !important; padding: .72rem .82rem !important;
        border-radius: 14px !important;
        border: 1px solid rgba(105,166,204,.10) !important;
        background: linear-gradient(105deg, rgba(19,43,67,.82), rgba(10,28,47,.68)) !important;
        color: #b6cce0 !important;
        font-size: .86rem !important; font-weight: 720 !important;
        box-shadow: inset 0 1px rgba(255,255,255,.025);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:not(:first-child) p {
        color: inherit !important; font-size: inherit !important; font-weight: inherit !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(105deg, rgba(12,164,197,.30), rgba(51,99,190,.22)) !important;
        border: 1px solid rgba(49,218,240,.62) !important;
        color: #f5fdff !important;
        box-shadow: inset 3px 0 #35e2f4, 0 0 22px rgba(25,195,224,.14), 0 9px 22px rgba(0,0,0,.2) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover:not(:has(input:checked)) {
        transform: translateX(3px) !important;
        border-color: rgba(55,208,235,.36) !important;
        background: linear-gradient(105deg, rgba(22,119,153,.22), rgba(38,73,133,.16)) !important;
        color: #ecfaff !important;
    }
    .nav-sidebar-footer {
        display: flex; align-items: center; gap: .6rem;
        margin-top: 1.15rem; padding: .75rem .7rem;
        border-radius: 13px; border: 1px solid rgba(95,159,197,.14);
        background: linear-gradient(110deg, rgba(17,42,66,.8), rgba(10,27,45,.72));
    }
    .nav-footer-shield { color: #47dff2; font-size: 1.35rem; }
    .nav-sidebar-footer strong { display: block; color: #b9d1e5; font-size: .58rem; letter-spacing: 1px; }
    .nav-sidebar-footer small { display: block; color: #718fa9; font-size: .57rem; margin-top: .18rem; }
    .nav-footer-dot { margin-left: auto; flex: 0 0 7px; }

    /* FINAL SIDEBAR POLISH: clean icon-led controls, no default radio bullets */
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: .72rem !important;
        min-height: 58px !important;
        margin: 0 0 .62rem !important;
        padding: .82rem .95rem !important;
        border-radius: 15px !important;
        position: relative !important;
        overflow: hidden !important;
        background: linear-gradient(115deg, rgba(18,43,68,.92), rgba(9,25,43,.82)) !important;
        border: 1px solid rgba(105,177,211,.13) !important;
        box-shadow: inset 0 1px rgba(255,255,255,.035), 0 7px 18px rgba(0,0,0,.10) !important;
        color: #b9cde0 !important;
        font-size: .9rem !important;
        font-weight: 700 !important;
        letter-spacing: .05px !important;
        transition: transform .2s ease, border-color .2s ease, background .2s ease, box-shadow .2s ease !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label::before {
        content: "";
        position: absolute;
        left: 0; top: 13%; bottom: 13%; width: 3px;
        border-radius: 0 5px 5px 0;
        background: transparent;
        transition: background .2s ease, box-shadow .2s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label p {
        color: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
        line-height: 1.35 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover {
        transform: translateX(4px) !important;
        border-color: rgba(48,214,239,.38) !important;
        background: linear-gradient(110deg, rgba(18,117,151,.28), rgba(36,72,132,.22)) !important;
        box-shadow: 0 9px 23px rgba(0,0,0,.2), 0 0 16px rgba(35,194,225,.07) !important;
        color: #f1fbff !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
        transform: translateX(2px) !important;
        color: #f5fdff !important;
        background: linear-gradient(110deg, rgba(8,153,188,.36), rgba(42,89,161,.27)) !important;
        border: 1px solid rgba(49,222,245,.62) !important;
        box-shadow: inset 0 1px rgba(255,255,255,.08), 0 0 0 1px rgba(35,199,229,.08), 0 0 24px rgba(24,190,222,.16), 0 10px 24px rgba(0,0,0,.22) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked)::before {
        background: #38e2f4;
        box-shadow: 0 0 12px rgba(56,226,244,.9);
    }
    .nav-brand {
        padding: 1.25rem .65rem 1.2rem !important;
        border-radius: 20px !important;
        background: radial-gradient(ellipse at 50% 0%, rgba(42,216,241,.2), transparent 58%),
                    linear-gradient(150deg, rgba(17,47,76,.96), rgba(7,21,38,.92)) !important;
        border-color: rgba(57,211,239,.28) !important;
        box-shadow: inset 0 1px rgba(255,255,255,.07), 0 15px 32px rgba(0,0,0,.22), 0 0 22px rgba(26,185,218,.06) !important;
    }
    .nav-brand-icon {
        width: 64px !important; height: 64px !important;
        border-radius: 20px !important;
        font-size: 2.3rem !important;
        background: linear-gradient(145deg, rgba(26,197,228,.28), rgba(83,111,255,.24)) !important;
        box-shadow: 0 0 28px rgba(28,197,230,.2), inset 0 1px rgba(255,255,255,.16) !important;
    }
    .nav-sidebar-footer {
        margin-top: 1.35rem !important;
        padding: .9rem .8rem !important;
        border-radius: 15px !important;
        border-color: rgba(71,191,220,.2) !important;
        background: linear-gradient(115deg, rgba(16,47,71,.9), rgba(8,27,45,.88)) !important;
        box-shadow: inset 0 1px rgba(255,255,255,.04), 0 8px 22px rgba(0,0,0,.13) !important;
    }
    .nav-footer-shield { font-size: 1.5rem !important; text-shadow: 0 0 14px rgba(55,218,241,.5); }

    </style>
    """,
    unsafe_allow_html=True,
)


def show_chart(fig, *args, **kwargs):
    """Render every Plotly chart with a consistent transparent dark-fintech canvas."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#dceafa"),
        colorway=["#35d8ef", "#568cff", "#8b7cff", "#39d6a0", "#ff7185", "#f6bd60"],
        title_font=dict(color="#f2f8ff", size=17),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="rgba(133,174,205,.22)", tickfont=dict(color="#a9bfd3"), title_font=dict(color="#b8ccdc")),
        yaxis=dict(showgrid=True, gridcolor="rgba(133,174,205,.12)", zeroline=False, linecolor="rgba(133,174,205,.18)", tickfont=dict(color="#a9bfd3"), title_font=dict(color="#b8ccdc")),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c7d9e9"),
        ),
        margin=dict(l=18, r=18, t=58, b=22),
    )
    return st.plotly_chart(fig, *args, **kwargs)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_data(path_string):

    data = pd.read_csv(path_string)

    if "transaction_date" in data.columns:
        data["transaction_date"] = pd.to_datetime(
            data["transaction_date"],
            errors="coerce"
        )

    if "is_fraud" in data.columns:
        data["is_fraud"] = pd.to_numeric(
            data["is_fraud"],
            errors="coerce"
        ).fillna(0).astype(int)

    if "transaction_amount" in data.columns:
        data["transaction_amount"] = pd.to_numeric(
            data["transaction_amount"],
            errors="coerce"
        )

    return data


@st.cache_data
def load_csv(path_string):

    if os.path.exists(path_string):
        return pd.read_csv(path_string)

    return pd.DataFrame()


def load_pickle(path):

    try:
        return joblib.load(path)

    except Exception as exc:
        st.warning(f"Could not load {path.name}: {exc}")
        return None


# =========================================================
# FEATURE NAME EXTRACTION
# =========================================================

def extract_feature_names(feature_artifact, model):

    if feature_artifact is not None:

        if isinstance(feature_artifact, dict):

            for key in [
                "features",
                "feature_names",
                "columns",
                "X_columns",
            ]:

                if key in feature_artifact:
                    return list(feature_artifact[key])

        if isinstance(
            feature_artifact,
            (list, tuple, np.ndarray, pd.Index)
        ):
            return list(feature_artifact)

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "feature_name_"):

        names = model.feature_name_()

        if names:
            return list(names)

    return None


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_dashboard_features(data):

    df = data.copy()

    df["_original_order"] = np.arange(len(df))

    # Convert numeric columns
    numeric_columns = [
        "transaction_amount",
        "failed_attempts",
        "distance_from_home_km",
    ]

    for col in numeric_columns:

        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    if "transaction_date" in df.columns:

        df["transaction_date"] = pd.to_datetime(
            df["transaction_date"],
            errors="coerce"
        )

    # Sort in the same chronological order used by training
    sort_columns = [
        col for col in [
            "customer_id",
            "transaction_date",
            "transaction_time",
        ]
        if col in df.columns
    ]

    if sort_columns:

        df = df.sort_values(
            sort_columns,
            kind="mergesort"
        )

    # Historical mean spend using previous transactions
    if {
        "customer_id",
        "transaction_amount",
    }.issubset(df.columns):

        df["Hist_Mean_Spend"] = (
            df.groupby("customer_id")["transaction_amount"]
            .transform(
                lambda x: (
                    x.shift(1)
                    .expanding()
                    .mean()
                )
            )
        )

        df["Hist_Mean_Spend"] = (
            df["Hist_Mean_Spend"].fillna(0)
        )

        df["Spend_Ratio"] = (
            df["transaction_amount"]
            / (df["Hist_Mean_Spend"] + 1)
        )

    else:

        df["Hist_Mean_Spend"] = 0
        df["Spend_Ratio"] = 0

    # Date-based features
    if "transaction_date" in df.columns:

        df["transaction_month"] = (
            df["transaction_date"].dt.month
        )

        df["transaction_day"] = (
            df["transaction_date"].dt.day
        )

    # Failed attempts feature
    if "failed_attempts" in df.columns:

        df["high_failed_attempts"] = (
            df["failed_attempts"] >= 2
        ).astype(int)

    else:

        df["high_failed_attempts"] = 0

    # Distance feature
    if "distance_from_home_km" in df.columns:

        df["high_distance"] = (
            df["distance_from_home_km"] > 100
        ).astype(int)

    else:

        df["high_distance"] = 0

    # Restore original row order
    df = df.sort_values("_original_order")

    df = df.drop(
        columns=["_original_order"],
        errors="ignore"
    )

    return df


# =========================================================
# MODEL INPUT PREPARATION
# =========================================================


def make_model_input(data, feature_names, model=None):

    if not feature_names:
        raise ValueError(
            "Saved model feature names are not available."
        )

    df = create_dashboard_features(data)

    drop_columns = [
        "transaction_id",
        "customer_id",
        "transaction_date",
        "transaction_time",
        "fraud_type",
        "is_fraud",
    ]

    X = df.drop(
        columns=drop_columns,
        errors="ignore"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Logistic Regression Pipeline:
    # Let its internal preprocessor handle encoding and imputation.
    is_pipeline = (
        hasattr(model, "named_steps")
        and "preprocessor" in model.named_steps
    )

    if is_pipeline:

        missing_features = [
            col for col in feature_names
            if col not in X.columns
        ]

        X = X.reindex(
            columns=feature_names
        )

    else:

        # Tree-based models use one-hot encoded features.
        X = pd.get_dummies(
            X,
            dummy_na=True
        )

        missing_features = [
            col for col in feature_names
            if col not in X.columns
        ]

        X = X.reindex(
            columns=feature_names,
            fill_value=0
        )

        X = X.fillna(0)

    return X, missing_features



# =========================================================
# FRAUD PROBABILITY
# =========================================================

def positive_class_probability(model, X):

    if hasattr(model, "predict_proba"):

        probabilities = np.asarray(
            model.predict_proba(X)
        )

        if probabilities.ndim == 2:

            classes = getattr(
                model,
                "classes_",
                None
            )

            if classes is not None:

                classes = list(classes)

                if 1 in classes:
                    fraud_index = classes.index(1)
                else:
                    fraud_index = 1

            else:
                fraud_index = 1

            return probabilities[:, fraud_index]

        return probabilities.reshape(-1)

    if hasattr(model, "decision_function"):

        scores = np.asarray(
            model.decision_function(X),
            dtype=float
        )

        scores = np.clip(
            scores,
            -30,
            30
        )

        return 1 / (1 + np.exp(-scores))

    return np.asarray(
        model.predict(X),
        dtype=float
    ).reshape(-1)


# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================

@st.cache_resource
def load_prediction_artifacts():

    loaded = []

    for item in MODEL_CANDIDATES:

        model_path = item["model"]
        feature_path = item["features"]
        threshold_path = item["threshold"]

        if not model_path.exists():
            continue

        model = load_pickle(model_path)

        if model is None:
            continue

        feature_artifact = None

        if feature_path is not None and feature_path.exists():
            feature_artifact = load_pickle(feature_path)

        feature_names = extract_feature_names(
            feature_artifact,
            model
        )

        # Do not run prediction without correct feature names
        if not feature_names:
            continue

        threshold = 0.5

        if threshold_path.exists():

            threshold_artifact = load_pickle(
                threshold_path
            )

            try:

                if isinstance(threshold_artifact, dict):

                    threshold = float(
                        threshold_artifact.get(
                            "threshold",
                            threshold_artifact.get(
                                "best_threshold",
                                0.5
                            )
                        )
                    )

                else:

                    threshold = float(
                        np.asarray(
                            threshold_artifact
                        ).reshape(-1)[0]
                    )

            except (
                TypeError,
                ValueError,
                IndexError,
            ):

                threshold = 0.5

        loaded.append({

            "name": item["name"],

            "model": model,

            "feature_names": feature_names,

            "threshold": float(
                np.clip(threshold, 0, 1)
            ),

        })

    return loaded


# =========================================================
# COMMON FUNCTIONS
# =========================================================

def format_money(value):

    return f"${value:,.2f}"


def safe_column(data, column):

    if column in data.columns:
        return data[column]

    return pd.Series(
        [0] * len(data),
        index=data.index
    )


# =========================================================
# LOAD REQUIRED DATA
# =========================================================

@st.cache_resource
def resolve_dataset_path():
    """Use the local CSV when available; otherwise fetch the public Kaggle file."""
    if DATA_PATH.exists():
        return DATA_PATH

    try:
        import kagglehub

        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path = kagglehub.dataset_download(
            "nafiulislam490/bank-transaction-fraud-detection-dataset",
            path="bank_fraud.csv",
            output_dir=str(DATA_PATH.parent),
        )
        downloaded_path = Path(downloaded_path)

        if downloaded_path.is_file():
            return downloaded_path

        candidate = DATA_PATH.parent / "bank_fraud.csv"
        if candidate.is_file():
            return candidate

        raise FileNotFoundError(
            "KaggleHub completed, but bank_fraud.csv was not found."
        )

    except Exception as exc:
        st.error(
            "Dataset not found locally and the public Kaggle download failed: "
            f"{exc}"
        )
        st.info(
            "Ensure kagglehub is listed in requirements.txt and the app can "
            "access the internet."
        )
        st.stop()


try:
    resolved_data_path = resolve_dataset_path()
    df = load_data(str(resolved_data_path))
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()


if "is_fraud" not in df.columns:

    st.error(
        "Dataset must contain an 'is_fraud' column."
    )

    st.stop()


model_results = load_csv(
    str(MODEL_RESULTS_PATH)
)

prediction_artifacts = load_prediction_artifacts()


# =========================================================
# COMMON METRICS
# =========================================================

total_transactions = len(df)

fraud_count = int(
    (df["is_fraud"] == 1).sum()
)

legitimate_count = int(
    (df["is_fraud"] == 0).sum()
)

fraud_rate = (
    fraud_count / total_transactions * 100
    if total_transactions else 0
)

fraud_rate_bps = fraud_rate * 100

fraud_amount = 0

if "transaction_amount" in df.columns:

    fraud_amount = pd.to_numeric(
        df.loc[
            df["is_fraud"] == 1,
            "transaction_amount"
        ],
        errors="coerce"
    ).fillna(0).sum()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown(
    """
    <div class="nav-brand">
      <div class="nav-brand-icon">◈</div>
      <div class="nav-brand-title">FRAUD DETECTION</div>
      <div class="nav-brand-subtitle">FINANCIAL RISK INTELLIGENCE</div>
      <div class="nav-brand-status"><span></span> AI MONITORING ACTIVE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="nav-section-label">WORKSPACE</div>', unsafe_allow_html=True)

nav_icons = {
    "Executive Overview": "▦",
    "Risk Hotspots": "⌖",
    "Exploratory Data Analysis": "▤",
    "Machine Learning": "✧",
    "Fraud Operations": "⬡",
}

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Executive Overview",
        "Risk Hotspots",
        "Exploratory Data Analysis",
        "Machine Learning",
        "Fraud Operations",
    ],
    format_func=lambda item: f"{nav_icons.get(item, '•')}   {item}",
    label_visibility="collapsed",
)

st.sidebar.markdown(
    """
    <div class="nav-sidebar-footer">
      <div class="nav-footer-shield">⬡</div>
      <div><strong>SECURE ENVIRONMENT</strong><small>Fraud intelligence workspace · 2026</small></div>
      <span class="nav-footer-dot"></span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ Financial Fraud Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="section-caption">'
    'AI-Powered Transaction Risk Monitoring'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":

    st.header("Executive Overview")

    st.caption(
        "Overall financial transaction and fraud intelligence"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    c2.metric(
        "Fraud Transactions",
        f"{fraud_count:,}"
    )

    c3.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    c4.metric(
        "Fraud Rate (bps)",
        f"{fraud_rate_bps:,.1f} bps"
    )

    left, right = st.columns(2)

    with left:

        chart_data = pd.DataFrame({

            "Transaction Type": [
                "Fraud",
                "Legitimate"
            ],

            "Count": [
                fraud_count,
                legitimate_count
            ],

        })

        fig = px.pie(
            chart_data,
            names="Transaction Type",
            values="Count",
            hole=0.55,
            title="Fraud vs Legitimate Transactions",
        )

        show_chart(
            fig,
            use_container_width=True,
            theme=None
        )

    with right:

        st.subheader(
            "Fraud by Transaction Date"
        )

        if "transaction_date" in df.columns:

            daily = (
                df.dropna(
                    subset=["transaction_date"]
                )
                .set_index("transaction_date")
                ["is_fraud"]
                .resample("D")
                .sum()
                .reset_index()
            )

            daily.columns = [
                "Date",
                "Fraud Count"
            ]

            fig = px.line(
                daily,
                x="Date",
                y="Fraud Count",
                title="Daily Fraud Transactions",
            )

            show_chart(
                fig,
                use_container_width=True,
                theme=None
            )

        else:

            st.info(
                "Transaction date is unavailable."
            )

    st.divider()

    st.subheader("Financial Impact")

    c1, c2 = st.columns(2)

    c1.metric(
        "Recorded Fraud Transaction Amount",
        format_money(fraud_amount)
    )

    c2.metric(
        "Legitimate Transactions",
        f"{legitimate_count:,}"
    )

    st.info(
        "Dollars Saved is not reported as a confirmed value. "
        "A valid estimate requires held-out model predictions "
        "and explicit financial cost assumptions. "
        "Historical fraud amount is not the same as money saved."
    )

    st.subheader("Recent Transactions")

    recent_columns = [
        "transaction_id",
        "transaction_date",
        "transaction_amount",
        "merchant_category",
        "is_international",
        "is_fraud",
    ]

    available_columns = [
        col for col in recent_columns
        if col in df.columns
    ]

    st.dataframe(
        df[available_columns]
        .tail(10)
        .iloc[::-1],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PAGE 2: RISK HOTSPOTS
# =========================================================

elif page == "Risk Hotspots":

    st.header("Risk Hotspots")

    st.caption(
        "Merchant category and international transaction risk"
    )

    if "merchant_category" in df.columns:

        st.subheader(
            "Merchant Category Risk"
        )

        merchant_risk = (

            df.groupby(
                "merchant_category",
                dropna=False
            )

            .agg(

                Transactions=("is_fraud", "count"),

                Fraud_Count=("is_fraud", "sum"),

            )

            .reset_index()

        )

        merchant_risk["Fraud Rate (%)"] = (

            merchant_risk["Fraud_Count"]

            / merchant_risk["Transactions"]
            .replace(0, np.nan)

            * 100

        )

        merchant_risk = merchant_risk.sort_values(
            "Fraud Rate (%)",
            ascending=False
        )

        fig = px.bar(
            merchant_risk,
            x="merchant_category",
            y="Fraud Rate (%)",
            color="Fraud Rate (%)",
            text_auto=".2f",
            title="Fraud Rate by Merchant Category",
        )

        show_chart(
            fig,
            use_container_width=True,
            theme=None
        )

        st.dataframe(
            merchant_risk,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Merchant category column is unavailable."
        )

    st.divider()

    if "is_international" in df.columns:

        st.subheader(
            "International Transaction Analysis"
        )

        international = (

            df.groupby(
                "is_international",
                dropna=False
            )

            .agg(

                Transactions=("is_fraud", "count"),

                Fraud_Count=("is_fraud", "sum"),

            )

            .reset_index()

        )

        international["Fraud Rate (%)"] = (

            international["Fraud_Count"]

            / international["Transactions"]
            .replace(0, np.nan)

            * 100

        )

        international["Transaction Type"] = (
            international["is_international"].astype(str)
        )

        fig = px.bar(
            international,
            x="Transaction Type",
            y="Transactions",
            color="Transaction Type",
            title="International Transaction Volume",
        )

        show_chart(
            fig,
            use_container_width=True,
            theme=None
        )

        fig = px.bar(
            international,
            x="Transaction Type",
            y="Fraud Rate (%)",
            color="Transaction Type",
            title="Fraud Rate: International vs Domestic",
        )

        show_chart(
            fig,
            use_container_width=True,
            theme=None
        )

        st.dataframe(
            international,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "International transaction column is unavailable."
        )


# =========================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS
# =========================================================

elif page == "Exploratory Data Analysis":

    st.header(
        "Exploratory Data Analysis"
    )

    st.caption(
        "Understanding transaction behaviour and fraud patterns"
    )

    if "transaction_amount" in df.columns:

        st.subheader(
            "Transaction Amount Density"
        )

        amount_data = df.copy()

        amount_data["Transaction Type"] = (
            amount_data["is_fraud"].map({
                0: "Legitimate",
                1: "Fraud"
            })
        )

        fig = px.histogram(
            amount_data,
            x="transaction_amount",
            color="Transaction Type",
            marginal="box",
            barmode="overlay",
            opacity=0.65,
            nbins=60,
            title="Transaction Amount: Fraud vs Legitimate",
        )

        show_chart(
            fig,
            use_container_width=True,
            theme=None
        )

        amount_stats = (

            df.groupby("is_fraud")["transaction_amount"]

            .agg([
                "count",
                "mean",
                "median",
                "min",
                "max"
            ])

            .reset_index()

        )

        amount_stats["Transaction Type"] = (
            amount_stats["is_fraud"].map({
                0: "Legitimate",
                1: "Fraud"
            })
        )

        st.subheader(
            "Transaction Amount Statistics"
        )

        st.dataframe(
            amount_stats.drop(
                columns=["is_fraud"]
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Transaction amount column is unavailable."
        )

    st.divider()

    if "hour_of_day" in df.columns:

        st.subheader(
            "Fraud by Hour of Day"
        )

        hourly = (

            df.groupby("hour_of_day")["is_fraud"]

            .agg([
                "count",
                "sum"
            ])

            .reset_index()

        )

        hourly["Fraud Rate (%)"] = (

            hourly["sum"]

            / hourly["count"].replace(0, np.nan)

            * 100

        )

        fig = px.line(
            hourly,
            x="hour_of_day",
            y="Fraud Rate (%)",
            markers=True,
            title="Hourly Fraud Rate",
        )

        show_chart(
            fig,
            use_container_width=True,
            theme=None
        )

    if "payment_method" in df.columns:

        st.subheader(
            "Fraud by Payment Method"
        )

        payment = (

            df.groupby("payment_method")["is_fraud"]

            .mean()

            .mul(100)

            .reset_index(
                name="Fraud Rate (%)"
            )

        )

        fig = px.bar(
            payment,
            x="payment_method",
            y="Fraud Rate (%)",
            color="Fraud Rate (%)",
            title="Fraud Rate by Payment Method",
        )

        show_chart(
            fig,
            use_container_width=True,
            theme=None
        )


# =========================================================
# PAGE 4: MACHINE LEARNING
# =========================================================


elif page == "Machine Learning":

    st.header("Machine Learning Performance")

    st.caption(
        "Model evaluation, fraud detection performance "
        "and operational threshold analysis."
    )

    # -----------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------

    if model_results.empty:

        st.warning(
            "Model comparison file not found. "
            "Run the model comparison script first."
        )

    else:

        st.subheader("Model Comparison")

        st.dataframe(
            model_results,
            use_container_width=True,
            hide_index=True
        )

        if {"Model", "PR-AUC"}.issubset(model_results.columns):

            fig = px.bar(
                model_results,
                x="Model",
                y="PR-AUC",
                color="Model",
                text_auto=".4f",
                title="Precision-Recall AUC Comparison",
            )

            show_chart(
                fig,
                use_container_width=True,
                theme=None
            )

        metric_columns = [
            col for col in ["Precision", "Recall", "FPR"]
            if col in model_results.columns
        ]

        if metric_columns and "Model" in model_results.columns:

            metric_data = model_results.melt(
                id_vars="Model",
                value_vars=metric_columns,
                var_name="Metric",
                value_name="Score",
            )

            fig = px.bar(
                metric_data,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                title="Model Precision, Recall and FPR",
            )

            show_chart(
                fig,
                use_container_width=True,
                theme=None
            )

        if "Inference Latency (ms)" in model_results.columns:

            fig = px.bar(
                model_results,
                x="Model",
                y="Inference Latency (ms)",
                color="Model",
                title="Batch Average Inference Latency",
            )

            show_chart(
                fig,
                use_container_width=True,
                theme=None
            )

            st.caption(
                "Batch average latency is not single-transaction "
                "API latency. Assignment target: below 25 ms "
                "per API transaction call."
            )

    # -----------------------------------------
    # RECALL AT 95% PRECISION
    # -----------------------------------------

    st.divider()

    st.subheader("Recall at 95% Precision")

    precision_file = MODELS_DIR / "precision_recall_95.csv"

    if precision_file.exists():

        try:

            precision_results = pd.read_csv(precision_file)

            val_result = precision_results[
                precision_results["Dataset"] == "Validation"
            ]

            test_result = precision_results[
                precision_results["Dataset"] == "Test"
            ]

            if not val_result.empty and not test_result.empty:

                val_row = val_result.iloc[0]
                test_row = test_result.iloc[0]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Validation Precision",
                        f"{val_row['Actual Precision'] * 100:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Validation Recall",
                        f"{val_row['Recall'] * 100:.4f}%"
                    )

                with col3:
                    st.metric(
                        "Test Recall",
                        f"{test_row['Recall'] * 100:.4f}%"
                    )

                st.dataframe(
                    precision_results,
                    use_container_width=True,
                    hide_index=True
                )

                if bool(val_row["Target Achieved on Validation"]):

                    st.success(
                        "The 95% precision target was achieved "
                        "on validation data."
                    )

                else:

                    st.warning(
                        "The 95% precision target was not achieved "
                        "on validation data."
                    )

                if test_row["Actual Precision"] < 0.95:

                    st.warning(
                        "The selected validation threshold did not "
                        "achieve 95% precision on independent test data. "
                        "Do not claim that the test target was achieved."
                    )

                st.info(
                    f"Validation-selected threshold: "
                    f"{val_row['Threshold']:.6f}. "
                    "The same threshold was applied to test data."
                )

            else:

                st.warning(
                    "Validation or Test results are missing "
                    "from precision_recall_95.csv."
                )

        except Exception as exc:

            st.error(
                f"Could not load 95% precision results: {exc}"
            )

    else:

        st.warning(
            "Run models/precision_recall_95.py to generate "
            "the 95% precision evaluation report."
        )

    # -----------------------------------------
    # FALSE DECLINE RATE
    # -----------------------------------------
    

    # -----------------------------------------
    # FINAL TEST SET EVALUATION
    # -----------------------------------------

    st.divider()
    st.subheader("Final Temporal Test Evaluation")

    precision_file = MODELS_DIR / "precision_recall_95.csv"

    if precision_file.exists():

        test_results = pd.read_csv(precision_file)

        test_row = test_results[
            test_results["Dataset"] == "Test"
        ]

        if not test_row.empty:

            result = test_row.iloc[0]

            st.caption(
                "Independent temporal test-set evaluation "
                "using the threshold selected on validation data."
            )

            a, b, c, d = st.columns(4)

            a.metric(
                "Test Precision",
                f"{result['Actual Precision'] * 100:.2f}%"
            )

            b.metric(
                "Test Recall",
                f"{result['Recall'] * 100:.4f}%"
            )

            c.metric(
                "Test PR-AUC",
                f"{result['PR-AUC']:.4f}"
            )

            d.metric(
                "Test FPR",
                f"{result['FPR'] * 100:.2f}%"
            )

            st.write("**Test Confusion Matrix**")

            cm1, cm2, cm3, cm4 = st.columns(4)

            cm1.metric("TP", f"{int(result['TP']):,}")
            cm2.metric("FP", f"{int(result['FP']):,}")
            cm3.metric("FN", f"{int(result['FN']):,}")
            cm4.metric("TN", f"{int(result['TN']):,}")

            st.dataframe(
                test_results,
                use_container_width=True,
                hide_index=True
            )

            st.warning(
                "The 95% precision target was not achieved "
                "on the independent test set. Test Recall is "
                "0% at the validation-selected threshold."
            )

        else:

            st.warning("Test evaluation row not found.")

    else:

        st.warning(
            "Run models/precision_recall_95.py first."
        )

    # -----------------------------------------
    # FINANCIAL IMPACT ANALYSIS
    # -----------------------------------------

    st.divider()
    st.subheader("Test Set Financial Impact Analysis")

    dollar_file = MODELS_DIR / "dollar_loss_analysis.csv"

    if dollar_file.exists():

        dollar_results = pd.read_csv(dollar_file)

        st.dataframe(
            dollar_results,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Potentially preventable fraud amount is an estimate, "
            "not confirmed realized savings. Dataset currency and "
            "operational cost assumptions are not established."
        )

        # -------------------------------------------------
        # OPERATIONAL THRESHOLD EVALUATION (SEPARATE REPORT)
        # -------------------------------------------------
        st.divider()
        st.subheader("Operational Threshold Evaluation")
        st.caption(
            "Separate from the 95% precision experiment. This view uses "
            "the operational threshold and held-out test confusion-matrix "
            "counts produced by dollar_loss_analysis.py."
        )

        # Support either a one-row (wide) CSV or Metric/Value CSV.
        metric_values = {}
        if not dollar_results.empty:
            normalized_columns = {
                ''.join(ch.lower() for ch in str(col) if ch.isalnum()): col
                for col in dollar_results.columns
            }
            if {"metric", "value"}.issubset(normalized_columns):
                metric_col = normalized_columns["metric"]
                value_col = normalized_columns["value"]
                for _, metric_row in dollar_results.iterrows():
                    key = ''.join(
                        ch.lower() for ch in str(metric_row[metric_col])
                        if ch.isalnum()
                    )
                    metric_values[key] = metric_row[value_col]
            else:
                first_row = dollar_results.iloc[0]
                for col, value in first_row.items():
                    key = ''.join(ch.lower() for ch in str(col) if ch.isalnum())
                    metric_values[key] = value

        def _read_metric(*aliases):
            for alias in aliases:
                key = ''.join(ch.lower() for ch in alias if ch.isalnum())
                if key in metric_values:
                    try:
                        value = pd.to_numeric(metric_values[key], errors="coerce")
                        if pd.notna(value):
                            return float(value)
                    except (TypeError, ValueError):
                        pass
            return None

        op_threshold = _read_metric("threshold", "operational_threshold", "selected_threshold")
        op_tp = _read_metric("tp", "true_positives")
        op_fp = _read_metric("fp", "false_positives")
        op_fn = _read_metric("fn", "false_negatives")
        op_tn = _read_metric("tn", "true_negatives")

        if None not in (op_tp, op_fp, op_fn, op_tn):
            op_precision = op_tp / (op_tp + op_fp) if (op_tp + op_fp) else 0.0
            op_recall = op_tp / (op_tp + op_fn) if (op_tp + op_fn) else 0.0
            op_fpr = op_fp / (op_fp + op_tn) if (op_fp + op_tn) else 0.0

            st.success(
                f"Operational threshold: {op_threshold:.6f}"
                if op_threshold is not None
                else "Operational threshold evaluation loaded."
            )
            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("Operational Precision", f"{op_precision * 100:.2f}%")
            oc2.metric("Operational Recall", f"{op_recall * 100:.2f}%")
            oc3.metric("Operational FPR", f"{op_fpr * 100:.2f}%")
            oc4.metric("Fraud Detected (TP)", f"{int(op_tp):,}")

            oc5, oc6, oc7 = st.columns(3)
            oc5.metric("False Alerts (FP)", f"{int(op_fp):,}")
            oc6.metric("Missed Fraud (FN)", f"{int(op_fn):,}")
            oc7.metric("Legitimate Approved (TN)", f"{int(op_tn):,}")

            st.caption(
                "These metrics are computed from the operational-threshold "
                "test confusion matrix. They do not represent the 95% "
                "precision experiment above."
            )
        else:
            st.warning(
                "Operational test confusion-matrix columns (TP, FP, FN, TN) "
                "were not found in dollar_loss_analysis.csv. The financial "
                "table is shown above, but operational metrics cannot be "
                "computed from this file. Check that the latest "
                "models/dollar_loss_analysis.py output is present."
            )

    else:

        st.warning(
            "Dollar loss analysis file not found. "
            "Run models/dollar_loss_analysis.py first."
        )

    st.divider()

    st.subheader("False Decline Rate")

    st.caption(
        "False Decline Rate = Legitimate transactions incorrectly "
        "flagged as fraud / All legitimate transactions."
    )

    if not model_results.empty and "FPR" in model_results.columns:

        fdr_data = model_results[["Model", "FPR"]].copy()

        fdr_data["False Decline Rate (%)"] = (
            fdr_data["FPR"] * 100
        )

        st.dataframe(
            fdr_data[["Model", "False Decline Rate (%)"]],
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "False Decline Rate is calculated using FPR. "
            "It represents legitimate transactions incorrectly "
            "flagged by the model at its evaluated threshold."
        )

    # -----------------------------------------
    # PRECISION-RECALL CURVE
    # -----------------------------------------

    
    st.divider()

    st.subheader("Precision-Recall Curve")

    st.warning(
        "The curve and threshold marker below use the full dataset "
        "for diagnostic visualization only. They are not independent "
        "held-out test results. The threshold was selected using "
        "validation data."
    )

    if prediction_artifacts:

        selected_pr_model = st.selectbox(
            "Select model for PR Curve",
            [
                item["name"]
                for item in prediction_artifacts
            ],
            key="pr_model"
        )

        artifact = next(
            item for item in prediction_artifacts
            if item["name"] == selected_pr_model
        )

        try:

            X, missing_features = make_model_input(
                df,
                artifact["feature_names"],
                artifact["model"]
            )

            with st.spinner(
                "Generating Precision-Recall Curve..."
            ):

                probabilities = positive_class_probability(
                    artifact["model"],
                    X
                )

            y_true = df["is_fraud"].astype(int).to_numpy()

            precision, recall, thresholds = (
                precision_recall_curve(
                    y_true,
                    probabilities
                )
            )

            pr_auc = average_precision_score(
                y_true,
                probabilities
            )

            fig = go.Figure()

            # Main PR Curve
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=selected_pr_model,
                    fill="tozeroy",
                )
            )

            # Add 95% precision threshold marker for LightGBM
            if selected_pr_model == "LightGBM":

                threshold_file = (
                    MODELS_DIR / "precision_recall_95.csv"
                )

                if threshold_file.exists():

                    threshold_results = pd.read_csv(
                        threshold_file
                    )

                    validation_row = threshold_results[
                        threshold_results["Dataset"] == "Validation"
                    ]

                    if not validation_row.empty:

                        selected_threshold = float(
                            validation_row.iloc[0]["Threshold"]
                        )

                        threshold_predictions = (
                            probabilities >= selected_threshold
                        ).astype(int)

                        tp = int(
                            ((y_true == 1) &
                             (threshold_predictions == 1)).sum()
                        )

                        fp = int(
                            ((y_true == 0) &
                             (threshold_predictions == 1)).sum()
                        )

                        fn = int(
                            ((y_true == 1) &
                             (threshold_predictions == 0)).sum()
                        )

                        marker_precision = (
                            tp / (tp + fp)
                            if (tp + fp) > 0 else 0
                        )

                        marker_recall = (
                            tp / (tp + fn)
                            if (tp + fn) > 0 else 0
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=[marker_recall],
                                y=[marker_precision],
                                mode="markers+text",
                                name="95% Precision Threshold",
                                marker=dict(
                                    color="red",
                                    size=13,
                                    symbol="diamond",
                                    line=dict(
                                        color="white",
                                        width=2
                                    )
                                ),
                                text=[
                                    f"Threshold: {selected_threshold:.4f}"
                                ],
                                textposition="top right",
                            )
                        )

                        st.info(
                            f"Validation-selected threshold: "
                            f"{selected_threshold:.6f} | "
                            f"Full-data diagnostic Precision: "
                            f"{marker_precision:.4f} | "
                            f"Full-data diagnostic Recall: "
                            f"{marker_recall:.6f}"
                        )

                else:

                    st.warning(
                        "precision_recall_95.csv not found. "
                        "Run models/precision_recall_95.py first."
                    )

            fig.update_layout(
                title=(
                    f"{selected_pr_model} Precision-Recall Curve "
                    f"(AP = {pr_auc:.4f})"
                ),
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            show_chart(
                fig,
                use_container_width=True,
                theme=None
            )

        except Exception as exc:

            st.error(
                f"Could not generate PR Curve: {exc}"
            )

    else:

        st.info(
            "No compatible saved prediction model is available."
        )


# =========================================================
# PAGE 5: FRAUD OPERATIONS
# =========================================================

elif page == "Fraud Operations":

    st.header(
        "Fraud Operations"
    )

    st.caption(
        "Model-scored transaction triage and investigation queue"
    )

    st.info(
        "This is a local academic demonstration. "
        "It is not connected to a live bank or payment network."
    )

    if not prediction_artifacts:

        st.warning(
            "No compatible saved prediction model was found. "
            "Historical fraud records are shown as a fallback. "
            "These are NOT model predictions."
        )

        historical_queue = df[
            df["is_fraud"] == 1
        ].copy()

        st.metric(
            "Historical Fraud Records",
            f"{len(historical_queue):,}"
        )

        show_columns = [
            col for col in [
                "transaction_id",
                "transaction_date",
                "transaction_amount",
                "merchant_category",
                "payment_method",
                "is_international",
                "is_fraud",
            ]
            if col in historical_queue.columns
        ]

        st.dataframe(
            historical_queue[show_columns].head(500),
            use_container_width=True,
            hide_index=True
        )

    else:

        model_names = [
            item["name"]
            for item in prediction_artifacts
        ]

        selected_model = st.selectbox(
            "Select Prediction Model",
            model_names,
            key="operations_model"
        )

        artifact = next(
            item for item in prediction_artifacts
            if item["name"] == selected_model
        )

        try:

            X, missing_features = make_model_input(
                df,
                artifact["feature_names"],
                artifact["model"]
            )

            if missing_features:

                st.warning(
                    f"{len(missing_features)} expected model "
                    "features were not found in the generated input. "
                    "They have been filled with zero. "
                    "Verify that the training and dashboard "
                    "preprocessing pipelines match."
                )

            with st.spinner(
                "Scoring transactions..."
            ):

                start_time = time.perf_counter()

                probabilities = positive_class_probability(
                    artifact["model"],
                    X
                )

                elapsed_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                per_transaction_ms = (
                    elapsed_ms / len(X)
                    if len(X) else 0
                )

            if len(probabilities) != len(df):

                st.error(
                    "Model output size does not match "
                    "the transaction count."
                )

                st.stop()

            queue = df.copy()

            queue["Fraud Probability (PD)"] = np.clip(
                probabilities,
                0,
                1
            )

            queue["Predicted Fraud"] = (
                queue["Fraud Probability (PD)"]
                >= artifact["threshold"]
            ).astype(int)

            # Assignment triage policy
            queue["Risk Level"] = np.select(
                [
                    queue["Fraud Probability (PD)"] < 0.15,

                    queue["Fraud Probability (PD)"] <= 0.65,
                ],
                [
                    "Low",
                    "Medium",
                ],
                default="High",
            )

            queue["Recommended Action"] = np.select(
                [
                    queue["Fraud Probability (PD)"] < 0.15,

                    queue["Fraud Probability (PD)"] <= 0.65,
                ],
                [
                    "Auto Approve",

                    "Step-Up Authentication",
                ],
                default="Hard Decline & Freeze Alert",
            )

            # Queue includes medium and high risk transactions
            high_queue = queue[
                queue["Fraud Probability (PD)"] >= 0.15
            ].copy()

            high_queue = high_queue.sort_values(
                "Fraud Probability (PD)",
                ascending=False
            )

            # Summary metrics
            low_count = int(
                (queue["Risk Level"] == "Low").sum()
            )

            medium_count = int(
                (queue["Risk Level"] == "Medium").sum()
            )

            high_count = int(
                (queue["Risk Level"] == "High").sum()
            )

            st.divider()

            st.subheader(
                "Transaction Risk Summary"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Auto Approve",
                f"{low_count:,}"
            )

            c2.metric(
                "Step-Up Authentication",
                f"{medium_count:,}"
            )

            c3.metric(
                "Hard Decline",
                f"{high_count:,}"
            )

            c4.metric(
                "Batch Avg. Inference",
                f"{per_transaction_ms:.4f} ms"
            )

            st.caption(
                "Inference time is measured for the complete batch "
                "and divided by transaction count. It does not "
                "represent single-request API latency."
            )

            st.divider()

            st.subheader(
                "Risk Distribution"
            )

            risk_distribution = pd.DataFrame({

                "Risk Level": [
                    "Low",
                    "Medium",
                    "High"
                ],

                "Transactions": [
                    low_count,
                    medium_count,
                    high_count
                ]

            })

            fig = px.pie(
                risk_distribution,
                names="Risk Level",
                values="Transactions",
                hole=0.55,
                title="Transaction Risk Distribution",
            )

            show_chart(
                fig,
                use_container_width=True,
                theme=None
            )

            st.divider()

            st.subheader(
                "Fraud Investigation Queue"
            )

            st.caption(
                "Transactions with PD >= 0.15 are displayed first."
            )

            # Risk filter
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                [
                    "Medium",
                    "High"
                ],
                default=[
                    "Medium",
                    "High"
                ]
            )

            filtered_queue = high_queue[
                high_queue["Risk Level"].isin(risk_filter)
            ].copy()

            # Transaction amount filter
            if "transaction_amount" in filtered_queue.columns:

                amount_values = pd.to_numeric(
                    filtered_queue["transaction_amount"],
                    errors="coerce"
                ).fillna(0)

                max_amount = float(
                    amount_values.max()
                )

                if max_amount > 0:

                    minimum_amount = st.slider(
                        "Minimum Transaction Amount",
                        min_value=0.0,
                        max_value=max_amount,
                        value=0.0,
                    )

                    filtered_queue = filtered_queue[
                        pd.to_numeric(
                            filtered_queue["transaction_amount"],
                            errors="coerce"
                        ).fillna(0) >= minimum_amount
                    ]

            display_columns = [
                col for col in [
                    "transaction_id",
                    "transaction_date",
                    "transaction_time",
                    "transaction_amount",
                    "merchant_category",
                    "is_international",
                    "Fraud Probability (PD)",
                    "Risk Level",
                    "Recommended Action",
                    "Predicted Fraud",
                ]
                if col in filtered_queue.columns
            ]

            st.dataframe(
                filtered_queue[display_columns].head(1000),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                f"Showing {min(len(filtered_queue), 1000):,} "
                f"of {len(filtered_queue):,} matching transactions."
            )

            # CSV download
            csv_data = filtered_queue.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Fraud Investigation Queue",
                data=csv_data,
                file_name="fraud_operations_queue.csv",
                mime="text/csv",
            )

            st.divider()

            st.subheader(
                "Historical Label Comparison (Diagnostic)"
            )

            y_true = queue["is_fraud"].astype(int)

            y_pred = queue["Predicted Fraud"].astype(int)

            tp = int(
                ((y_true == 1) & (y_pred == 1)).sum()
            )

            fp = int(
                ((y_true == 0) & (y_pred == 1)).sum()
            )

            fn = int(
                ((y_true == 1) & (y_pred == 0)).sum()
            )

            tn = int(
                ((y_true == 0) & (y_pred == 0)).sum()
            )

            precision_value = (
                tp / (tp + fp) * 100
                if (tp + fp) else 0
            )

            recall_value = (
                tp / (tp + fn) * 100
                if (tp + fn) else 0
            )

            fpr_value = (
                fp / (fp + tn) * 100
                if (fp + tn) else 0
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "True Positives",
                f"{tp:,}"
            )

            m2.metric(
                "False Positives",
                f"{fp:,}"
            )

            m3.metric(
                "False Negatives",
                f"{fn:,}"
            )

            m4.metric(
                "False Positive Rate",
                f"{fpr_value:.3f}%"
            )

            st.write(
                f"**Precision:** {precision_value:.2f}%"
            )

            st.write(
                f"**Recall:** {recall_value:.2f}%"
            )

            st.warning(
                "This diagnostic compares predictions with labels "
                "from the same dataset used to display the dashboard. "
                "It is not an unbiased held-out test result. "
                "Use a separate temporal test set for final evaluation."
            )


            st.divider()

            st.subheader("Operational Recommendations")

            st.caption(
                "Proposed fraud prevention policies. "
                "These are targets, not measured achievements."
            )

            st.markdown("### 1. Dynamic Step-Up Authentication")

            st.write(
                "Low Risk (PD < 0.15): Auto Approve"
            )

            st.write(
                "Medium Risk (0.15 to 0.65): "
                "Step-Up Authentication"
            )

            st.write(
                "Use 3D Secure 2.0 or biometric "
                "authentication for medium-risk transactions."
            )

            st.info(
                "Assignment target (not yet measured): authentication completion rate above 92%."
            )

            st.markdown(
                "### 2. High-Risk Merchant Monitoring"
            )

            st.write(
                "Monitor international Electronics and Crypto "
                "Exchange transaction frequency."
            )

            st.write(
                "Proposed limit: Maximum 2 attempts per hour "
                "for selected high-risk categories."
            )

            st.info(
                "Assignment target (not yet measured): chargeback rate below 0.9%."
            )

            st.markdown(
                "### 3. Fraud Investigation Strategy"
            )

            st.write(
                "Prioritize high-risk transactions for investigation "
                "and review medium-risk transactions through "
                "additional authentication."
            )

            st.markdown(
                "### 4. Continuous Model Monitoring"
            )

            st.write(
                "Monitor Precision, Recall, PR-AUC and FPR regularly. "
                "Retrain the model when fraud patterns change."
            )

            st.warning(
                "78% fraud prevention is an assignment target, not an achieved result. "
                "not an achieved result in this project."
            )


        except Exception as exc:

            st.error(
                "Model prediction failed. Verify that the saved "
                "model, feature list, threshold and training "
                f"preprocessing match. Details: {exc}"
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Financial Fraud Detection System | Academic Project | 2026 | "
    "Local demonstration only — not connected to a live banking system."
)