import sys
import math
import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from datetime import datetime, timedelta
import base64
from pathlib import Path
import streamlit as st


def get_agri_image_path():
    candidates = [
        Path(__file__).resolve().parents[1] / "assets" / "agri.png",
        Path(__file__).resolve().parent / "assets" / "agri.png",
        Path("App/assets/agri.png"),
        Path("assets/agri.png"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

IMAGE_PATH = get_agri_image_path()

_st_markdown = st.markdown
def _dedented_markdown(body, *args, **kwargs):
    if isinstance(body, str) and kwargs.get("unsafe_allow_html"):
        body = "\n".join(line.lstrip() for line in body.split("\n"))
    return _st_markdown(body, *args, **kwargs)
st.markdown = _dedented_markdown

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import importlib
import backend.database as backend_db
importlib.reload(backend_db)
import backend.crop_recommendation as crop_rec_mod
importlib.reload(crop_rec_mod)

from backend.predict import predict, get_soil_data, city_soil, crop_kc

from backend.weather import get_weather, get_forecast, CITY_COORDS
from backend.database import (
    save_climate_risk,
    save_crop_recommendation,
    save_irrigation,
    save_yield,
    save_market_price,
    save_prediction,
    register_farmer as supabase_register_farmer,
    get_farmer as supabase_get_farmer,
    update_last_login as supabase_update_last_login,
)

try:
    from backend.database import save_feedback
except Exception:
    save_feedback = getattr(backend_db, "save_feedback", None)
save_yield = getattr(backend_db, "save_yield", save_yield)
save_market_price = getattr(backend_db, "save_market_price", save_market_price)
from datetime import datetime, timedelta

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIK  = os.path.join(BASE, "Pickles")

# ─── Load all models once ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "irrigation_model":    joblib.load(os.path.join(PIK,"Ir","best_irrigation_model.pkl")),
        "irrigation_enc":      joblib.load(os.path.join(PIK,"Ir","onehot_encoders.pkl")),
        "irrigation_cols":     joblib.load(os.path.join(PIK,"Ir","feature_columns.pkl")),
        "yield_model":         joblib.load(os.path.join(PIK,"Yield","yield_predict_model.pkl")),
        "yield_enc":           joblib.load(os.path.join(PIK,"Yield","onehot_encoders.pkl")),
        "yield_state_map":     joblib.load(os.path.join(PIK,"Yield","state_mapping.pkl")),
        "market_model":        joblib.load(os.path.join(PIK,"Market","market_price_rf_model.pkl")),
        "market_maps":         joblib.load(os.path.join(PIK,"Market","Market_Label_Mappings.pkl")),
        "climate_model":       joblib.load(os.path.join(PIK,"Climate","climate_risk_model.pkl")),
        "climate_enc":         joblib.load(os.path.join(PIK,"Climate","encoders.pkl")),
        "crop_model":          joblib.load(os.path.join(PIK,"Crop","model.pkl")),
        "crop_columns":        joblib.load(os.path.join(PIK,"Crop","feature_columns.pkl")),
        "crop_label_enc":      joblib.load(os.path.join(PIK,"Crop","label_encoder.pkl")),
        "crop_encoders":       joblib.load(os.path.join(PIK,"Crop","crop_encoders.pkl")),
    }

M = load_models()

# ─── Lookup data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_lookups():
    city_soil = pd.read_csv(os.path.join(BASE,"backend","city_soil_lookup.csv"))
    crop_kc   = pd.read_csv(os.path.join(BASE,"backend","crop_kc_lookup.csv"))
    return city_soil, crop_kc

CITY_SOIL, CROP_KC = load_lookups()

# ─── FastAPI REST API Integration ─────────────────────────────────────────────
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000")

def check_fastapi_connection():
    try:
        r = requests.get(f"{FASTAPI_BASE_URL}/health", timeout=1.5)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None

def _run_fastapi_in_thread():
    try:
        import uvicorn
        from backend_api.main import app as fastapi_app
        uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="error")
    except Exception:
        pass

@st.cache_resource
def ensure_fastapi_running():
    ok, _ = check_fastapi_connection()
    if not ok:
        import threading, time
        t = threading.Thread(target=_run_fastapi_in_thread, daemon=True)
        t.start()
        time.sleep(1.2)
    return True

ensure_fastapi_running()

def call_fastapi_api(endpoint: str, method: str = "POST", json_data: dict = None, params: dict = None):
    url = f"{FASTAPI_BASE_URL}{endpoint}"
    try:
        if method.upper() == "POST":
            resp = requests.post(url, json=json_data, timeout=10.0)
        else:
            resp = requests.get(url, params=params, timeout=10.0)
        if resp.status_code == 200:
            return True, resp.json()
        else:
            return False, f"API {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)

def api_predict_climate_risk(city: str, crop: str):
    ok, res = call_fastapi_api("/api/predict/climate-risk", method="POST", json_data={"city": city, "crop": crop})
    if ok:
        return res
    from backend.climate_risk import predict_climate_risk
    return predict_climate_risk(city=city, crop=crop, model=M["climate_model"], encoders=M["climate_enc"], city_soil_df=CITY_SOIL)

def api_recommend_crop(city: str):
    import importlib
    import backend.crop_recommendation as cr
    importlib.reload(cr)
    arts = cr.get_latest_crop_model_artifacts()
    return cr.recommend_crop(
        city=city,
        model=arts["model"],
        feature_columns=arts["feature_columns"],
        label_encoder=arts["label_encoder"],
        crop_encoders=arts["crop_encoders"],
        city_soil_df=CITY_SOIL
    )

def api_predict_irrigation(user_input: dict, weather: dict):
    farm_size_val = user_input.get("farm_size")
    farm_size_float = float(farm_size_val) if farm_size_val is not None else 2.0

    pump_lpm_val = user_input.get("pump_lpm")
    pump_lpm_float = float(pump_lpm_val) if pump_lpm_val is not None else 100.0

    ok, res = call_fastapi_api("/api/predict/irrigation", method="POST", json_data={
        "city": user_input.get("city"),
        "crop": user_input.get("crop"),
        "growth_stage": user_input.get("growth_stage") or "Development",
        "farm_size_acres": farm_size_float,
        "pump_lpm": pump_lpm_float,
        "irrigation_method": user_input.get("irrigation_method") or "Drip"
    })
    if ok:
        return res
    return predict(user_input, weather)

def api_predict_yield(district: str, state: str, season: str, crop: str, area_acres: float):
    ok, res = call_fastapi_api("/api/predict/yield", method="POST", json_data={
        "district": district, "state": state, "season": season, "crop": crop, "area_acres": float(area_acres)
    })
    if ok:
        return res
    area_ha = float(area_acres) * 0.4047
    from backend.yield_estimator import predict_yield as backend_predict_yield
    return backend_predict_yield(district=district, state=state, season=season, crop=crop, area=area_ha, model=M["yield_model"], encoder=M["yield_enc"], state_map=M["yield_state_map"], city_soil_df=CITY_SOIL)

def api_predict_market(crop: str, state: str, district: str, day: int, month: int, year: int, quarter: int, qty: float):
    ok, res = call_fastapi_api("/api/predict/market", method="POST", json_data={
        "commodity": crop, "state": state, "district": district, "day": day, "month": month, "year": year, "quarter": quarter, "arrival_qty": float(qty)
    })
    if ok:
        return float(res.get("price_per_quintal", 2000.0))
    from backend.fusion import _predict_market_price
    res_m = _predict_market_price(city=district, crop=crop, arrival_qty_quintals=qty, M=M)
    return float(res_m.get("predicted_price", 2000.0))

def api_run_fusion(city: str, farm_size_acres: float):
    ok, res = call_fastapi_api("/api/predict/fusion", method="POST", json_data={
        "city": city, "farm_size_acres": float(farm_size_acres)
    })
    if ok:
        return res
    from backend.fusion import run_fusion
    return run_fusion(city=city, farm_size_acres=farm_size_acres, city_soil_df=CITY_SOIL)

DISCLAIMER = """<div class='disclaimer'>⚠️ <strong>Disclaimer:</strong> These predictions are generated by Machine Learning models trained on historical data.
Results are estimates only and may not reflect exact real-world conditions.
Please consult your local agricultural officer before making critical farming decisions.</div>"""

# ─── Constants ────────────────────────────────────────────────────────────────
CROPS    = ["Banana","Beans","Black Gram Dal(Urd Dal)","Cotton","Grapes",
            "Karbuja(Musk Melon)","Maize","Mango","Orange","Papaya",
            "Pomegranate","Rice","Tender Coconut","Water Melon"]
CITIES   = sorted(CITY_SOIL["City"].tolist())
SEASONS  = ["Summer","Monsoon","Post-Monsoon","Winter"]
STAGES   = {"🌰 Just Planted":"Initial","🌿 Growing":"Development",
            "🌸 Flowering":"Mid-season","🌾 Almost Ready":"Late-season"}
STATES   = ["Telangana","Andhra Pradesh"]
CROP_EMOJI = {"Banana":"🍌","Beans":"🫘","Black Gram Dal(Urd Dal)":"🌰","Cotton":"☁️",
              "Grapes":"🍇","Karbuja(Musk Melon)":"🍈","Maize":"🌽","Mango":"🥭",
              "Orange":"🍊","Papaya":"🟠","Pomegranate":"🔴","Rice":"🌾",
              "Tender Coconut":"🥥","Water Melon":"🍉"}

CITY_COORDS = {
    "Adilabad":(19.66,78.53),"Anakapalli":(17.69,83.00),"Bapatla":(15.90,80.47),
    "Chittoor":(13.22,79.10),"Eluru":(16.71,81.09),"Hanumakonda":(17.99,79.59),
    "Hyderabad":(17.38,78.47),"Jagtial":(18.79,78.91),"Jangaon":(17.72,79.15),
    "Kakinada":(16.98,82.24),"Kamareddy":(18.32,78.34),"Karimnagar":(18.43,79.13),
    "Khammam":(17.25,80.15),"Kurnool":(15.83,78.04),"Mahabubabad":(17.60,80.00),
    "Mahabubnagar":(16.74,77.98),"Mancherial":(18.87,79.46),"Medak":(18.04,78.26),
    "Mulugu":(18.19,80.00),"Nagarkurnool":(16.48,78.32),"Nalgonda":(17.05,79.27),
    "Nandyal":(15.47,78.48),"Narayanpet":(16.74,77.49),"Nirmal":(19.10,78.35),
    "Srikakulam":(18.30,83.90),"Tirupati":(13.63,79.42),"Visakhapatnam":(17.68,83.22),
    "Vizianagaram":(18.10,83.40),
}

COMMODITY_MAP = {"Cotton":14,"Black Gram Dal(Urd Dal)":13,"Beans":12,"Pomegranate":11,
                 "Grapes":10,"Rice":9,"Banana":8,"Orange":7,"Mango":6,
                 "Tender Coconut":5,"Maize":4,"Papaya":3,"Karbuja(Musk Melon)":2,"Water Melon":1}
STATE_MAP_MARKET = {"Telangana":1,"Andhra Pradesh":0}
YIELD_DISTRICTS_BY_STATE = {
    "Telangana": ["Adilabad","Hyderabad","Karimnagar","Khammam","Mahbubnagar","Medak","Nalgonda","Nizamabad","Rangareddi","Warangal"],
    "Andhra Pradesh": ["Anantapur","Chittoor","East Godavari","Guntur","Kadapa","Krishna","Kurnool","Prakasam","SPSR Nellore","Srikakulam","Visakhapatnam","Vizianagaram","West Godavari"],
}
MARKET_DISTRICTS_BY_STATE = {
    "Telangana": ["Adilabad","Hyderabad","Karimnagar","Khammam","Mahbubnagar","Medak","Nalgonda","Nizamabad","Rangareddi","Warangal"],
    "Andhra Pradesh": ["Anantapur","Chittoor","East Godavari","Guntur","Kadapa","Krishna","Kurnool","Prakasam","SPSR Nellore","Srikakulam","Visakhapatnam","Vizianagaram","West Godavari"],
}
DISTRICT_MAP = {"Guntur":46,"Rajanna Siricilla":45,"Nalgonda":44,"SPSR Nellore":43,
                "Peddapalli":42,"Hanumakonda":41,"Chittor":40,"Chittoor":40,"Alluri Sitharama Raju":39,
                "Bhupalapally":38,"Mancherial":37,"Mulugu":36,"Kurnool":35,"Asifabad":34,
                "Karimnagar":33,"Khammam":32,"Mahabubabad":31,"Warangal":30,
                "Nagarkurnool":29,"Adilabad":28,"NTR":27,"Nirmal":26,
                "Bhadradri Kothagudem":25,"Suryapet":24,"Siddipet":23,
                "Vikarabad":22,"Jogulamba Gadwal":21,"East Godavari":20,"Nandyal":19,
                "Anantapur":46,"Kadapa":43,"Krishna":20,"Prakasam":46,"Srikakulam":20,
                "Visakhapatnam":20,"Visakhapatanam":20,"Vizianagaram":20,"West Godavari":20,
                "Hyderabad":28,"Mahbubnagar":29,"Medak":23,"Nizamabad":26,"Rangareddi":22}

# Helper for state-specific district selections. Keeps the dropdown aligned with the chosen state.
def _get_valid_district(state, district, mapping):
    valid_options = mapping.get(state)
    if valid_options is None:
        valid_options = next(iter(mapping.values()))
    return district if district in valid_options else valid_options[0]

KC_BINS   = [0,0.5,0.9,1.2,2.0]
KC_LABELS = ["low","medium","high","very_high"]
EFFICIENCY= {"Drip":0.90,"Sprinkler":0.75,"Flood":0.55}
ACRE_HA   = 0.4047

CROP_CLASS_MAP = {1:"Rice",2:"Maize",3:"Chickpea",4:"Kidneybeans",5:"Pigeonpeas",
                  6:"Mothbeans",7:"Mungbean",8:"Blackgram",9:"Lentil",
                  10:"Pomegranate",11:"Banana",12:"Mango",13:"Grapes",
                  14:"Watermelon",15:"Muskmelon",16:"Apple",17:"Orange",
                  18:"Papaya",19:"Coconut",20:"Cotton",21:"Jute",22:"Coffee"}

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Agri Fusion | Farmer Portal", page_icon="🌾", layout="wide")

# ─── CSS ──────────────────────────────────────────────────────────────────────
# ─── Theme (dark / light toggle) ───────────────────────────────────────────────
# Read the toggle's remembered value *before* the widget itself is drawn later
# in the sidebar — Streamlit keeps session_state across reruns, so this is safe.
DARK_MODE = st.session_state.get("dark_mode_toggle", False)

if DARK_MODE:
    T = dict(
        bg1="#060913", bg2="#0f172a", bg3="#05141c",
        sb1="rgba(6,9,19,0.95)", sb2="rgba(15,23,42,0.92)", sb3="rgba(6,9,19,0.95)",
        sidebar_text="#ffffff", sidebar_border="rgba(56,189,248,0.2)",
        card_bg="rgba(15,23,42,0.7)", card_border="rgba(56,189,248,0.28)",
        hero_bg1="rgba(15,23,42,0.9)", hero_bg2="rgba(6,78,59,0.55)",
        mcard_bg="rgba(15,23,42,0.6)",
        heading="#ffffff", body="rgba(241,245,249,0.92)", dim="#94a3b8",
        accent_text="#34d399", gold_text="#fbbf24",
        input_bg="rgba(15,23,42,0.8)", input_border="rgba(56,189,248,0.35)", input_text="#ffffff",
        hr="rgba(56,189,248,0.2)", dataframe_bg="rgba(15,23,42,0.6)",
        disclaimer_bg="rgba(245,158,11,0.18)", disclaimer_text="#fde68a",
        force_card_text="rgba(241,245,249,0.92)",
        nav_idle_bg="rgba(15,23,42,0.6)", nav_idle_text="#f8fafc",
        shadow="rgba(0,0,0,0.5)",
    )
else:
    T = dict(
        bg1="#0f172a", bg2="#1e293b", bg3="#0f2942",
        sb1="rgba(15,23,42,0.92)", sb2="rgba(30,41,59,0.88)", sb3="rgba(15,23,42,0.92)",
        sidebar_text="#ffffff", sidebar_border="rgba(56,189,248,0.25)",
        card_bg="rgba(255,255,255,0.08)", card_border="rgba(255,255,255,0.18)",
        hero_bg1="rgba(255,255,255,0.12)", hero_bg2="rgba(56,189,248,0.12)",
        mcard_bg="rgba(255,255,255,0.08)",
        heading="#ffffff", body="rgba(255,255,255,0.92)", dim="#cbd5e1",
        accent_text="#38bdf8", gold_text="#fbbf24",
        input_bg="rgba(255,255,255,0.1)", input_border="rgba(56,189,248,0.35)", input_text="#ffffff",
        hr="rgba(56,189,248,0.22)", dataframe_bg="rgba(255,255,255,0.08)",
        disclaimer_bg="rgba(251,191,36,0.18)", disclaimer_text="#fef08a",
        force_card_text="rgba(255,255,255,0.92)",
        nav_idle_bg="rgba(255,255,255,0.08)", nav_idle_text="#ffffff",
        shadow="rgba(0,0,0,0.35)",
    )

# ─── CSS ──────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
*, html, body, [class*="css"] { font-family:'Plus Jakarta Sans','Inter',sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,__SB1__ 0%,__SB2__ 50%,__SB3__ 100%) !important;
    backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);
    border-right: 1px solid __SIDEBAR_BORDER__ !important;
}
section[data-testid="stSidebar"] * { color: __SIDEBAR_TEXT__ !important; }

/* ═══ Sidebar navigation — clean translucent pills ═══ */
section[data-testid="stSidebar"] .stButton > button {
    background:__NAV_IDLE_BG__ !important;
    border:1.5px solid __CARD_BORDER__ !important;
    color:__NAV_IDLE_TEXT__ !important;
    font-weight:600 !important; font-size:14.5px !important;
    text-align:left !important; justify-content:flex-start !important;
    border-radius:14px !important; padding:12px 16px !important;
    box-shadow: 0 2px 8px __SHADOW__ !important; margin-bottom:9px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
section[data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {
    border-color:#38bdf8 !important;
    background:rgba(56,189,248,0.18) !important;
    color:#ffffff !important;
    transform:translateX(3px) !important;
}
section[data-testid="stSidebar"] .stButton > button:disabled {
    background:linear-gradient(135deg, #0ea5e9 0%, #10b981 100%) !important;
    border-color:#0284c7 !important; color:#ffffff !important;
    opacity:1 !important; cursor:default !important; font-weight:700 !important;
    box-shadow:0 6px 18px rgba(14,165,233,0.35) !important;
}
/* Logout button */
section[data-testid="stSidebar"] .stButton:last-of-type > button:not(:disabled) {
    background:linear-gradient(135deg, rgba(16,185,129,0.20), rgba(5,150,105,0.16)) !important;
    border:1.5px solid rgba(74,222,128,0.40) !important;
    color:#ecfdf5 !important; text-align:center !important; justify-content:center !important;
    box-shadow:0 4px 14px rgba(16,185,129,0.18) !important;
}

/* Main background — NEVER WHITE! */
.main {
    background: linear-gradient(135deg,__BG1__ 0%,__BG2__ 50%,__BG3__ 100%) !important;
    min-height:100vh; position:relative; overflow:hidden;
}
.block-container { background:transparent !important; padding-top:24px !important; }

/* ═══ Ambient farm animation layer ═══ */
.farm-sky { position:relative; height:0; overflow:visible; }
.cloud { position:absolute; top:-6px; font-size:26px; opacity:0.35; filter:blur(0.2px);
    animation: drift 38s linear infinite; }
.cloud.c2 { top:14px; font-size:20px; opacity:0.25; animation-duration:52s; animation-delay:-14s; }
.cloud.c3 { top:-2px; font-size:16px; opacity:0.22; animation-duration:64s; animation-delay:-30s; }
@keyframes drift { from { left:-10%; } to { left:110%; } }

.leaf { position:fixed; top:-40px; font-size:16px; opacity:0.4; pointer-events:none;
    animation: leafFall linear infinite; z-index:0; }
@keyframes leafFall {
    0%   { transform:translateY(0) rotate(0deg); opacity:0; }
    10%  { opacity:0.45; }
    100% { transform:translateY(110vh) rotate(360deg); opacity:0; }
}

.sun-badge { display:inline-block; animation: sunPulse 4s ease-in-out infinite; }
@keyframes sunPulse {
    0%,100% { transform:scale(1) rotate(0deg); filter:drop-shadow(0 0 6px rgba(251,191,36,0.4)); }
    50%     { transform:scale(1.12) rotate(8deg); filter:drop-shadow(0 0 16px rgba(251,191,36,0.6)); }
}

/* Tractor driving across the hero banner */
.tractor-track { position:relative; height:34px; margin-top:10px; overflow:hidden; }
.tractor-emoji { position:absolute; left:-15%; bottom:0; font-size:26px;
    animation: driveAcross 18s linear infinite; }
.tractor-emoji .smoke { position:absolute; left:-14px; top:-6px; font-size:11px; opacity:0.6;
    animation: puff 1.1s ease-out infinite; }
@keyframes driveAcross {
    0%   { left:-15%; transform:scaleX(1); }
    48%  { left:105%; transform:scaleX(1); }
    50%  { left:105%; transform:scaleX(-1); }
    98%  { left:-15%; transform:scaleX(-1); }
    100% { left:-15%; transform:scaleX(1); }
}
@keyframes puff { 0%{opacity:0.6; transform:translateY(0) scale(0.6);} 100%{opacity:0; transform:translateY(-14px) scale(1.3);} }

/* Water-drop celebration */
.drop-row { display:flex; gap:10px; justify-content:center; margin:6px 0 4px; }
.drop-row span { font-size:26px; display:inline-block; animation: dropBounce 1.4s ease-in-out infinite; }
.drop-row span:nth-child(2) { animation-delay:0.15s; }
.drop-row span:nth-child(3) { animation-delay:0.3s; }
.drop-row span:nth-child(4) { animation-delay:0.45s; }
.drop-row span:nth-child(5) { animation-delay:0.6s; }
@keyframes dropBounce {
    0%,100% { transform:translateY(0) scale(1); }
    35%     { transform:translateY(-10px) scale(1.12); }
    55%     { transform:translateY(0px) scale(0.94); }
}

/* Card entrance animation */
@keyframes growIn { from { opacity:0; transform:translateY(12px) scale(0.98); } to { opacity:1; transform:translateY(0) scale(1); } }
.glass, .mcard, .rcard, .hero { animation: growIn 0.45s ease-out both; }

/* ═══ Equal-size boxes ═══ */
div[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { display:flex !important; }
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div { width:100% !important; }
.glass, .mcard {
    height:100% !important; width:100% !important; box-sizing:border-box !important;
    display:flex !important; flex-direction:column !important; justify-content:center !important;
}

/* Cool Transparent Glass Card */
.glass {
    background:__CARD_BG__;
    backdrop-filter:blur(22px) saturate(180%); -webkit-backdrop-filter:blur(22px);
    border:1.5px solid __CARD_BORDER__; border-radius:20px;
    padding:24px; margin-bottom:16px;
    box-shadow: 0 12px 36px -4px __SHADOW__, inset 0 1px 0 rgba(255,255,255,0.12);
    transition:transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}
.glass:hover {
    transform:translateY(-3px);
    border-color:#38bdf8;
    box-shadow: 0 18px 44px -4px rgba(56,189,248,0.22);
}

/* Hero Banner */
.hero {
    background:linear-gradient(135deg,__HERO_BG1__,__HERO_BG2__);
    border:1.5px solid __CARD_BORDER__; border-radius:24px;
    padding:44px 40px 30px; margin-bottom:28px;
    backdrop-filter:blur(22px) saturate(180%); -webkit-backdrop-filter:blur(22px);
    box-shadow: 0 14px 38px -6px __SHADOW__, inset 0 1px 0 rgba(255,255,255,0.15);
    position:relative;
}
.hero h1 { font-family:'Plus Jakarta Sans',sans-serif; font-size:38px; font-weight:800; color:__HEADING__; margin:0 0 12px; line-height:1.2; letter-spacing:-0.5px; }
.hero p  { font-size:16.5px; color:__BODY__; margin:0; line-height:1.6; }
.welcome-line { font-size:15px; color:__GOLD_TEXT__; font-weight:700; letter-spacing:0.3px; margin-bottom:8px; text-align:center; }

/* Metric Card */
.mcard {
    background:__MCARD_BG__; border:1.5px solid __CARD_BORDER__;
    border-radius:18px; padding:20px; text-align:center;
    backdrop-filter:blur(18px); transition:transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 4px 20px __SHADOW__;
}
.mcard:hover { transform:translateY(-2px); box-shadow: 0 8px 26px rgba(56,189,248,0.2); }
.mcard .val { font-size:30px; font-weight:800;
    background:linear-gradient(135deg,#38bdf8,#34d399);
    -webkit-background-clip:text; background-clip:text; color:transparent !important; }
.mcard .lbl { font-size:12.5px; color:__DIM__; margin-top:4px; font-weight:500; }

/* Result Card */
.rcard {
    background:linear-gradient(135deg,__HERO_BG1__,__HERO_BG2__);
    border:1.5px solid __CARD_BORDER__; border-radius:20px; padding:28px;
    backdrop-filter:blur(22px);
    box-shadow: 0 14px 38px -4px __SHADOW__;
}
.rcard h2 { color:__ACCENT_TEXT__ !important; font-size:22px; margin:0 0 8px; font-weight:700; }
.rcard .big { font-size:38px; font-weight:800;
    background:linear-gradient(135deg,#38bdf8,#34d399);
    -webkit-background-clip:text; background-clip:text; color:transparent !important; }
.rcard p { color:__BODY__ !important; margin:4px 0; }

/* Force Readable Text INSIDE Cards & Boxes */
.glass, .glass *:not(.val):not(.big),
.mcard, .mcard *:not(.val):not(.big),
.rcard, .rcard *:not(.val):not(.big),
.hero, .hero *:not(.val):not(.big) { color: __FORCE_CARD_TEXT__ !important; }
.rcard h2, .step-hdr { color: __ACCENT_TEXT__ !important; }
.welcome-line { color: __GOLD_TEXT__ !important; }

/* Tag Badges */
.tag { display:inline-block; background:rgba(56,189,248,0.15);
    border:1px solid rgba(56,189,248,0.35); color:__ACCENT_TEXT__ !important;
    border-radius:20px; padding:4px 14px; font-size:12px; font-weight:600; margin:0 6px 10px 0; }

/* Disclaimer */
.disclaimer { background:__DISCLAIMER_BG__; border-left:4px solid #f59e0b;
    border-radius:12px; padding:14px 18px; color:__DISCLAIMER_TEXT__ !important; font-size:13px; margin-bottom:20px; backdrop-filter:blur(12px); }

/* Step header */
.step-hdr { color:__ACCENT_TEXT__ !important; font-size:13px; font-weight:700;
    letter-spacing:1.5px; text-transform:uppercase; margin-bottom:6px; }

/* Form Controls & Inputs — High Contrast & Readability */
.stSelectbox label, .stNumberInput label, .stRadio label,
.stSlider label, .stTextInput label { color:__BODY__ !important; font-size:14px !important; font-weight:600 !important; }
.stSelectbox > div > div, .stNumberInput > div > div > input, .stTextInput > div > div > input {
    background:__INPUT_BG__ !important;
    border:1.5px solid __INPUT_BORDER__ !important;
    border-radius:12px !important; color:__INPUT_TEXT__ !important;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.15) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.stSelectbox > div > div:hover, .stNumberInput > div > div > input:focus, .stTextInput > div > div > input:focus {
    border-color:#38bdf8 !important;
    box-shadow:0 0 0 3px rgba(56,189,248,0.25) !important;
}
/* Selectbox options popup styling */
div[data-baseweb="popover"] ul {
    background:#0f172a !important; border:1px solid rgba(56,189,248,0.3) !important; border-radius:12px !important;
}
div[data-baseweb="popover"] li {
    color:#ffffff !important; background:transparent !important;
}
div[data-baseweb="popover"] li:hover {
    background:rgba(56,189,248,0.2) !important;
}

/* Action Buttons */
.stButton > button {
    background:linear-gradient(135deg,#0ea5e9 0%,#10b981 100%) !important;
    color:#ffffff !important; font-weight:700 !important;
    border:none !important; border-radius:12px !important;
    padding:14px 28px !important; font-size:15.5px !important;
    width:100% !important; transition:transform 0.18s ease, box-shadow 0.18s ease !important;
    box-shadow: 0 6px 22px rgba(14,165,233,0.35) !important;
}
.stButton > button:hover:not(:disabled) { transform:translateY(-2px) !important; box-shadow:0 10px 28px rgba(14,165,233,0.48) !important; }
.stButton > button:disabled {
    opacity:0.6 !important; cursor:default !important;
}
h1,h2,h3,h4 { color:__HEADING__ !important; font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; }
p, li { color:__BODY__ !important; }
.stTabs [data-baseweb="tab"] { color:__DIM__ !important; font-weight:600 !important; }
.stTabs [aria-selected="true"] { color:__ACCENT_TEXT__ !important; border-bottom:2.5px solid #38bdf8 !important; }
.stDataFrame { background:__DATAFRAME_BG__ !important; border-radius:14px !important; border:1px solid __INPUT_BORDER__ !important; }
hr { border-color:__HR__ !important; }

/* Auth Screen */
.auth-wrap { max-width:460px; margin:20px auto 0; }
.auth-logo { text-align:center; font-size:48px; margin-bottom:2px; }
.auth-title { text-align:center; font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:26px; color:__HEADING__ !important; margin:6px 0 2px; }
.auth-sub { text-align:center; font-size:13px; color:__DIM__ !important; margin-bottom:18px; }
.field-error { color:#ef4444; font-weight:600; font-size:12.5px; margin-top:4px; }

/* Native Streamlit Bordered Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:linear-gradient(160deg,__HERO_BG1__,__HERO_BG2__) !important;
    border:1.5px solid __CARD_BORDER__ !important;
    border-radius:22px !important;
    backdrop-filter:blur(22px) saturate(180%);
    box-shadow:0 14px 38px __SHADOW__;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div { padding:6px 4px; }
div[data-testid="stVerticalBlockBorderWrapper"] * { color: __FORCE_CARD_TEXT__ !important; }
div[data-testid="stVerticalBlockBorderWrapper"] .auth-title,
div[data-testid="stVerticalBlockBorderWrapper"] h1,
div[data-testid="stVerticalBlockBorderWrapper"] h2,
div[data-testid="stVerticalBlockBorderWrapper"] h3 { color: __HEADING__ !important; }

/* Quick-switch chip row */
.crumb { font-size:12.5px; color:__DIM__ !important; margin-bottom:14px; }
.crumb b { color:__GOLD_TEXT__ !important; }

/* Outlined Prediction Switcher Buttons */
.pred-switcher-row .stButton > button {
    background: rgba(15, 23, 42, 0.5) !important;
    border: 1.5px solid #38bdf8 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    border-radius: 12px !important;
    padding: 10px 12px !important;
    box-shadow: 0 4px 12px rgba(56, 189, 248, 0.15) !important;
    transition: all 0.2s ease !important;
}
.pred-switcher-row .stButton > button:hover:not(:disabled) {
    background: rgba(56, 189, 248, 0.2) !important;
    border-color: #34d399 !important;
    color: #ffffff !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(56, 189, 248, 0.3) !important;
}
.pred-switcher-row .stButton > button:disabled {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.35), rgba(16, 185, 129, 0.35)) !important;
    border: 2px solid #38bdf8 !important;
    color: #ffffff !important;
    opacity: 1 !important;
    font-weight: 800 !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.45) !important;
}
</style>
"""
for _k, _v in T.items():
    _CSS = _CSS.replace(f"__{_k.upper()}__", _v)
_CSS = _CSS.replace("__MCARD_BG__", T["mcard_bg"])
st.markdown(_CSS, unsafe_allow_html=True)

# Additional styling for the Home (agriculture) section — uses theme variables from T
st.markdown("""
<style>
/* Home hero layout */
/* ═══════════════════════════════════════════════════════════════════════════
   AGRICULTURE HOME PAGE
   ═══════════════════════════════════════════════════════════════════════════ */

/* ──────────────────────────────────────────────────────────────────────────
   HERO
   ────────────────────────────────────────────────────────────────────────── */

.agri-home-hero {
    display:grid;
    grid-template-columns:1.05fr 0.95fr;
    gap:28px;
    align-items:stretch;
    margin-bottom:34px;
    animation:growIn 0.6s ease-out both;
}

.agri-hero-single-card {
    padding:38px 36px 32px;
    border-radius:26px;
    background:
        linear-gradient(
            135deg,
            rgba(16,185,129,0.13),
            rgba(14,165,233,0.10),
            rgba(255,255,255,0.04)
        );
    border:1px solid rgba(255,255,255,0.13);
    backdrop-filter:blur(22px);
    box-shadow:
        0 18px 50px rgba(0,0,0,0.20),
        inset 0 1px 0 rgba(255,255,255,0.10);
    margin-bottom:34px;
    animation:growIn 0.6s ease-out both;
}

.agri-kicker {
    display:inline-block;
    font-size:11px;
    font-weight:800;
    letter-spacing:1.5px;
    color:#43e97b;
    margin-bottom:18px;
}

.agri-hero-left h1 {
    font-size:42px !important;
    line-height:1.12 !important;
    margin:0 0 18px !important;
    letter-spacing:-1px;
}

.agri-hero-left h1 span {
    background:linear-gradient(90deg,#43e97b,#38bdf8);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
}

.agri-hero-text {
    font-size:15px;
    line-height:1.8;
    color:rgba(255,255,255,0.72);
    margin-bottom:22px;
    max-width:680px;
}

.agri-hero-highlight {
    display:flex;
    align-items:flex-start;
    gap:13px;
    padding:15px 17px;
    border-radius:16px;
    background:rgba(67,233,123,0.08);
    border:1px solid rgba(67,233,123,0.18);
}

.agri-hero-highlight > span {
    font-size:27px;
}

.agri-hero-highlight b {
    display:block;
    color:#ffffff;
    font-size:13px;
    margin-bottom:5px;
}

.agri-hero-highlight small {
    display:block;
    color:rgba(255,255,255,0.58);
    font-size:11.5px;
    line-height:1.6;
}


/* ──────────────────────────────────────────────────────────────────────────
   HERO IMAGE
   ────────────────────────────────────────────────────────────────────────── */

.agri-hero-right {
    min-height:420px;
}

.agri-hero-right-card {
    padding:30px 24px;
    border-radius:26px;
    height:100%;
    box-sizing:border-box;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    background:
        linear-gradient(
            135deg,
            rgba(16,185,129,0.13),
            rgba(14,165,233,0.10),
            rgba(255,255,255,0.04)
        );
    border:1px solid rgba(255,255,255,0.13);
    backdrop-filter:blur(22px);
    box-shadow:
        0 18px 50px rgba(0,0,0,0.20),
        inset 0 1px 0 rgba(255,255,255,0.10);
}

.agri-image-frame {
    height:100%;
    min-height:420px;
    position:relative;
    overflow:hidden;
    border-radius:26px;
    border:1px solid rgba(255,255,255,0.16);
    box-shadow:
        0 20px 55px rgba(0,0,0,0.28),
        inset 0 1px 0 rgba(255,255,255,0.15);
    background:
        linear-gradient(
            135deg,
            rgba(16,185,129,0.20),
            rgba(14,165,233,0.15)
        );
}

.agri-main-image {
    width:100%;
    height:100%;
    min-height:420px;
    object-fit:cover;
    display:block;
    transition:transform 0.7s ease;
}

.agri-image-frame:hover .agri-main-image {
    transform:scale(1.04);
}

.agri-image-frame::after {
    content:"";
    position:absolute;
    inset:0;
    background:
        linear-gradient(
            180deg,
            rgba(0,0,0,0.02) 35%,
            rgba(0,0,0,0.55) 100%
        );
    pointer-events:none;
}

.image-overlay {
    position:absolute;
    left:18px;
    right:18px;
    bottom:18px;
    display:flex;
    flex-wrap:wrap;
    gap:7px;
    z-index:2;
}

.overlay-pill {
    padding:7px 11px;
    border-radius:999px;
    background:rgba(0,0,0,0.45);
    border:1px solid rgba(255,255,255,0.20);
    backdrop-filter:blur(10px);
    color:#ffffff;
    font-size:11px;
    font-weight:600;
}


/* ──────────────────────────────────────────────────────────────────────────
   SECTION TITLES
   ────────────────────────────────────────────────────────────────────────── */

.agri-section-title {
    display:flex;
    align-items:center;
    gap:14px;
    margin:28px 0 18px;
}

.agri-section-title > span {
    display:flex;
    align-items:center;
    justify-content:center;
    width:46px;
    height:46px;
    border-radius:14px;
    background:rgba(67,233,123,0.10);
    border:1px solid rgba(67,233,123,0.18);
    font-size:23px;
}

.agri-section-title h2 {
    margin:0 !important;
    font-size:23px !important;
}

.agri-section-title p {
    margin:4px 0 0;
    font-size:12.5px;
    color:rgba(255,255,255,0.50);
}


/* ──────────────────────────────────────────────────────────────────────────
   DOMAIN CARDS
   ────────────────────────────────────────────────────────────────────────── */

.agri-domain-card {
    min-height:205px;
    padding:22px;
    border-radius:20px;
    background:rgba(255,255,255,0.045);
    border:1px solid rgba(255,255,255,0.11);
    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        background 0.25s ease;
}

.agri-domain-card:hover {
    transform:translateY(-5px);
    border-color:rgba(67,233,123,0.40);
    background:rgba(67,233,123,0.07);
}

.domain-icon {
    font-size:31px;
    margin-bottom:14px;
}

.domain-title {
    font-size:15px;
    font-weight:750;
    color:#43e97b;
    margin-bottom:9px;
}

.domain-text {
    font-size:12.5px;
    line-height:1.7;
    color:rgba(255,255,255,0.64);
}


/* ──────────────────────────────────────────────────────────────────────────
   FARMING CYCLE
   ────────────────────────────────────────────────────────────────────────── */

.agri-cycle {
    display:grid;
    grid-template-columns:repeat(6,1fr);
    gap:10px;
}

.cycle-item {
    position:relative;
    min-height:155px;
    padding:17px 12px;
    text-align:center;
    border-radius:17px;
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.09);
    transition:all 0.22s ease;
}

.cycle-item:hover {
    transform:translateY(-4px);
    border-color:rgba(56,189,248,0.35);
    background:rgba(56,189,248,0.06);
}

.cycle-number {
    position:absolute;
    top:8px;
    right:9px;
    font-size:9px;
    color:rgba(255,255,255,0.25);
    font-weight:700;
}

.cycle-icon {
    font-size:27px;
    margin-top:7px;
    margin-bottom:9px;
}

.cycle-title {
    color:#ffffff;
    font-size:11.5px;
    font-weight:700;
    margin-bottom:5px;
}

.cycle-description {
    color:rgba(255,255,255,0.47);
    font-size:9.5px;
    line-height:1.45;
}


/* ──────────────────────────────────────────────────────────────────────────
   PROBLEM CARDS
   ────────────────────────────────────────────────────────────────────────── */

.agri-problem-card {
    min-height:185px;
    padding:21px;
    margin-bottom:16px;
    border-radius:19px;
    background:rgba(255,255,255,0.045);
    border:1px solid rgba(255,255,255,0.10);
    transition:all 0.25s ease;
}

.agri-problem-card:hover {
    transform:translateY(-4px);
    border-color:rgba(251,191,36,0.30);
    background:rgba(251,191,36,0.045);
}

.problem-icon {
    font-size:29px;
    margin-bottom:12px;
}

.problem-title {
    color:#fbbf24;
    font-size:14px;
    font-weight:750;
    margin-bottom:8px;
}

.problem-text {
    color:rgba(255,255,255,0.62);
    font-size:12px;
    line-height:1.65;
}


/* ──────────────────────────────────────────────────────────────────────────
   FARMER DECISIONS
   ────────────────────────────────────────────────────────────────────────── */

.agri-decision-card {
    min-height:180px;
    padding:21px 17px;
    text-align:center;
    border-radius:19px;
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.055),
            rgba(255,255,255,0.025)
        );
    border:1px solid rgba(255,255,255,0.10);
    transition:all 0.25s ease;
}

.agri-decision-card:hover {
    transform:translateY(-5px);
    border-color:rgba(67,233,123,0.30);
}

.decision-icon {
    font-size:31px;
    margin-bottom:12px;
}

.decision-title {
    color:#ffffff;
    font-size:13px;
    font-weight:750;
    margin-bottom:8px;
}

.decision-text {
    color:rgba(255,255,255,0.55);
    font-size:11px;
    line-height:1.6;
}


/* ──────────────────────────────────────────────────────────────────────────
   DATA + AGRICULTURE
   ────────────────────────────────────────────────────────────────────────── */

.agri-intelligence {
    display:grid;
    grid-template-columns:1fr 1.15fr;
    gap:28px;
    padding:30px;
    border-radius:24px;
    background:
        linear-gradient(
            135deg,
            rgba(16,185,129,0.09),
            rgba(14,165,233,0.07),
            rgba(255,255,255,0.035)
        );
    border:1px solid rgba(255,255,255,0.12);
}

.intelligence-label {
    font-size:10px;
    font-weight:800;
    letter-spacing:1.3px;
    color:#43e97b;
    margin-bottom:13px;
}

.intelligence-left h2 {
    font-size:27px !important;
    line-height:1.25 !important;
    margin:0 0 14px !important;
}

.intelligence-left h2 span {
    color:#38bdf8;
}

.intelligence-left p {
    font-size:13px;
    line-height:1.8;
    color:rgba(255,255,255,0.62);
}

.intelligence-right {
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:11px;
}

.intelligence-node {
    padding:17px;
    border-radius:16px;
    background:rgba(0,0,0,0.16);
    border:1px solid rgba(255,255,255,0.09);
}

.intelligence-node span {
    display:block;
    font-size:25px;
    margin-bottom:7px;
}

.intelligence-node b {
    display:block;
    color:#ffffff;
    font-size:12.5px;
    margin-bottom:4px;
}

.intelligence-node small {
    color:rgba(255,255,255,0.45);
    font-size:9.5px;
    line-height:1.4;
}


/* ──────────────────────────────────────────────────────────────────────────
   FINAL MESSAGE
   ────────────────────────────────────────────────────────────────────────── */

.agri-final {
    display:flex;
    align-items:center;
    gap:18px;
    padding:22px 25px;
    border-radius:20px;
    background:
        linear-gradient(
            90deg,
            rgba(67,233,123,0.10),
            rgba(56,189,248,0.06)
        );
    border:1px solid rgba(67,233,123,0.18);
    margin-bottom:20px;
}

.final-icon {
    font-size:38px;
}

.final-title {
    font-size:15px;
    font-weight:750;
    color:#ffffff;
    margin-bottom:5px;
}

.final-text {
    font-size:12px;
    line-height:1.65;
    color:rgba(255,255,255,0.58);
}


/* ──────────────────────────────────────────────────────────────────────────
   RESPONSIVE
   ────────────────────────────────────────────────────────────────────────── */

@media (max-width: 1100px) {

    .agri-home-hero {
        grid-template-columns:1fr;
    }

    .agri-hero-right,
    .agri-image-frame,
    .agri-main-image {
        min-height:330px;
    }

    .agri-cycle {
        grid-template-columns:repeat(4,1fr);
    }

    .agri-intelligence {
        grid-template-columns:1fr;
    }
}

@media (max-width: 750px) {

    .agri-hero-left {
        padding:30px 22px;
    }

    .agri-hero-left h1 {
        font-size:32px !important;
    }

    .agri-cycle {
        grid-template-columns:repeat(2,1fr);
    }

    .intelligence-right {
        grid-template-columns:1fr;
    }

    .agri-final {
        align-items:flex-start;
    }
}
</style>
""", unsafe_allow_html=True)


# ─── Decorative ambient animation (a few drifting leaves + clouds) ────────────
st.markdown("""
<div class="farm-sky">
  <span class="cloud">☁️</span><span class="cloud c2">☁️</span><span class="cloud c3">☁️</span>
</div>
<span class="leaf" style="left:6%;  animation-duration:13s; animation-delay:0s;">🍃</span>
<span class="leaf" style="left:34%; animation-duration:16s; animation-delay:3s;">🌿</span>
<span class="leaf" style="left:62%; animation-duration:14s; animation-delay:6s;">🍃</span>
<span class="leaf" style="left:85%; animation-duration:18s; animation-delay:1.5s;">🌾</span>
""", unsafe_allow_html=True)

# ─── Farmer Auth (sign up / sign in) ───────────────────────────────────────────
import hashlib, hmac, re, json

USERS_FILE = os.path.join(BASE, "farmer_accounts.json")

def _load_users() -> dict:
    """Load registered farmer accounts from disk. Never raises — returns {} on any problem."""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}

def _save_users(users: dict) -> bool:
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as fh:
            json.dump(users, fh, indent=2)
        return True
    except OSError:
        return False

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000).hex()

def _normalize_phone(phone: str) -> str:
    return re.sub(r"\D", "", phone or "")

def _verify_password(password: str, record: dict | None) -> bool:
    if not isinstance(record, dict):
        return False

    if "hash" in record and "salt" in record:
        expected = _hash_password(password, record.get("salt", ""))
        return hmac.compare_digest(expected, record.get("hash", ""))

    password_hash = record.get("password_hash") or record.get("hash") or ""
    password_salt = record.get("password_salt") or record.get("salt") or ""
    if not password_hash:
        return False
    if ":" in password_hash and password_salt in (None, ""):
        salt, expected_hash = password_hash.split(":", 1)
        return hmac.compare_digest(_hash_password(password, salt), expected_hash)
    if not password_salt:
        return False
    expected = _hash_password(password, password_salt)
    return hmac.compare_digest(expected, password_hash)


def register_farmer(name: str, phone: str, password: str):
    """Returns (success: bool, message: str)."""
    name = (name or "").strip()
    phone_clean = _normalize_phone(phone)

    if len(name) < 2:
        return False, "Please enter your full name."
    if not re.match(r"^[A-Za-z ,.'-]+$", name):
        return False, "Name should only contain letters and spaces."
    if len(phone_clean) != 10:
        return False, "Enter a valid 10-digit mobile number."
    if len(password or "") < 4:
        return False, "Password should be at least 4 characters."

    users = _load_users()
    if phone_clean in users:
        return False, "An account with this phone number already exists. Please sign in."

    try:
        remote_record = supabase_get_farmer(phone_clean)
        if isinstance(remote_record, dict):
            return False, "An account with this phone number already exists. Please sign in."
    except Exception:
        pass

    salt = os.urandom(16).hex()
    password_hash = _hash_password(password, salt)
    users[phone_clean] = {
        "name": name,
        "phone": phone_clean,
        "salt": salt,
        "hash": password_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if not _save_users(users):
        return False, "Could not save your account right now. Please try again."

    try:
        supabase_register_farmer(name, phone_clean, f"{salt}:{password_hash}")
    except Exception:
        pass
    return True, "Account created! You can now sign in."


def authenticate_farmer(phone: str, password: str):
    """Returns (success: bool, name_or_message: str)."""
    phone_clean = _normalize_phone(phone)
    if len(phone_clean) != 10:
        return False, "Enter a valid 10-digit mobile number."
    if not password:
        return False, "Please enter your password."

    users = _load_users()
    record = users.get(phone_clean)
    if record and _verify_password(password, record):
        try:
            supabase_update_last_login(phone_clean)
        except Exception:
            pass
        return True, record.get("name", "Farmer")

    try:
        remote_record = supabase_get_farmer(phone_clean)
        if isinstance(remote_record, dict) and _verify_password(password, remote_record):
            users[phone_clean] = {
                "name": remote_record.get("full_name") or remote_record.get("name") or "Farmer",
                "phone": phone_clean,
                "salt": remote_record.get("password_salt") or remote_record.get("salt") or "",
                "hash": remote_record.get("password_hash") or remote_record.get("hash") or "",
                "created_at": remote_record.get("created_at") or datetime.now().isoformat(timespec="seconds"),
            }
            _save_users(users)
            try:
                supabase_update_last_login(phone_clean)
            except Exception:
                pass
            return True, users[phone_clean]["name"]
    except Exception:
        pass

    if not record:
        return False, "No account found for this number. Please sign up first."
    return False, "Incorrect password. Please try again."

def render_auth_screen():
    """Renders the sign-in / sign-up experience and halts the script until authenticated."""
    st.markdown("""
    <div class="hero" style="max-width:640px; margin:10px auto 20px; text-align:center; padding:36px 32px 22px;">
        <span class="sun-badge" style="font-size:44px;">☀️</span>
        <h1 style="font-size:32px;">Agri Fusion — Farmer Portal</h1>
        <p>Sign in with your phone number to reach your AI-powered farming dashboard.</p>
        <div class="tractor-track">
            <div class="tractor-emoji">🚜<span class="smoke">💨</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        tab_login, tab_signup = st.tabs(["🔐  Sign In", "📝  Sign Up"])

        with tab_login:
            with st.container(border=True):
                st.markdown("""
                <div class="auth-logo">🌾</div>
                <div class="auth-title">Welcome Back, Farmer</div>
                <div class="auth-sub">Sign in to continue to your dashboard</div>
                """, unsafe_allow_html=True)
                with st.form("login_form", clear_on_submit=False):
                    login_phone = st.text_input("📱 Phone Number", placeholder="98765 43210", max_chars=10)
                    login_pass  = st.text_input("🔒 Password", type="password", placeholder="Your password")
                    login_btn   = st.form_submit_button("🚜 Sign In", use_container_width=True)
                if login_btn:
                    ok, result = authenticate_farmer(login_phone, login_pass)
                    if ok:
                        st.session_state.authenticated = True
                        st.session_state.farmer_name   = result
                        st.session_state.farmer_phone  = _normalize_phone(login_phone)
                        st.success(f"Welcome back, {result}! Taking you to your dashboard… 🌱")
                        st.rerun()
                    else:
                        st.markdown(f'<div class="field-error">⚠️ {result}</div>', unsafe_allow_html=True)

        with tab_signup:
            with st.container(border=True):
                st.markdown("""
                <div class="auth-logo">🧑\u200d🌾</div>
                <div class="auth-title">Join Agri Fusion</div>
                <div class="auth-sub">Create your free farmer account</div>
                """, unsafe_allow_html=True)
                with st.form("signup_form", clear_on_submit=False):
                    su_name  = st.text_input("👨‍🌾 Farmer Name", placeholder="e.g. Ramesh Kumar")
                    su_phone = st.text_input("📱 Phone Number", placeholder="98765 43210", max_chars=10)
                    su_pass  = st.text_input("🔒 Create Password", type="password", placeholder="At least 4 characters")
                    su_pass2 = st.text_input("🔒 Confirm Password", type="password", placeholder="Re-enter password")
                    su_btn   = st.form_submit_button("🌱 Create My Account", use_container_width=True)
                if su_btn:
                    if su_pass != su_pass2:
                        st.markdown('<div class="field-error">⚠️ Passwords do not match.</div>', unsafe_allow_html=True)
                    else:
                        ok, msg = register_farmer(su_name, su_phone, su_pass)
                        if ok:
                            st.success(f"🎉 {msg}")
                            st.balloons()
                        else:
                            st.markdown(f'<div class="field-error">⚠️ {msg}</div>', unsafe_allow_html=True)

# ─── Auth gate: block the rest of the app until signed in ─────────────────────
# Sync with st.query_params so browser refresh (F5) keeps user logged in on current page
q_auth  = st.query_params.get("auth")
q_phone = st.query_params.get("phone")
q_name  = st.query_params.get("name")

if "authenticated" not in st.session_state:
    if q_auth == "1" and q_phone:
        st.session_state.authenticated = True
        st.session_state.farmer_phone = q_phone
        st.session_state.farmer_name  = q_name or "Farmer"
    else:
        st.session_state.authenticated = False

if "farmer_name" not in st.session_state or not st.session_state.farmer_name:
    st.session_state.farmer_name = q_name or ""
if "farmer_phone" not in st.session_state or not st.session_state.farmer_phone:
    st.session_state.farmer_phone = q_phone or ""

if not st.session_state.authenticated:
    render_auth_screen()
    st.stop()

# Ensure query params stay synced for page refresh (F5)
if st.session_state.farmer_phone:
    st.query_params["auth"]  = "1"
    st.query_params["phone"] = st.session_state.farmer_phone
    if st.session_state.farmer_name:
        st.query_params["name"]  = st.session_state.farmer_name

FARMER_NAME = st.session_state.farmer_name or "Farmer"

# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_weather(lat, lon):
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude":lat,"longitude":lon,"timezone":"auto",
            "current":"temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation",
            "hourly":"et0_fao_evapotranspiration,precipitation",
        }, timeout=10)
        d = r.json()
        c = d.get("current", {}) or {}
        h = d.get("hourly", {}) or {}
        et0_arr = h.get("et0_fao_evapotranspiration") or [3.5]
        rain_arr = h.get("precipitation") or [0.0]
        return {
            "temperature":       float(c.get("temperature_2m")) if c.get("temperature_2m") is not None else 27.5,
            "relative_humidity": float(c.get("relative_humidity_2m")) if c.get("relative_humidity_2m") is not None else 65.0,
            "wind_speed":        float(c.get("wind_speed_10m")) if c.get("wind_speed_10m") is not None else 9.5,
            "solar_radiation":   float(c.get("shortwave_radiation")) if c.get("shortwave_radiation") is not None else 550.0,
            "et0":               float(et0_arr[0]) if et0_arr and et0_arr[0] is not None else 3.5,
            "rainfall":          float(rain_arr[0]) if rain_arr and rain_arr[0] is not None else 0.0,
        }
    except Exception:
        return {"temperature": 27.5, "relative_humidity": 65.0, "wind_speed": 9.5, "solar_radiation": 550.0, "et0": 3.5, "rainfall": 0.0}

def get_forecast(lat, lon, days=7):
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude":lat,"longitude":lon,"timezone":"auto","forecast_days":days,
        "daily":"temperature_2m_max,precipitation_sum,et0_fao_evapotranspiration,shortwave_radiation_sum",
        "hourly":"relative_humidity_2m,wind_speed_10m",
    }, timeout=10)
    d = r.json()["daily"]; h = r.json()["hourly"]
    return [{"date":d["time"][i],"temperature":d["temperature_2m_max"][i],
             "rainfall":d["precipitation_sum"][i],"et0":d["et0_fao_evapotranspiration"][i],
             "solar_radiation":(d["shortwave_radiation_sum"][i] or 0)*1000,
             "relative_humidity":h["relative_humidity_2m"][i*24],
             "wind_speed":h["wind_speed_10m"][i*24]} for i in range(len(d["time"]))]

def predict_irrigation(city, crop, growth_stage, farm_size, farm_unit, method, pump_lpm, weather):
    STAGE_MAP = {"Initial":1,"Development":2,"Mid-season":3,"Late-season":4}
    SRISK     = {"Low":0,"Medium":1}
    SMAP      = {"Telangana":0,"Andhra Pradesh":1}
    soil = CITY_SOIL[CITY_SOIL["City"]==city].iloc[0].to_dict()
    kc_r = CROP_KC[(CROP_KC["Crop"]==crop)&(CROP_KC["Growth_Stage"]==growth_stage)].iloc[0]
    row  = {"state":SMAP[soil["State"]],"temperature":weather["temperature"],
            "relative_humidity":weather["relative_humidity"],"rainfall":weather["rainfall"],
            "wind_speed":weather["wind_speed"],"solar_radiation":weather["solar_radiation"],
            "et0":weather["et0"],"climate_risk_score":soil["Climate_Risk_Score"],
            "climate_risk":SRISK[soil["Climate_Risk"]],"soil_ph":soil["Soil_pH"],
            "organic_carbon":soil["Organic_Carbon"],"sand_percentage":soil["Sand_Percentage"],
            "silt_percentage":soil["Silt_Percentage"],"clay_percentage":soil["Clay_Percentage"],
            "cec":soil["CEC"],"bulk_density":soil["Bulk_Density"],"field_capacity":soil["Field_Capacity"],
            "wilting_point":soil["Wilting_Point"],"available_water":soil["Available_Water"],
            "nitrogen":soil["Nitrogen"],"soil_type":soil["Soil_Type"],"crop":crop,
            "growth_stage":STAGE_MAP[growth_stage],"root_depth_m":float(kc_r["Root_Depth_m"]),"kc":float(kc_r["Kc"])}
    df = pd.DataFrame([row])
    df["kc_band"] = pd.cut(df["kc"],bins=KC_BINS,labels=KC_LABELS)
    df = df.drop(columns=["kc"])
    for col,enc in M["irrigation_enc"].items():
        enc_d = enc.transform(df[[col]])
        enc_df= pd.DataFrame(enc_d,columns=enc.get_feature_names_out([col]),index=df.index)
        df = pd.concat([df.drop(columns=[col]),enc_df],axis=1)
    df = df.reindex(columns=M["irrigation_cols"],fill_value=0)
    mm   = float(M["irrigation_model"].predict(df)[0])
    ha   = farm_size*ACRE_HA if farm_unit=="Acres" else farm_size
    lit  = mm*ha*10000
    eff  = EFFICIENCY.get(method,1.0) if method else 1.0
    adj  = lit/eff
    mins = round(adj/pump_lpm) if pump_lpm else None
    days_next = 1 if mm>10 else (2 if mm>=5 else 3)
    return {"mm":round(mm,2),"liters":round(adj,0),"motor_mins":mins,
            "irrigate":mm>1.5,"next_days":days_next}

def predict_climate_risk(city, state, crop, season, temperature, humidity, precipitation, wind_speed):
    enc = M["climate_enc"]
    row = pd.DataFrame([{"city":city.lower(),"state":state.lower(),
                          "Crop":crop.lower(),"Season":season.lower()}])
    cats = enc["city"].transform(row[["city"]])
    cats_df = pd.DataFrame(cats, columns=enc["city"].get_feature_names_out(["city"]))
    for col in ["state","Crop","Season"]:
        e = enc[col]
        t = e.transform(row[[col]])
        d = pd.DataFrame(t,columns=e.get_feature_names_out([col]))
        cats_df = pd.concat([cats_df,d],axis=1)
    nums = pd.DataFrame([{"temperature_2m":temperature,"relative_humidity_2m":humidity,
                           "precipitation":precipitation,"wind_speed_10m":wind_speed}])
    X = pd.concat([nums.reset_index(drop=True),cats_df.reset_index(drop=True)],axis=1)
    X = X.reindex(columns=[f for f in range(M["climate_model"].n_features_in_)],fill_value=0)
    # use raw feature count
    X_raw = pd.concat([nums,cats_df],axis=1)
    X_raw = X_raw.reindex(columns=list(range(M["climate_model"].n_features_in_)),fill_value=0)
    pred = M["climate_model"].predict(X_raw if X_raw.shape[1]==M["climate_model"].n_features_in_ else X)[0]
    return pred

def predict_market_price(commodity, state, district, day, month, year, quarter, arrival_qty):
    row = pd.DataFrame([{"Commodity":COMMODITY_MAP.get(commodity,1),
                          "State":STATE_MAP_MARKET.get(state,0),
                          "District":DISTRICT_MAP.get(district,1),
                          "Day":day,"Month":month,"Year":year,
                          "Quarter":quarter,"Arrival_Quantity":arrival_qty}])
    return round(float(M["market_model"].predict(row)[0]),2)



# ─── Prediction models ─────────────────────────────────────────────────────
PREDICTION_MODELS = [
    ("🌾  Crop Recommendation",   "🌾", "Crop Recommendation",  "Best crop for your soil & season"),
    ("🌡️  Climate Risk",         "🌡️", "Climate Risk",         "Is today safe for field work?"),
    ("💧  Irrigation Advisor",    "💧", "Irrigation Advisor",   "How much water today?"),
    ("📈  Yield Estimator",       "📈", "Yield Estimator",      "What harvest to expect"),
    ("💰  Market Price",          "💰", "Market Price",         "Today's mandi price"),
]
PRED_LABELS = [m[0] for m in PREDICTION_MODELS]
if "pred_choice" not in st.session_state or st.session_state.pred_choice not in PRED_LABELS:
    st.session_state.pred_choice = PRED_LABELS[0]


def render_prediction_breadcrumb(current_label: str):
    """Bordered chip row at the top of a prediction page so a farmer can
    tap between models without going through a hub page."""
    st.markdown(
        "<div class='crumb'>🔮 <b>Predictions</b> — tap a model to switch, "
        "or run <b>Agri Fusion</b> from the sidebar to get all 5 at once.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='pred-switcher-row'>", unsafe_allow_html=True)
    cols = st.columns(len(PREDICTION_MODELS))
    for col, (label, icon, name, desc) in zip(cols, PREDICTION_MODELS):
        active = (label == current_label)
        with col:
            if st.button(f"{icon}  {name}", key=f"chip_{name}",
                         use_container_width=True, disabled=active,
                         help=desc):
                st.session_state.pred_choice = label
                st.query_params["pred"] = label
                st.rerun()
    st.markdown("</div><div style='height:8px;'></div>", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:10px; padding:16px 2px 2px;'>
        <div style='font-size:30px;'>🌾</div>
        <div>
            <div style='font-size:19px; font-weight:800; color:#fff; line-height:1.15;'>Agri Fusion</div>
            <div style='font-size:10.5px; color:rgba(255,255,255,0.4); margin-top:1px;'>AI Farming Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:rgba(124,179,66,0.12); border:1px solid rgba(124,179,66,0.3);
                border-radius:14px; padding:9px 14px; margin:14px 0 16px; text-align:center;'>
        <div style='font-size:13px; color:rgba(255,255,255,0.55);'>Welcome back,</div>
        <div style='font-size:15px; font-weight:700; color:#fff;'>🧑‍🌾 {FARMER_NAME}</div>
        <div class="tractor-track" style="height:20px; margin-top:4px;">
            <div class="tractor-emoji" style="font-size:16px;">🚜<span class="smoke">💨</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_labels = [
        "🏠  Home",
        "📊  Project Overview",
        "🔮  Predictions",
        "🚀  Agri Fusion (All-in-One)",
        "💬  Feedback",
    ]
    q_page = st.query_params.get("page")
    q_pred = st.query_params.get("pred")

    if "sidebar_nav" not in st.session_state:
        if q_page and q_page in nav_labels:
            st.session_state.sidebar_nav = q_page
        else:
            st.session_state.sidebar_nav = nav_labels[2]

    if "pred_choice" not in st.session_state or st.session_state.pred_choice not in PRED_LABELS:
        if q_pred and q_pred in PRED_LABELS:
            st.session_state.pred_choice = q_pred
        else:
            st.session_state.pred_choice = PRED_LABELS[0]

    selected_nav = st.session_state.sidebar_nav
    for label in nav_labels:
        if st.button(label, key=f"nav_{label}", use_container_width=True, disabled=(label == selected_nav)):
            st.session_state.sidebar_nav = label
            st.query_params["page"] = label
            if label == "🔮  Predictions":
                st.session_state.pred_choice = PRED_LABELS[0]
                st.query_params["pred"] = PRED_LABELS[0]
            st.rerun()

    if selected_nav == "🔮  Predictions":
        page = st.session_state.pred_choice
    else:
        page = selected_nav

    st.query_params["page"] = selected_nav
    if selected_nav == "🔮  Predictions":
        st.query_params["pred"] = st.session_state.pred_choice

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # ── FastAPI Connection Status Box (Placed at Bottom of Sidebar) ─────────────
    is_fastapi_up, fastapi_info = check_fastapi_connection()
    if is_fastapi_up:
        st.markdown("""
        <div style='background:rgba(46,125,50,0.2); border:1px solid rgba(76,175,80,0.5);
                    border-radius:12px; padding:10px 14px; margin-bottom:14px; font-size:11.5px; color:#81c784;'>
            🟢 <b>FastAPI Backend Connected</b><br>
            <span style='color:rgba(255,255,255,0.7); font-size:10.5px;'>REST API: http://127.0.0.1:8000</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(255,152,0,0.15); border:1px solid rgba(255,152,0,0.35);
                    border-radius:12px; padding:10px 14px; margin-bottom:14px; font-size:11.5px; color:#ffb74d;'>
            ⚡ <b>Backend: Direct Python Fallback</b><br>
            <span style='color:rgba(255,255,255,0.7); font-size:10.5px;'>FastAPI Server Offline</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🚪 Logout", key="sidebar_logout", use_container_width=True):
        st.query_params.clear()
        st.session_state.authenticated = False
        st.session_state.farmer_name = ""
        st.session_state.farmer_phone = ""
        st.rerun()
    st.markdown("<div style='font-size:11px; color:rgba(255,255,255,0.3); text-align:center; margin-top:10px;'>Kammara Sneha | 2026<br>Data Science Intern @ Vajra.ai</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME — AGRICULTURE DOMAIN
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠  Home":

    # ──────────────────────────────────────────────────────────────────────────
    # LOAD AGRICULTURE IMAGE
    # ──────────────────────────────────────────────────────────────────────────

    IMAGE_PATH = get_agri_image_path()

    # ──────────────────────────────────────────────────────────────────────────
    # HERO — AGRICULTURE INTRODUCTION
    # ──────────────────────────────────────────────────────────────────────────

    hero_col1, hero_col2 = st.columns([1.1, 0.9], gap="large")
    with hero_col1:
        st.markdown("""
        <div>
            <div class="agri-kicker">
                🌱 AGRICULTURE • FOOD • FARMING • LIVELIHOODS
            </div>
            <h1 style="font-size:38px !important; line-height:1.15 !important; margin:0 0 16px !important; letter-spacing:-0.5px; color:#ffffff;">
                Agriculture is more than<br>
                <span style="background:linear-gradient(90deg,#43e97b,#38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">growing a crop.</span>
            </h1>
            <p class="agri-hero-text" style="font-size:14.5px; line-height:1.75; color:rgba(255,255,255,0.72); margin-bottom:20px;">
                Agriculture is the complete process of producing food
                and other useful products from the land — from preparing
                the soil and selecting the right crop to managing water,
                nutrients, weather, pests, harvesting and finally selling
                the produce.
            </p>
            <div class="agri-hero-highlight">
                <span>🌾</span>
                <div>
                    <b>Every farming season is a chain of decisions.</b>
                    <small>
                        A decision about soil, water, crop, weather or
                        market can affect the final harvest and farmer
                        income.
                    </small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with hero_col2:
        if IMAGE_PATH.exists():
            st.image(str(IMAGE_PATH), use_container_width=True)
            st.markdown("""
            <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; justify-content:center;">
                <span class="overlay-pill">🌱 Soil</span>
                <span class="overlay-pill">🌦️ Climate</span>
                <span class="overlay-pill">💧 Water</span>
                <span class="overlay-pill">🌾 Crop</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:300px; font-size:48px; color:#43e97b;">
                🌾
                <span style="font-size:18px; font-weight:700; color:#ffffff; margin-top:10px;">Agri Fusion Portal</span>
            </div>
            """, unsafe_allow_html=True)



    # ──────────────────────────────────────────────────────────────────────────
    # WHAT IS AGRICULTURE?
    # ──────────────────────────────────────────────────────────────────────────

    st.markdown("""
    <div class="agri-section-title">
        <span>🌍</span>
        <div>
            <h2>What is the Agriculture Domain?</h2>
            <p>
                Agriculture connects natural resources, science, farming
                practices, economics and human decision-making.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    domain_cols = st.columns(4)

    domain_cards = [

        (
            "🌱",
            "Crop Production",
            "Selecting suitable crops, preparing land, sowing seeds, managing crop growth and harvesting at the right time."
        ),

        (
            "🧪",
            "Soil Management",
            "Understanding soil pH, nutrients, texture, organic matter and moisture so crops can receive the right conditions."
        ),

        (
            "💧",
            "Water Management",
            "Providing the right amount of water at the right time while avoiding both water stress and unnecessary irrigation."
        ),

        (
            "🌦️",
            "Climate & Weather",
            "Temperature, rainfall, humidity, wind and seasonal changes directly influence crop growth and farm decisions."
        ),

    ]

    for col, (icon, title, text) in zip(domain_cols, domain_cards):

        with col:

            st.markdown(
                f"""
                <div class="agri-domain-card">

                    <div class="domain-icon">{icon}</div>

                    <div class="domain-title">
                        {title}
                    </div>

                    <div class="domain-text">
                        {text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ──────────────────────────────────────────────────────────────────────────
    # FARMING IS A COMPLETE CYCLE
    # ──────────────────────────────────────────────────────────────────────────

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="agri-section-title">
        <span>🔄</span>
        <div>
            <h2>The Farming Cycle</h2>
            <p>
                A successful crop depends on decisions made throughout the
                entire agricultural cycle.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    cycle_steps = [
        ("01", "🗺️", "Land Selection", "Choose suitable land"),
        ("02", "🧪", "Soil Testing", "Understand soil condition"),
        ("03", "🌾", "Crop Selection", "Choose the right crop"),
        ("04", "🚜", "Land Preparation", "Prepare the field"),
        ("05", "🌱", "Sowing", "Plant at the right time"),
        ("06", "💧", "Irrigation", "Manage crop water"),
        ("07", "🧪", "Fertilization", "Supply crop nutrients"),
        ("08", "🛡️", "Crop Protection", "Control weeds, pests and disease"),
        ("09", "📈", "Monitoring", "Track crop growth"),
        ("10", "🌾", "Harvest", "Harvest at maturity"),
        ("11", "🏠", "Storage", "Protect harvested produce"),
        ("12", "💰", "Market", "Sell at a suitable price"),
    ]


    cycle_html = """
    <div class="agri-cycle">
    """

    for number, icon, title, description in cycle_steps:

        cycle_html += f"""
        <div class="cycle-item">

            <div class="cycle-number">
                {number}
            </div>

            <div class="cycle-icon">
                {icon}
            </div>

            <div class="cycle-title">
                {title}
            </div>

            <div class="cycle-description">
                {description}
            </div>

        </div>
        """

    cycle_html += "</div>"

    st.markdown(cycle_html, unsafe_allow_html=True)


    # ──────────────────────────────────────────────────────────────────────────
    # MAJOR PROBLEMS IN AGRICULTURE
    # ──────────────────────────────────────────────────────────────────────────

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="agri-section-title">
        <span>⚠️</span>
        <div>
            <h2>Why Farming is Difficult</h2>
            <p>
                Farmers work with many factors that can change from one season
                to another.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    problem_cols = st.columns(3)

    problems = [

        (
            "🌡️",
            "Climate Uncertainty",
            "Unexpected heat, delayed rainfall, heavy rain and changing seasons can affect sowing, crop growth and harvest."
        ),

        (
            "💧",
            "Water Scarcity",
            "Farmers must balance crop water requirements with limited groundwater, borewell capacity, rainfall and irrigation resources."
        ),

        (
            "🌱",
            "Soil Problems",
            "Low nutrients, unsuitable pH, poor soil structure and declining soil health can reduce crop productivity."
        ),

        (
            "🐛",
            "Pests & Diseases",
            "Insects, fungal diseases and other crop stresses can spread quickly when environmental conditions are favourable."
        ),

        (
            "📉",
            "Uncertain Yield",
            "Weather, soil, crop management and biological conditions can make final production difficult to estimate."
        ),

        (
            "💰",
            "Market Uncertainty",
            "Crop prices can change with supply, demand, arrivals, season, location and market conditions."
        ),

    ]


    for col, (icon, title, text) in zip(problem_cols, problems):

        with col:

            st.markdown(
                f"""
                <div class="agri-problem-card">

                    <div class="problem-icon">
                        {icon}
                    </div>

                    <div class="problem-title">
                        {title}
                    </div>

                    <div class="problem-text">
                        {text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ──────────────────────────────────────────────────────────────────────────
    # FARMER DECISIONS
    # ──────────────────────────────────────────────────────────────────────────

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="agri-section-title">
        <span>🧑‍🌾</span>
        <div>
            <h2>What Does a Farmer Decide?</h2>
            <p>
                Farming is a continuous decision-making process.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    decision_cols = st.columns(5)

    decisions = [

        (
            "🌾",
            "What to grow?",
            "Which crop is suitable for my land, soil, climate and season?"
        ),

        (
            "🌱",
            "When to sow?",
            "Is the current weather and soil condition suitable for planting?"
        ),

        (
            "💧",
            "When to irrigate?",
            "Does the crop need water today and how much should be applied?"
        ),

        (
            "📈",
            "What harvest to expect?",
            "How much produce could the farm potentially produce?"
        ),

        (
            "💰",
            "When to sell?",
            "What market price and nearby market conditions should I consider?"
        ),

    ]


    for col, (icon, title, text) in zip(decision_cols, decisions):

        with col:

            st.markdown(
                f"""
                <div class="agri-decision-card">

                    <div class="decision-icon">
                        {icon}
                    </div>

                    <div class="decision-title">
                        {title}
                    </div>

                    <div class="decision-text">
                        {text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ──────────────────────────────────────────────────────────────────────────
    # AGRICULTURE + DATA
    # ──────────────────────────────────────────────────────────────────────────

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="agri-intelligence">

        <div class="intelligence-left">

            <div class="intelligence-label">
                🌱 FROM TRADITIONAL FARMING TO DATA-DRIVEN FARMING
            </div>

            <h2>
                Better decisions start with
                <span>better information.</span>
            </h2>

            <p>
                Modern agriculture can combine farmer experience with
                information from soil, weather, climate, crop and market data.
                This does not replace the farmer's knowledge. It helps turn
                available information into easier and more timely decisions.
            </p>

        </div>

        <div class="intelligence-right">

            <div class="intelligence-node">
                <span>🌦️</span>
                <b>Weather</b>
                <small>Temperature • Rainfall • Humidity</small>
            </div>

            <div class="intelligence-node">
                <span>🧪</span>
                <b>Soil</b>
                <small>pH • Nutrients • Texture • Moisture</small>
            </div>

            <div class="intelligence-node">
                <span>🌾</span>
                <b>Crop</b>
                <small>Type • Growth • Water requirement</small>
            </div>

            <div class="intelligence-node">
                <span>📊</span>
                <b>Data & AI</b>
                <small>Patterns • Predictions • Recommendations</small>
            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)


    # ──────────────────────────────────────────────────────────────────────────
    # FINAL MESSAGE
    # ──────────────────────────────────────────────────────────────────────────

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="agri-final">

        <div class="final-icon">
            🌾
        </div>

        <div>
            <div class="final-title">
                Farming is a balance between nature, knowledge and decisions.
            </div>

            <div class="final-text">
                Soil tells us what the land can provide.
                Weather tells us what nature may bring.
                The crop tells us what it needs.
                The farmer brings experience, judgement and action.
            </div>
        </div>

    </div>
    """, unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════════════════
# PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Project Overview":  # paste as elif in app.py

    st.markdown("# 📊 Project Overview")

    tabs = st.tabs([
        "🌍 Domain & Problem",
        "📦 Data Sources",
        "🌾 Crop Recommendation",
        "🌡️ Climate Risk",
        "💧 Irrigation",
        "📈 Yield",
        "💰 Market Price",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 0 — DOMAIN & PROBLEM
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("## 🌍 Agricultural Domain Overview")
        st.markdown("""
        <div class='glass'>
        <b style='color:#7cb342; font-size:15px;'>What problem are we solving?</b><br><br>
        <p>Farmers in <b>Telangana and Andhra Pradesh</b> face five major challenges every season:</p>
        <ol style='color:rgba(255,255,255,0.78); line-height:2.0;'>
            <li><b>Climate uncertainty</b> — they don't know if extreme weather will damage their crops before it happens</li>
            <li><b>Wrong crop selection</b> — they plant the wrong crop for their soil, leading to poor yield</li>
            <li><b>Water mismanagement</b> — over-irrigation wastes water; under-irrigation reduces yield</li>
            <li><b>Yield surprise</b> — no estimate of how much crop to expect at harvest time</li>
            <li><b>Market blindness</b> — they sell at whatever price is offered, not knowing the fair market rate</li>
        </ol>
        <p><b style='color:#f0b429;'>Agri Fusion</b> solves all five using five separate AI models — each answering one specific question for the farmer.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🤖 Five AI Models — Five Questions Answered")
        overview_data = pd.DataFrame({
            "Model": [
                "🌾 Crop Recommendation",
                "🌡️ Climate Risk",
                "💧 Irrigation Advisor",
                "📈 Yield Estimator",
                "💰 Market Price",
            ],
            "Farmer's Question": [
                "Which crop should I plant?",
                "Is it safe to farm today?",
                "How much water does my crop need?",
                "How much will I harvest?",
                "What price will I get at the market?",
            ],
            "ML Problem Type": [
                "Multi-Class Classification",
                "Classification (Time-Series based)",
                "Regression",
                "Regression",
                "Regression",
            ],
            "Target Variable": [
                "Crop Name (22 classes)",
                "Risk Level (Low/Moderate/High/Extreme)",
                "Water need (mm/day)",
                "Yield (tonnes/hectare)",
                "Price (₹ per quintal)",
            ],
            "Best Model Selected": [
                "Random Forest Classifier",
                "Random Forest Classifier",
                "Linear Regression",
                "Random Forest Regressor",
                "Random Forest Regressor (tuned)",
            ],
        })
        st.dataframe(overview_data, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — DATA SOURCES
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("## 📦 Data Sources")
        st.markdown("""
        <div class='glass'>
        <b style='color:#7cb342; font-size:15px;'>Where did the data come from?</b><br><br>
        <p>Each model was trained using data collected from publicly available agricultural and
        meteorological sources. Below is a complete reference for each source used.</p>
        </div>
        """, unsafe_allow_html=True)

        src_cols = st.columns(2)
        with src_cols[0]:
            st.markdown("""
            <div class='glass' style='padding:20px;'>
                <div style='font-size:22px; margin-bottom:8px;'>🌤️ Open-Meteo API</div>
                <div style='color:#7cb342; font-weight:700; font-size:13px; margin-bottom:8px;'>
                    Used in: Climate Risk, Irrigation, Yield
                </div>
                <div style='color:rgba(255,255,255,0.7); font-size:13px; line-height:1.7;'>
                    Provides decades of high-resolution historical weather data including
                    temperature, precipitation, soil moisture, solar radiation, wind speed,
                    humidity, and ET₀ (reference evapotranspiration).<br><br>
                    <b style='color:#f0b429;'>Source:</b>
                    https://open-meteo.com/en/docs/historical-weather-api<br>
                    <b style='color:#f0b429;'>Data API Used:</b> ERA5 Reanalysis (1940–present)<br>
                    <b style='color:#f0b429;'>Resolution:</b> Hourly, 9km grid
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='glass' style='padding:20px;'>
                <div style='font-size:22px; margin-bottom:8px;'>🏔️ SoilGrids (ISRIC)</div>
                <div style='color:#7cb342; font-weight:700; font-size:13px; margin-bottom:8px;'>
                    Used in: Climate Risk, Irrigation, Yield, Crop Recommendation
                </div>
                <div style='color:rgba(255,255,255,0.7); font-size:13px; line-height:1.7;'>
                    Global gridded soil property maps derived from machine learning models
                    trained on soil profile observations and environmental covariates.<br><br>
                    <b style='color:#f0b429;'>Source:</b>
                    https://www.isric.org/explore/soilgrids<br>
                    <b style='color:#f0b429;'>Properties:</b> Soil pH, Organic Carbon, Clay %,
                    Sand %, Silt %, CEC, Bulk Density, Nitrogen<br>
                    <b style='color:#f0b429;'>Depth:</b> 0–5 cm, 5–15 cm layers
                </div>
            </div>
            """, unsafe_allow_html=True)

        with src_cols[1]:
            st.markdown("""
            <div class='glass' style='padding:20px;'>
                <div style='font-size:22px; margin-bottom:8px;'>📋 FAO-56 Standards</div>
                <div style='color:#7cb342; font-weight:700; font-size:13px; margin-bottom:8px;'>
                    Used in: Irrigation (Kc values, ET calculations)
                </div>
                <div style='color:rgba(255,255,255,0.7); font-size:13px; line-height:1.7;'>
                    FAO Irrigation and Drainage Paper No. 56 — the international standard
                    for computing crop water requirements using the Penman-Monteith method.
                    Provides crop coefficients (Kc) for each growth stage.<br><br>
                    <b style='color:#f0b429;'>Source:</b>
                    https://www.fao.org/publications/card/en/c/70d2c0fc-4f2a-5b83-94db-d5a6ff0e2cb6/<br>
                    <b style='color:#f0b429;'>Used for:</b> Kc values, Root depth, Water requirement formula
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='glass' style='padding:20px;'>
                <div style='font-size:22px; margin-bottom:8px;'>📊 Government Agricultural Data</div>
                <div style='color:#7cb342; font-weight:700; font-size:13px; margin-bottom:8px;'>
                    Used in: Yield, Market Price
                </div>
                <div style='color:rgba(255,255,255,0.7); font-size:13px; line-height:1.7;'>
                    Agricultural yield data collected from the
                    <b>Ministry of Agriculture & Farmers Welfare, India</b>.
                    Market price data collected from the
                    <b>Agmarknet portal (APEDA)</b> — India's national agricultural
                    market price database covering crops across districts.<br><br>
                    <b style='color:#f0b429;'>Yield Source:</b>
                    data.gov.in — District-wise crop production statistics<br>
                    <b style='color:#f0b429;'>Market Source:</b>
                    agmarknet.gov.in — Daily arrival and price data
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — CROP RECOMMENDATION
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("## 🌾 Crop Recommendation")
        st.markdown("""
        <div class='glass'>
        <b style='color:#7cb342;'>ML Problem Type:</b> Multi-Class Classification<br>
        <b style='color:#7cb342;'>Target Variable:</b> Crop label (22 crop types)<br>
        <b style='color:#7cb342;'>Training Records:</b> Generated dataset — 28 cities × 40 samples × all seasons
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Models Trained & Compared")
        crop_results = pd.DataFrame({
            "Model": ["Logistic Regression","Decision Tree","Random Forest","XGBoost"],
            "Notes": [
                "Baseline — linear boundary, lower accuracy on 22 crop classes",
                "Prone to overfitting — needs depth control",
                "✅ Best — stable accuracy, handles soil + weather feature mix well",
                "Strong competitor — slightly complex for 22-class problem",
            ],
            "Fit": ["Moderate","Overfit","✅ Good Fit","Good Fit"],
        })
        st.dataframe(crop_results, use_container_width=True, hide_index=True)

        st.markdown("### 📥 Features Used (7 total)")
        st.markdown("""
        | Feature | Unit | Why It Matters |
        |---|---|
        | Nitrogen (N) | kg/ha | Primary macronutrient — different crops need different N levels |
        | Phosphorus (P) | kg/ha | Root development and energy transfer |
        | Potassium (K) | kg/ha | Stress resistance and water regulation |
        | Temperature | °C | Each crop has a specific temperature range for optimal growth |
        | Humidity | % | Affects disease pressure and transpiration |
        | Soil pH | 0–14 | Determines nutrient availability |
        | Rainfall | mm | Water availability — critical crop selection factor |
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — CLIMATE RISK
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("## 🌡️ Climate Risk Prediction")
        st.markdown("""
        <div class='glass'>
        <b style='color:#7cb342;'>ML Problem Type:</b> Multi-Class Classification
        (Time-series based weather patterns)<br>
        <b style='color:#7cb342;'>Target Variable:</b> Climate_Risk → Low / Moderate / High / Extreme<br>
        <b style='color:#7cb342;'>Training Records:</b> 1,048,575 rows
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Models Trained & Compared")
        climate_results = pd.DataFrame({
            "Model": ["Logistic Regression","KNN","SVC","Decision Tree",
                      "Random Forest","XGBoost"],
            "Test Accuracy": ["~65%","~72%","~74%","~80%","86.6%","~85%"],
            "Notes": [
                "Underfit — linear boundary insufficient for climate patterns",
                "Slow on 1M rows, moderate accuracy",
                "Slow on 1M rows, sampled training required",
                "Overfit — memorizes training data",
                "✅ Best — balanced accuracy across all risk classes",
                "Strong but slightly lower F1 on minority classes",
            ],
            "Fit": ["Underfit","Moderate","Moderate","Overfit","Good Fit","Good Fit"],
        })
        st.dataframe(climate_results, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Best Model", "Random Forest")
        c2.metric("Test Accuracy", "86.6%")
        c3.metric("Classes", "4 (Low/Moderate/High/Extreme)")

        st.markdown("""
        **⚠️ Known challenge:** Dataset is highly imbalanced — 95%+ records are "Low" risk.
        This means the model has low precision for High/Extreme classes.
        Future improvement: SMOTE oversampling or class-weighted training.
        """)

        st.markdown("### 📥 Features Used (25 numeric + 93 one-hot = 118 total)")
        st.markdown("""
        | Category | Features |
        |---|---|
        | Weather | Temperature, Humidity, Precipitation, Wind Speed, Wind Gusts, Wind Direction, Surface Pressure, Cloud Cover, Shortwave Radiation, ET₀ |
        | Derived | Heat Index, Rainfall Last 7 Days, Rainfall Last 30 Days, Consecutive Dry Days, Growing Degree Days, Soil Moisture, Soil Temperature |
        | Soil | Soil pH, Organic Carbon, Clay %, Sand %, Silt %, Elevation |
        | Categorical (OHE) | City (28), State (2), Crop (14), Season (4) |
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — IRRIGATION
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("## 💧 Irrigation Advisor")
        st.markdown("""
        <div class='glass'>
        <b style='color:#7cb342;'>ML Problem Type:</b> Regression<br>
        <b style='color:#7cb342;'>Target Variable:</b> water_requirement_mm_day (mm/day)<br>
        <b style='color:#7cb342;'>Training Records:</b> 357,504 rows<br>
        <b style='color:#7cb342;'>Formula basis:</b> FAO-56 Penman-Monteith — ETc = ET₀ × Kc
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Models Trained & Compared")
        irr_results = pd.DataFrame({
            "Model": ["Linear Regression","KNN","SVR","Decision Tree","Random Forest","XGBoost"],
            "Train RMSE": [0.9279, 2.3927, 3.0903, 1.1738, 1.1653, 1.0946],
            "Test RMSE":  [0.9258, 2.9803, 3.1017, 1.1687, 1.1595, 1.0894],
            "Train R²":   [0.9113, 0.4036, 0.0051, 0.8580, 0.8600, 0.8765],
            "Test R²":    [0.9113, 0.0804, 0.0040, 0.8586, 0.8608, 0.8771],
            "Adjusted R²":[0.9112, 0.0795, 0.0030, 0.8585, 0.8607, 0.8770],
            "Fit":        ["✅ Good Fit","❌ Underfit","❌ Underfit",
                           "✅ Good Fit","✅ Good Fit","✅ Good Fit"],
        })
        st.dataframe(
            irr_results.style.highlight_max(subset=["Test R²"], color="#1b5e20"),
            use_container_width=True, hide_index=True
        )
        st.caption("KNN and SVR trained on 20,000-row sample due to computational limits at 357K rows.")

        st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-size: 20px !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 14px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        c1.metric("🏆 Best Model", "Linear Regression")
        c2.metric("Test R²", "0.9113 (91.1%)")
        c3.metric("Test RMSE", "0.9258 mm/day")

        st.success(
            "Linear Regression selected — highest Test R² (91.1%), "
            "within mentor's 85–95% target range, minimal train-test gap "
            "confirming Good Fit."
        )

        st.markdown("### 📥 Key Features (70 total after encoding)")
        st.markdown("""
        | Category | Features |
        |---|---|
        | Weather (live) | Temperature, Humidity, Rainfall, Wind Speed, Solar Radiation, ET₀ |
        | Soil (lookup) | Soil pH, Organic Carbon, Clay %, Sand %, Silt %, CEC, Bulk Density, Field Capacity, Wilting Point, Available Water, Nitrogen |
        | Crop | Crop name (OHE), Growth Stage (Ordinal), Root Depth, Kc Band (Binned) |
        | Location | City (OHE), State (Binary) |
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — YIELD
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("## 📈 Yield Estimator")
        st.markdown("""
        <div class='glass'>
        <b style='color:#7cb342;'>ML Problem Type:</b> Regression<br>
        <b style='color:#7cb342;'>Target Variable:</b> Crop Yield (tonnes per hectare)<br>
        <b style='color:#7cb342;'>Training Records:</b> 15,276 rows (district-wise multi-year data)<br>
        <b style='color:#7cb342;'>Special handling:</b> Coconut dataset separated (different scale) — modelled independently
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Models Trained & Compared")
        yield_results = pd.DataFrame({
            "Model": ["Random Forest","XGBoost","Decision Tree","KNN",
                      "Linear Regression","SVR"],
            "Train RMSE": [1.009, 1.062, 0.000, 2.737, 4.021, 6.624],
            "Test RMSE":  [2.980, 3.028, 3.720, 4.050, 4.563, 7.822],
            "Train R²":   [0.9954, 0.9950, 1.0000, 0.9665, 0.9277, 0.8038],
            "Test R²":    [0.9661, 0.9603, 0.8890, 0.8787, 0.8565, 0.5742],
            "Fit":        ["✅ Good Fit","✅ Good Fit","❌ Overfit",
                           "Moderate","Moderate","❌ Underfit"],
        })
        st.dataframe(
            yield_results.style.highlight_max(subset=["Test R²"], color="#1b5e20"),
            use_container_width=True, hide_index=True
        )

        # Custom metric styling
        st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-size: 20px !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 14px !important;
        }
        </style>
        """, unsafe_allow_html=True)


        c1, c2, c3 = st.columns(3)

        c1.metric("🏆 Best Model", "Random Forest")
        c2.metric("Test R²", "0.9661 (96.6%)")
        c3.metric("Test RMSE", "2.98 tonnes/ha")

        st.success(
            "Random Forest selected — Test R² of 96.6% with Good Fit. "
            "Decision Tree was overfit (Train R²=1.0); XGBoost was close "
            "but Random Forest had better generalization."
        )

        st.markdown("### 📥 Features Used (24 input, 114 after encoding)")
        st.markdown("""
        | Category | Features |
        |---|---|
        | Location | State (binary), District (OHE), Latitude, Longitude, Elevation |
        | Farm | Area (hectares), Year |
        | Weather | Mean/Max/Min Temperature, Precipitation, Solar Radiation, Wind Speed, Humidity, ET₀, Soil Moisture, Soil Temperature |
        | Soil | Soil pH, Organic Carbon, Clay %, Sand %, Silt % |
        | Crop | Season (OHE), Crop name (OHE) |
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — MARKET PRICE
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown("## 💰 Market Price Prediction")
        st.markdown("""
        <div class='glass'>
        <b style='color:#7cb342;'>ML Problem Type:</b> Regression<br>
        <b style='color:#7cb342;'>Target Variable:</b> Market Price (₹ per quintal)<br>
        <b style='color:#7cb342;'>Data Source:</b> Agmarknet — India's national agricultural market price database<br>
        <b style='color:#7cb342;'>Columns:</b> State, District, Commodity, Date, Arrival Quantity, Market Price, Price Unit<br>
        <b style='color:#7cb342;'>Hyperparameter Tuning:</b> GridSearchCV applied — Best params: max_depth=20, n_estimators=100
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Models Trained & Compared")
        market_results = pd.DataFrame({
            "Model": ["Linear Regression","Decision Tree","Random Forest",
                      "Gradient Boosting","XGBoost"],
            "Notes": [
                "Low R² — market price has non-linear relationships",
                "Overfit — perfect train score, poor test",
                "✅ Best — strong R², best generalization",
                "Good competitor — slightly slower training",
                "Strong — similar to Random Forest",
            ],
            "Fit": ["Underfit","❌ Overfit","✅ Good Fit","Good Fit","Good Fit"],
        })
        st.dataframe(market_results, use_container_width=True, hide_index=True)

        c1, c2, c3, c4 = st.columns(4)

        c1.markdown("""
        <div class="metric-box">
            <div class="metric-label">🏆 Best Model</div>
            <div class="metric-value">Random Forest (Tuned)</div>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown("""
        <div class="metric-box">
            <div class="metric-label">Test R²</div>
            <div class="metric-value">0.9603 (96.0%)</div>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown("""
        <div class="metric-box">
            <div class="metric-label">MAE</div>
            <div class="metric-value">₹189.78</div>
        </div>
        """, unsafe_allow_html=True)

        c4.markdown("""
        <div class="metric-box">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">₹534.38</div>
        </div>
        """, unsafe_allow_html=True)
        st.success("Random Forest Regressor with GridSearchCV tuning selected — Test R² of 96.0%, MAE of ₹189.78, showing that most predictions are within ₹190 of the actual market price.")

        st.markdown("### 📥 Features Used (8 total)")
        st.markdown("""
        | Feature | Type | Description |
        |---|---|---|
        | Commodity | Label Encoded | Crop/commodity being sold |
        | State | Binary (0/1) | Telangana=1, Andhra Pradesh=0 |
        | District | Label Encoded | District market location |
        | Day | Numeric | Day of month |
        | Month | Numeric | Month of year |
        | Year | Numeric | Year of sale |
        | Quarter | Numeric | 1–4 (seasonal pricing pattern) |
        | Arrival Quantity | Numeric (quintals) | Volume of crop arriving at market |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# CLIMATE RISK
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🌡️  Climate Risk":
    render_prediction_breadcrumb(page)
    st.markdown("# 🌡️ Climate Risk Early Warning")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    # ── HOW IT WORKS ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class='glass' style='margin-bottom:20px;'>
        <div style='display:flex; align-items:center; gap:12px;'>
            <div style='font-size:32px;'>📡</div>
            <div>
                <div style='color:#7cb342; font-weight:700; font-size:15px; margin-bottom:4px;'>
                    Just select your city and crop — the rest is automatic
                </div>
                <div style='color:rgba(255,255,255,0.65); font-size:13px;'>
                    Live weather &bull; 30-day rainfall history &bull; Soil data &bull;
                    Heat index &bull; Dry spell count — all fetched from Open-Meteo automatically
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT FORM — 2 fields only ────────────────────────────────────────────
    with st.form("climate_form"):
        col1, col2 = st.columns(2)
        with col1:
            cr_city = st.selectbox("📍 Your City", CITIES)
        with col2:
            cr_crop = st.selectbox(
                "🌾 Your Crop",
                CROPS,
                help="Select the crop you are currently growing or planning to grow"
            )
        submitted_cr = st.form_submit_button(
            "🌡️ Check Climate Risk Now",
            use_container_width=True
        )

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if submitted_cr:
        with st.spinner("📡 Fetching live weather + 30-day history from Open-Meteo..."):
            try:
                result = api_predict_climate_risk(cr_city, cr_crop)

                # ── Risk config — farmer language, not ML labels ───────────────
                RISK_CFG = {
                    "LOW":      {
                        "color":   "#43e97b",
                        "bg":      "rgba(67,233,123,0.07)",
                        "label":   "Safe to Farm",
                        "emoji":   "🟢",
                        "msg":     "Your farm is safe today. No major weather risk detected.",
                    },
                    "MODERATE": {
                        "color":   "#ffd54f",
                        "bg":      "rgba(255,213,79,0.07)",
                        "label":   "Be Careful",
                        "emoji":   "🟡",
                        "msg":     "Some weather concerns today. Keep a close watch on your crop.",
                    },
                    "HIGH":     {
                        "color":   "#ff9800",
                        "bg":      "rgba(255,152,0,0.07)",
                        "label":   "Take Precaution",
                        "emoji":   "🟠",
                        "msg":     "Risk is high. Take protective action for your crop today.",
                    },
                    "EXTREME":  {
                        "color":   "#f44336",
                        "bg":      "rgba(244,67,54,0.07)",
                        "label":   "Danger",
                        "emoji":   "⛔",
                        "msg":     "Extreme risk! Protect your crop immediately and contact your agricultural officer.",
                    },
                }
                cfg   = RISK_CFG.get(result["risk_level"], RISK_CFG["LOW"])
                score = result["risk_score"]  # 0–100 integer
                s     = result["climate_summary"]

                # ── 1. MAIN RISK BANNER WITH METER ───────────────────────────
                # Traffic light dots — active = bright, inactive = dim
                tl_low  = "#43e97b" if result["risk_level"] == "LOW"      else "rgba(67,233,123,0.18)"
                tl_mod  = "#ffd54f" if result["risk_level"] == "MODERATE"  else "rgba(255,213,79,0.18)"
                tl_high = "#ff9800" if result["risk_level"] == "HIGH"      else "rgba(255,152,0,0.18)"
                tl_ext  = "#f44336" if result["risk_level"] == "EXTREME"   else "rgba(244,67,54,0.18)"

                st.markdown(f"""
                <div style='background:{cfg["bg"]};
                            border:2px solid {cfg["color"]};
                            border-radius:22px; padding:26px 28px;
                            margin-bottom:22px;'>

                    <!-- Context row -->
                    <div style='font-size:13px; color:rgba(255,255,255,0.45);
                                margin-bottom:16px;'>
                        🌾 {result["crop"]} &nbsp;|&nbsp;
                        📍 {result["city"]} &nbsp;|&nbsp;
                        🍂 {result["season"]} &nbsp;|&nbsp;
                        📅 {result["date"]}
                    </div>

                    <!-- Status row -->
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; flex-wrap:wrap; gap:16px;
                                margin-bottom:20px;'>
                        <div>
                            <div style='font-size:13px; color:rgba(255,255,255,0.45);
                                        margin-bottom:6px;'>Today's Climate Risk</div>
                            <div style='font-size:38px; font-weight:800;
                                        color:{cfg["color"]}; line-height:1.1;'>
                                {cfg["emoji"]} {cfg["label"]}
                            </div>
                            <div style='font-size:14px; color:rgba(255,255,255,0.65);
                                        margin-top:8px;'>
                                {cfg["msg"]}
                            </div>
                        </div>

                        <!-- Traffic light -->
                        <div style='display:flex; flex-direction:column;
                                    gap:8px; align-items:center;'>
                            <div style='width:28px; height:28px; border-radius:50%;
                                        background:{tl_ext}; border:2px solid rgba(244,67,54,0.4);'></div>
                            <div style='width:28px; height:28px; border-radius:50%;
                                        background:{tl_high}; border:2px solid rgba(255,152,0,0.4);'></div>
                            <div style='width:28px; height:28px; border-radius:50%;
                                        background:{tl_mod}; border:2px solid rgba(255,213,79,0.4);'></div>
                            <div style='width:28px; height:28px; border-radius:50%;
                                        background:{tl_low}; border:2px solid rgba(67,233,123,0.4);'></div>
                        </div>
                    </div>

                    <!-- Risk meter label row -->
                    <div style='display:flex; justify-content:space-between;
                                font-size:11px; color:rgba(255,255,255,0.35);
                                margin-bottom:6px; padding:0 2px;'>
                        <span>🟢 Safe (0)</span>
                        <span>🟡 Caution (30)</span>
                        <span>🟠 High (60)</span>
                        <span>⛔ Extreme (80-100)</span>
                    </div>

                    <!-- Gradient background bar -->
                    <div style='width:100%; height:16px; border-radius:10px;
                                background:linear-gradient(90deg,
                                    #43e97b 0%, #ffd54f 33%,
                                    #ff9800 66%, #f44336 100%);'>
                    </div>

                    <!-- Score pointer -->
                    <div style='position:relative; height:24px; margin-top:2px;'>
                        <div style='position:absolute;
                                    left:calc({min(score, 97)}% - 12px);
                                    top:0; font-size:18px;
                                    color:{cfg["color"]};
                                    line-height:1;'>▲</div>
                    </div>

                    <!-- Score number + confidence -->
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; margin-top:4px;'>
                        <div>
                            <span style='font-size:24px; font-weight:800;
                                         color:{cfg["color"]};'>{score}</span>
                            <span style='font-size:14px; color:rgba(255,255,255,0.4);'> / 100</span>
                        </div>
                        <div style='font-size:12px; color:rgba(255,255,255,0.35);'>
                            Model confidence: {result["confidence_pct"]}%
                        </div>
                    </div>

                </div>
                """, unsafe_allow_html=True)

                # ── 2. MAIN CLIMATE CONCERNS ──────────────────────────────────
                st.markdown("#### 🔍 Main Climate Concerns")
                SEV_COLOR = {
                    "LOW":      "#43e97b",
                    "MODERATE": "#ffd54f",
                    "HIGH":     "#ff9800",
                    "EXTREME":  "#f44336",
                }
                n_risks   = min(len(result["main_risks"]), 3)
                risk_cols = st.columns(n_risks)
                for idx, r in enumerate(result["main_risks"]):
                    col = risk_cols[idx % n_risks]
                    sev_c = SEV_COLOR.get(r["severity"], "#888")
                    col.markdown(f"""
                    <div class='mcard' style='border-color:{sev_c}; text-align:left;'>
                        <div style='font-size:17px; font-weight:700;
                                    color:{sev_c}; margin-bottom:4px;'>
                            {r["risk"]}
                        </div>
                        <div style='font-size:10px; font-weight:700;
                                    color:{sev_c}; letter-spacing:1.5px;
                                    margin-bottom:8px;'>
                            {r["severity"]}
                        </div>
                        <div style='font-size:12px; color:rgba(255,255,255,0.6);
                                    line-height:1.5;'>
                            {r["detail"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── 3. CROP IMPACT + WHAT TO DO ───────────────────────────────
                imp_col, rec_col = st.columns(2)

                with imp_col:
                    st.markdown(f"""
                    <div class='glass' style='height:100%;'>
                        <div style='color:#7cb342; font-weight:700;
                                    font-size:14px; margin-bottom:10px;'>
                            🌾 How will this affect your {result["crop"]}?
                        </div>
                        <div style='color:rgba(255,255,255,0.82);
                                    font-size:14px; line-height:1.7;'>
                            {result["crop_impact"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with rec_col:
                    recs_html = "".join(
                        f"""<div style='display:flex; gap:8px; margin-bottom:9px;
                                        align-items:flex-start;'>
                                <span style='margin-top:1px;'>✔</span>
                                <span style='color:rgba(255,255,255,0.8);
                                             font-size:13px; line-height:1.5;'>
                                    {rec.lstrip("✅🌡️🌿💧📅🌱🚧⏸️🧪👀💊🪵⏸📱🌤️📋")
                                       .strip()}
                                </span>
                            </div>"""
                        for rec in result["recommendations"]
                    )
                    st.markdown(f"""
                    <div class='glass' style='height:100%;'>
                        <div style='color:#7cb342; font-weight:700;
                                    font-size:14px; margin-bottom:12px;'>
                            ✅ What should you do today?
                        </div>
                        {recs_html}
                    </div>
                    """, unsafe_allow_html=True)

                # ── 4. TODAY'S WEATHER — auto-fetched details (collapsed) ─────
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📊 View weather data fetched automatically"):
                    st.caption(
                        "These values were fetched from Open-Meteo. "
                        "You did not enter any of these — the app collected them for your city."
                    )
                    wc1,wc2,wc3,wc4,wc5 = st.columns(5)
                    wc1.metric("🌡️ Temperature",   s["temperature"])
                    wc2.metric("💧 Humidity",       s["humidity"])
                    wc3.metric("🌧️ Rain Today",     s["rainfall_today"])
                    wc4.metric("💨 Wind Speed",     s["wind_speed"])
                    wc5.metric("🌡️ Heat Index",     s["heat_index"])

                    wc6,wc7,wc8,wc9,wc10 = st.columns(5)
                    wc6.metric("🌧️ Rain (7 days)",  s["rainfall_last_7_days"])
                    wc7.metric("🌧️ Rain (30 days)", s["rainfall_last_30_days"])
                    wc8.metric("☀️ Dry Days",        s["consecutive_dry_days"])
                    wc9.metric("💦 ET₀",             s["et0"])
                    wc10.metric("🪣 Soil Moisture",  s["soil_moisture"])

                # ── SAVE TO SUPABASE ──────────────────────────────────────────
                try:
                    soil_row = {}
                    if result.get("city"):
                        city_matches = CITY_SOIL[CITY_SOIL["City"].astype(str).str.strip().str.lower() == str(result["city"]).strip().lower()]
                        if not city_matches.empty:
                            soil_row = city_matches.iloc[0].to_dict()
                    save_climate_risk(
                        st.session_state.get("farmer_phone", "guest"),
                        result,
                        soil_row,
                    )
                except Exception:
                    pass

            except requests.exceptions.Timeout:
                st.error("⏱️ Weather data timed out. Please try again in a moment.")
            except requests.exceptions.ConnectionError:
                st.error("📡 No internet connection. Check your network and try again.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("Try a different city, or check if Open-Meteo API is reachable.")
# ══════════════════════════════════════════════════════════════════════════════
# CROP RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🌾  Crop Recommendation":
    render_prediction_breadcrumb(page)
    st.markdown("# 🌱 Crop Recommendation")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    # How it works banner
    st.markdown("""
    <div class='glass' style='margin-bottom:20px;'>
        <div style='display:flex; align-items:center; gap:12px;'>
            <div style='font-size:32px;'>📡</div>
            <div>
                <div style='color:#7cb342; font-weight:700; font-size:15px; margin-bottom:4px;'>
                    Just select your city — the rest is automatic
                </div>
                <div style='color:rgba(255,255,255,0.65); font-size:13px;'>
                    Live weather (Open-Meteo) & local soil profile (SoilGrids lookup) —
                    all fetched automatically for your location.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("crop_form"):
        cr_city = st.selectbox("📍 Your City / District", CITIES)
        submitted_crop = st.form_submit_button(
            "🌱 Get Crop Recommendation", use_container_width=True
        )

    if submitted_crop:
        with st.spinner("📡 Fetching live weather + soil data for your city..."):
            try:
                result = api_recommend_crop(cr_city)

                # ── 1. MAIN RECOMMENDATION BANNER ──────────────────────────
                st.markdown(f"""
                <div class='rcard' style='border:2px solid #7cb342; margin-bottom:20px;'>
                    <div style='display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;'>
                        <div>
                            <div style='color:rgba(255,255,255,0.55); font-size:13px; margin-bottom:6px;'>
                                📍 {result["location"]["city"]} &nbsp;|&nbsp; 📅 {result["fetched_at"]}
                            </div>
                            <div style='font-size:13px; color:rgba(255,255,255,0.5); margin-bottom:12px;'>
                                Recommended Optimal Crop
                            </div>
                            <div style='font-size:42px; font-weight:800; color:#7cb342;'>
                                {result["crop_emoji"]} {result["recommended_crop"]}
                            </div>
                        </div>
                        <div style='text-align:center; min-width:130px;'>
                            <div style='font-size:11px; color:rgba(255,255,255,0.5); margin-bottom:6px;'>MODEL CONFIDENCE</div>
                            <div style='font-size:52px; font-weight:800;
                                        background:linear-gradient(135deg,#7cb342,#fff);
                                        -webkit-background-clip:text; background-clip:text; color:transparent;'>
                                {result["confidence_percent"]}
                            </div>
                            <div style='font-size:14px; color:rgba(255,255,255,0.5);'>%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── 2. WHY THIS CROP + CARE TIPS ────────────────────────────
                why_col, care_col = st.columns([1, 1])

                with why_col:
                    st.markdown(f"""
                    <div class='glass'>
                        <div style='color:#7cb342; font-weight:700; font-size:14px; margin-bottom:10px;'>
                            🔍 Why This Crop
                        </div>
                        <div style='color:rgba(255,255,255,0.85); font-size:14px; line-height:1.6;'>
                            {result["recommendation"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with care_col:
                    st.markdown(f"""
                    <div class='glass'>
                        <div style='color:#7cb342; font-weight:700; font-size:14px; margin-bottom:10px;'>
                            ✅ How To Care For This Crop
                        </div>
                        <div style='color:rgba(255,255,255,0.8); font-size:13px; line-height:1.6;'>
                            {result["care_tips"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── 2b. ALTERNATIVE SUITABLE CROPS ─────────────────────────
                if result.get("alternative_crops"):
                    st.markdown("<h4 style='color:#7cb342; margin-top:20px; font-size:16px;'>🌾 Alternative Suitable Crops</h4>", unsafe_allow_html=True)
                    alt_cols = st.columns(len(result["alternative_crops"]))
                    for idx, alt in enumerate(result["alternative_crops"]):
                        with alt_cols[idx]:
                            st.markdown(f"""
                            <div class='glass' style='text-align:center; padding:12px;'>
                                <div style='font-size:28px;'>{alt["emoji"]}</div>
                                <div style='font-weight:700; color:#fff; font-size:14px;'>{alt["crop"]}</div>
                                <div style='font-size:12px; color:#7cb342; font-weight:600;'>{alt["confidence"]}% Match</div>
                            </div>
                            """, unsafe_allow_html=True)

                # ── 3. FEATURE METRICS BREAKDOWN ────────────────────────────
                with st.expander("📊 View Weather & Soil Data Used for Prediction"):
                    w, s = result["weather"], result["soil"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🌡️ Temperature", f"{w['temperature']} °C")
                    c2.metric("💧 Humidity", f"{w['humidity']} %")
                    c3.metric("🌧️ Rainfall Rate", f"{w['rainfall']} mm")

                    c4, c5, c6, c7 = st.columns(4)
                    c4.metric("🧪 Soil pH", s["soil_ph"])
                    c5.metric("🌱 Nitrogen (N)", f"{s['nitrogen']} kg/ha")
                    c6.metric("🧬 Organic Carbon", f"{s['organic_carbon']}%")
                    c7.metric("🧱 Soil Texture", f"{s.get('soil_type', 'Clay/Sand')}")

                try:
                    save_crop_recommendation(
                        st.session_state.get("farmer_phone", "guest"),
                        cr_city,
                        result,
                        result.get("weather"),
                    )
                except Exception:
                    pass

            except Exception as e:
                st.error(f"❌ Error obtaining recommendation: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# IRRIGATION
# ══════════════════════════════════════════════════════════════════════════════
elif page in ["💧  Irrigation Advisor", "🌾 Predictions"]:
    render_prediction_breadcrumb(page)
    st.markdown("# 💧 Crop Water Requirement & Irrigation Recommendation")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    # ── HOW IT WORKS ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class='glass' style='margin-bottom:24px;'>
        <div style='display:flex; align-items:center; gap:12px;'>
            <div style='font-size:32px;'>📡</div>
            <div>
                <div style='color:#7cb342; font-weight:700; font-size:15px; margin-bottom:4px;'>
                    Select your crop, stage and city — weather is fetched automatically
                </div>
                <div style='color:rgba(255,255,255,0.6); font-size:13px;'>
                    Live temperature, humidity, rainfall and ET₀ are collected from Open-Meteo.
                    You only need to tell us about your farm.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CONSTANTS ─────────────────────────────────────────────────────────────
    CROPS_IRR = [
        "Banana","Beans","Black Gram Dal(Urd Dal)","Cotton","Grapes",
        "Karbuja(Musk Melon)","Maize","Mango","Orange","Papaya",
        "Pomegranate","Rice","Tender Coconut","Water Melon",
    ]
    IRR_STAGES = {
        "🌰 Just Planted":  "Initial",
        "🌿 Growing":       "Development",
        "🌸 Flowering":     "Mid-season",
        "🌾 Almost Ready":  "Late-season",
    }
    CITIES_IRR = sorted(CITY_SOIL["City"].tolist())

    # ── INPUT FORM ────────────────────────────────────────────────────────────
    with st.form("irrigation_form"):
        st.markdown("""
        <div class='step-hdr'>📍 STEP 1 — Your Farm</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            irr_crop  = st.selectbox("🌾 What crop are you growing?",  CROPS_IRR)
            irr_stage = st.selectbox("🌱 How far along is the crop?",  list(IRR_STAGES.keys()))
            irr_city  = st.selectbox("📍 Which city is your farm near?", CITIES_IRR)
        with c2:
            irr_size  = st.number_input("🚜 Farm Size",
                                         min_value=0.5, max_value=1000.0,
                                         value=2.0, step=0.5)
            irr_unit  = st.radio("Measure in", ["Acres", "Hectares"], horizontal=True)
            irr_src   = st.selectbox("💧 Where does your water come from?",
                                     ["Borewell","Canal","River","Tank","Farm Pond"])

        st.markdown("""
        <div class='step-hdr' style='margin-top:16px;'>
            ⚙️ STEP 2 — Optional Details (for better advice)
        </div>
        """, unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            irr_method = st.selectbox("How do you water the field?",
                                      ["I don't know","Drip","Sprinkler","Flood"])
        with c4:
            irr_pump   = st.number_input(
                "Pump speed — Liters per minute (0 = skip)",
                min_value=0.0, value=0.0, step=10.0
            )

        irr_already_watered = st.checkbox(
            "✅ I already watered this crop today",
            value=False,
            help="If you already irrigated today, we'll skip today and show you tomorrow's best time instead."
        )

        irr_days = st.slider("📅 Show forecast for how many days?", 1, 7, 3)

        submitted_irr = st.form_submit_button(
            "💧 Get My Irrigation Recommendation",
            use_container_width=True
        )

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if submitted_irr:
        growth_stage = IRR_STAGES[irr_stage]
        lat, lon     = CITY_COORDS.get(irr_city, (17.38, 78.47))
        method       = irr_method if irr_method != "I don't know" else None
        pump         = irr_pump if irr_pump > 0 else None

        with st.spinner("📡 Fetching live weather and computing irrigation decision..."):
            try:
                user_input = {
                    "city":                     irr_city,
                    "crop":                     irr_crop,
                    "growth_stage":             growth_stage,
                    "farm_size":                irr_size,
                    "farm_size_unit":           irr_unit,
                    "irrigation_source":        irr_src,
                    "irrigation_method":        method,
                    "pump_lpm":                 pump,
                    "already_irrigated_today":  irr_already_watered,
                }

                import importlib
                import backend.predict as backend_predict_mod
                importlib.reload(backend_predict_mod)

                weather = get_weather(lat, lon)
                result  = backend_predict_mod.predict(user_input, weather)

                irr_status   = result.get("irrigation_status", "IRRIGATION REQUIRED TODAY")
                farmer_msg   = result.get("farmer_message", result.get("recommendation", "Irrigate your crop today."))
                water_mm     = result.get("water_requirement_mm_day", 0.0)
                net_irr_mm   = result.get("net_irrigation_mm", water_mm)
                total_liters = result.get("total_water_liters", result.get("total_liters", 0.0))
                water_disp   = result.get("water_display", f"Approximately {total_liters:,.0f} liters")
                best_time    = result.get("best_irrigation_time", "06:00 AM – 08:00 AM")
                best_date    = result.get("best_irrigation_date", datetime.now().strftime("%d %B %Y"))
                next_days    = result.get("next_irrigation_days", 1)
                next_disp    = result.get("next_irrigation_display", f"In {next_days} days")
                next_date    = result.get("next_irrigation_date", (datetime.now() + timedelta(days=next_days)).strftime("%d %B %Y"))
                motor_time     = result.get("motor_running_time")
                ref_motor_time = result.get("reference_motor_running_time")

                # ── 1. RESULT BANNER ──────────────────────────────────────────
                if "COMPLETED" in irr_status:
                    banner_color = "#7cb342"
                    banner_icon  = "✅"
                    banner_title = "✅ WATERING COMPLETED TODAY"
                elif "NO IRRIGATION" in irr_status:
                    banner_color = "#7cb342"
                    banner_icon  = "🌧️"
                    banner_title = "🌧️ NO IRRIGATION NEEDED TODAY"
                elif "TOMORROW" in irr_status:
                    banner_color = "#f0b429"
                    banner_icon  = "🌅"
                    banner_title = "🌅 IRRIGATE TOMORROW MORNING"
                else:
                    banner_color = "#7cb342"
                    banner_icon  = "💧"
                    banner_title = "💧 IRRIGATION REQUIRED TODAY"

                st.markdown(f"""
                <div style='background:linear-gradient(135deg,rgba(20,38,14,0.95),rgba(40,68,24,0.90));
                            border:2px solid {banner_color};
                            border-radius:22px; padding:30px 32px;
                            margin-bottom:22px; text-align:center;'>
                    <div style='font-size:48px; margin-bottom:10px;'>{banner_icon}</div>
                    <div style='font-size:28px; font-weight:800;
                                color:{banner_color}; margin-bottom:8px;'>
                        {banner_title}
                    </div>
                    <div style='font-size:14px; color:rgba(255,255,255,0.55);'>
                        🌾 {irr_crop} &nbsp;|&nbsp; 📍 {irr_city}
                        &nbsp;|&nbsp; 📅 {datetime.now().strftime("%d %B %Y")}
                    </div>
                    <div style='font-size:13px; color:rgba(255,255,255,0.75);
                                margin-top:6px;'>{farmer_msg}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── 2. KEY NUMBERS ────────────────────────────────────────────
                k1, k2, k3, k4 = st.columns(4)

                k1.markdown(f"""
                <div class='mcard'>
                    <div class='val'>{total_liters:,.0f} L</div>
                    <div class='lbl'>🪣 Estimated farm water<br>{water_disp}</div>
                </div>""", unsafe_allow_html=True)

                k2.markdown(f"""
                <div class='mcard'>
                    <div class='val'>{net_irr_mm} mm</div>
                    <div class='lbl'>💧 Net water to provide<br>({water_mm} mm/day crop need)</div>
                </div>""", unsafe_allow_html=True)

                k3.markdown(f"""
                <div class='mcard'>
                    <div class='val'>{best_time}</div>
                    <div class='lbl'>⏰ Best time to water<br>({best_date})</div>
                </div>""", unsafe_allow_html=True)

                k4.markdown(f"""
                <div class='mcard'>
                    <div class='val'>{next_disp}</div>
                    <div class='lbl'>📅 Next watering<br>{next_date}</div>
                </div>""", unsafe_allow_html=True)

                # ── 3. MOTOR RUNNING TIME ──────────────────────────────────────
                if motor_time and pump:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='glass' style='text-align:center; padding:22px;'>
                        <div style='font-size:14px; color:rgba(255,255,255,0.5);
                                    margin-bottom:6px;'>⚙️ Pump Running Time</div>
                        <div style='font-size:44px; font-weight:800;
                                    background:linear-gradient(135deg,#7cb342,#f0b429);
                                    -webkit-background-clip:text;
                                    background-clip:text; color:transparent;'>
                            Approximately {motor_time}
                        </div>
                        <div style='font-size:12px; color:rgba(255,255,255,0.35); margin-top:4px;'>
                            Based on {int(pump)} L/min pump &nbsp;·&nbsp; {irr_size} {irr_unit} farm
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif ref_motor_time and ref_motor_time != "0 minutes":
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class='glass' style='text-align:center; padding:20px; border:1px solid rgba(251,191,36,0.3);'>
                        <div style='font-size:14px; color:#fbbf24; font-weight:700; margin-bottom:4px;'>
                            ⚙️ Pump Running Time (Reference Estimate)
                        </div>
                        <div style='font-size:38px; font-weight:800;
                                    background:linear-gradient(135deg,#f0b429,#ffffff);
                                    -webkit-background-clip:text;
                                    background-clip:text; color:transparent;'>
                            Approximately {ref_motor_time}
                        </div>
                        <div style='font-size:12px; color:rgba(255,255,255,0.65); margin-top:8px; line-height:1.5;'>
                            ⚠️ <strong>Caution:</strong> This runtime is calculated using a standard <strong>100 L/min pump</strong> as a reference benchmark.<br>
                            To calculate the exact motor running time for your farm, please enter your actual pump flow rate (L/min) in Step 2.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class='glass' style='text-align:center; padding:18px; border-color:rgba(255,255,255,0.08);'>
                        <div style='font-size:14px; color:#fbbf24; font-weight:600; margin-bottom:4px;'>
                            ⚙️ Pump Running Time
                        </div>
                        <div style='font-size:13px; color:rgba(255,255,255,0.6);'>
                            Enter pump flow rate (L/min) in Step 2 to calculate exact motor running time.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── 4. WEATHER USED (collapsed) ───────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🌤️ Today's weather data used for this decision"):
                    wc = st.columns(6)
                    wc[0].metric("🌡️ Temp",    f"{weather['temperature']}°C")
                    wc[1].metric("💧 Humidity", f"{weather['relative_humidity']}%")
                    wc[2].metric("🌧️ Rainfall", f"{weather['rainfall']} mm")
                    wc[3].metric("💨 Wind",     f"{weather['wind_speed']} km/h")
                    wc[4].metric("☀️ Solar",    f"{weather['solar_radiation']} W/m²")
                    wc[5].metric("💦 ET₀",      f"{weather['et0']} mm/day")

                with st.expander("📋 View Calculation & AI Reason Details"):
                    st.markdown(f"""
                    • **Decision:** `{irr_status}`
                    • **Reason:** {result.get('reason', 'Based on current weather and crop water demand.')}
                    • **Location:** {result.get('location', irr_city)}
                    • **Crop:** {result.get('crop', irr_crop)} ({result.get('growth_stage', growth_stage)})
                    • **Crop Requirement (ML Target):** {water_mm} mm/day
                    • **Estimated Farm Area Water:** {total_liters:,.0f} L ({water_disp})
                    • **Pump Flow Rate:** {f"{int(pump)} L/min" if pump else "Not provided"}
                    • **Motor Running Time:** {motor_time if motor_time else "Enter pump flow rate to calculate exact motor time"}
                    """)

                # ── 5. MULTI-DAY SCHEDULE ─────────────────────────────────────
                if irr_days > 1:
                    st.markdown("---")
                    st.markdown(f"### 📅 Your {irr_days}-Day Irrigation Schedule")
                    st.caption(
                        "Each day uses its own forecast weather. "
                        f"Today's decision: {irr_status} — "
                        "the schedule reflects forecast recommendations below."
                    )

                    forecast   = get_forecast(lat, lon, days=irr_days)
                    skip_until = result.get("next_irrigation_days", 1)
                    rows       = []
                    show_pump_col = pump is not None and pump > 0

                    for day_idx, f in enumerate(forecast):
                        r_f           = backend_predict_mod.predict(user_input, f)
                        day_crop_need = r_f.get("crop_water_requirement_mm", r_f.get("water_requirement_mm_day", 0.0))
                        day_net_mm    = r_f.get("net_irrigation_mm", day_crop_need)
                        day_lit       = r_f.get("total_water_liters", r_f.get("total_liters", 0.0))
                        day_rain      = f.get("rainfall", 0) or 0
                        day_temp      = f.get("temperature", 28)

                        if day_idx >= skip_until and r_f.get("irrigation_required_today", r_f.get("irrigation_required", False)):
                            action            = "✅ Irrigate"
                            day_next          = r_f.get("next_irrigation_days", 1)
                            skip_until        = day_idx + day_next
                            day_time          = r_f.get("best_irrigation_time", "06:00 AM – 08:00 AM")
                            irr_water_display = f"~{day_lit:,.0f} L" if day_lit > 0 else "0 L"
                        else:
                            action            = "⏭️ Skip"
                            day_time          = "—"
                            irr_water_display = "0 L"

                        # Pump time only for irrigate days and only if pump given
                        if show_pump_col and action == "✅ Irrigate" and day_lit > 0:
                            dm = round(day_lit / pump)
                            if dm >= 60:
                                h = dm // 60; m = dm % 60
                                pump_disp = f"{h}h {m}m" if m else f"{h}h"
                            else:
                                pump_disp = f"{dm} min"
                        elif show_pump_col:
                            pump_disp = "—"
                        else:
                            pump_disp = None

                        rows.append({
                            "date":                  f["date"],
                            "temp":                  day_temp,
                            "rain":                  day_rain,
                            "crop_need":             f"{day_crop_need:.1f} mm",
                            "irrigation_water_disp": irr_water_display,
                            "time":                  day_time,
                            "pump":                  pump_disp,
                            "action":                action,
                        })

                    # Build header using exact clear scientific terminology
                    hdr_labels = ["Date","Temp","Rain","Crop Need","Irrigation Water","Best Time"]
                    if show_pump_col:
                        hdr_labels.append("Pump Time")
                    hdr_labels.append("Action")

                    col_widths = [1.8, 0.9, 0.9, 1.4, 1.8, 1.6]
                    if show_pump_col:
                        col_widths.append(1.3)
                    col_widths.append(1.3)

                    hdr_cols = st.columns(col_widths)
                    for col, lbl in zip(hdr_cols, hdr_labels):
                        col.markdown(
                            f"<div style='font-size:10px; font-weight:700; "
                            f"color:rgba(255,255,255,0.35); letter-spacing:1px; "
                            f"text-transform:uppercase; padding:4px 0 6px;'>"
                            f"{lbl}</div>",
                            unsafe_allow_html=True
                        )

                    for row in rows:
                        act_color = "#7cb342" if "Irrigate" in row["action"] \
                                    else "rgba(255,255,255,0.28)"
                        row_vals  = [
                            row["date"],
                            f"{row['temp']}°C",
                            f"{row['rain']} mm",
                            row["crop_need"],
                            row["irrigation_water_disp"],
                            row["time"],
                        ]
                        if show_pump_col:
                            row_vals.append(row["pump"])

                        row_cols = st.columns(col_widths)
                        for col, val in zip(row_cols[:-1], row_vals):
                            col.markdown(
                                f"<div style='font-size:12px; "
                                f"color:rgba(255,255,255,0.72); "
                                f"padding:9px 0; "
                                f"border-top:1px solid rgba(255,255,255,0.06);'>"
                                f"{val}</div>",
                                unsafe_allow_html=True
                            )
                        row_cols[-1].markdown(
                            f"<div style='font-size:12px; font-weight:700; "
                            f"color:{act_color}; padding:9px 0; "
                            f"border-top:1px solid rgba(255,255,255,0.06);'>"
                            f"{row['action']}</div>",
                            unsafe_allow_html=True
                        )

                # ── 6. SAVE TO SUPABASE ───────────────────────────────────────
                try:
                    soil_row = {}
                    if irr_city:
                        city_matches = CITY_SOIL[CITY_SOIL["City"].astype(str).str.strip().str.lower() == str(irr_city).strip().lower()]
                        if not city_matches.empty:
                            soil_row = city_matches.iloc[0].to_dict()
                    save_irrigation(
                        st.session_state.get("farmer_phone", "guest"),
                        user_input,
                        result,
                        weather,
                        soil_row,
                    )
                except Exception:
                    pass

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
                st.info("Check your internet connection or try a different city.")
# ══════════════════════════════════════════════════════════════════════════════
# YIELD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Yield Estimator":
    render_prediction_breadcrumb(page)
    st.markdown("# 📈 Yield Estimator")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    st.markdown("""
    <div class='glass' style='margin-bottom:24px;'>
        <div style='display:flex; align-items:center; gap:12px;'>
            <div style='font-size:32px;'>📡</div>
            <div>
                <div style='color:#7cb342; font-weight:700; font-size:15px; margin-bottom:4px;'>
                    Tell us your location and crop — weather and soil are fetched automatically
                </div>
                <div style='color:rgba(255,255,255,0.6); font-size:13px;'>
                    Live temperature, rainfall, humidity, ET₀, and soil data are collected
                    from Open-Meteo and your location database. No technical inputs needed.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dropdown values from model's exact categories ─────────────────────────
    YIELD_DISTRICTS = [
        "Adilabad","Anantapur","Chittoor","East Godavari","Guntur",
        "Hyderabad","Kadapa","Karimnagar","Khammam","Krishna","Kurnool",
        "Mahbubnagar","Medak","Nalgonda","Nizamabad","Prakasam",
        "Rangareddi","SPSR Nellore","Srikakulam","Visakhapatnam",
        "Vizianagaram","Warangal","West Godavari",
    ]
    YIELD_SEASONS = ["Kharif", "Rabi", "Whole Year"]
    YIELD_CROPS   = [
        "Arecanut","Arhar/Tur","Bajra","Banana","Beans & Matar (Vegetable)",
        "Bhindi","Bottle Gourd","Brinjal","Cabbage","Cashew Nut","Castor Seed",
        "Citrus Fruit","Coriander","Cotton(lint)","Cowpea(Lobia)","Cucumber",
        "Dry Chillies","Dry Ginger","Garlic","Ginger","Gram","Grapes",
        "Groundnut","Horse-gram","Jowar","Korra","Lemon","Linseed","Maize",
        "Mango","Masoor","Mesta","Moong(Green Gram)","Niger Seed","Onion",
        "Orange","Other Fibres","Other Fresh Fruits","Other Kharif Pulses",
        "Other Misc. Pulses","Other Oilseeds","Other Rabi Pulses",
        "Other Vegetables","Papaya","Peas (Vegetable)","Pome Fruit",
        "Pomegranate","Potato","Ragi","Rapeseed & Mustard","Rice",
        "Safflower","Samai","Sapota","Sesamum","Small Millets","Soyabean",
        "Sugarcane","Sunflower","Sweet Potato","Tapioca","Tobacco","Tomato",
        "Turmeric","Urad","Varagu","Wheat",
    ]

    # ── INPUT FORM & DYNAMIC STATE FILTER ─────────────────────────────────────
    st.markdown("""
    <div class='step-hdr'>🗺️ Select State</div>
    """, unsafe_allow_html=True)
    y_state = st.selectbox(
        "🗺️ State",
        STATES,
        key="yield_state",
    )
    yield_districts = YIELD_DISTRICTS_BY_STATE.get(
        y_state, YIELD_DISTRICTS_BY_STATE["Telangana"]
    )
    if "yield_district" not in st.session_state or st.session_state.yield_district not in yield_districts:
        st.session_state.yield_district = yield_districts[0]

    with st.form("yield_form"):

        st.markdown("""
        <div class='step-hdr'>📍 Farm Details</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            y_district = st.selectbox(
                "📍 District",
                yield_districts,
                key="yield_district",
            )
            y_season   = st.selectbox("🍂 Season",   YIELD_SEASONS,
                                      help="Kharif = June–October  |  Rabi = October–March  |  Whole Year = year-round crop")
        with c2:
            y_crop     = st.selectbox("🌾 Crop",     YIELD_CROPS)
            y_area     = st.number_input(
                "📐 Farm Area (Acres)",
                min_value=0.1, max_value=10000.0, value=2.0, step=0.5,
                help="Enter your farm size in Acres"
            )

        submitted_yield = st.form_submit_button(
            "📈 Estimate My Crop Yield",
            use_container_width=True
        )

    # ── PREDICTION ────────────────────────────────────────────────────────────
    if submitted_yield:
        y_district = _get_valid_district(y_state, y_district, YIELD_DISTRICTS_BY_STATE)
        # Convert acres to hectares for the model
        area_ha = round(y_area * 0.4047, 4)
        crop_emoji = CROP_EMOJI.get(y_crop, "🌾")

        try:
            with st.spinner("📡 Fetching live weather and estimating yield..."):
                result = api_predict_yield(y_district, y_state, y_season, y_crop, float(y_area))

                # Read response values with safe fallbacks
                yld = float(result.get("yield_per_hectare", 0.0))
                total = float(result.get("total_tonnes", 0.0))
                w = result.get("weather", {}) or {}
                if result.get("note"):
                    st.warning(result.get("note"))

                # ── MAIN RESULT ───────────────────────────────────────────────
                st.markdown(f"""
                <div class='rcard' style='padding:18px; display:flex; gap:18px; align-items:center; justify-content:space-between; margin-bottom:24px;'>

                    <div style='display:flex; gap:12px; align-items:center;'>
                        <div style='font-size:54px;'>{crop_emoji}</div>
                        <div style='text-align:left;'>
                            <div style='font-size:13px; color:rgba(255,255,255,0.6); margin-bottom:6px;'>
                                📍 {y_district}, {y_state} &nbsp;|&nbsp; 🍂 {y_season} &nbsp;|&nbsp; 📅 {datetime.now().year}
                            </div>
                            <div style='font-size:14px; color:rgba(255,255,255,0.45); margin-bottom:8px;'>Estimated Yield</div>
                            <div style='font-size:56px; font-weight:800; color:transparent; background:linear-gradient(135deg,#7cb342,#f0b429); -webkit-background-clip:text; background-clip:text;'>
                                {yld:.2f}
                            </div>
                            <div style='font-size:16px; color:rgba(255,255,255,0.75); margin-top:6px;'>tonnes per hectare</div>
                        </div>
                    </div>

                    <div style='min-width:220px; text-align:center;'>
                        <div style='font-size:14px; color:rgba(255,255,255,0.6);'>Total expected</div>
                        <div style='font-size:34px; font-weight:800; color:#43e97b; margin:8px 0;'>{total:.1f} tonnes</div>
                        <div style='font-size:13px; color:rgba(255,255,255,0.55);'>From {y_area} acres ({area_ha:.2f} ha)</div>
                    </div>

                </div>
                """, unsafe_allow_html=True)

                # ── KEY NUMBERS ───────────────────────────────────────────────
                k1, k2, k3 = st.columns(3)

                k1.markdown(f"""
                <div class='mcard'>
                    <div class='val'>{yld:.2f} t/ha</div>
                    <div class='lbl'>📊 Yield per Hectare</div>
                </div>
                """, unsafe_allow_html=True)

                k2.markdown(f"""
                <div class='mcard'>
                    <div class='val'>{total:.1f} tonnes</div>
                    <div class='lbl'>🏭 Total Expected<br>from {y_area} Acres</div>
                </div>
                """, unsafe_allow_html=True)

                k3.markdown(f"""
                <div class='mcard'>
                    <div class='val'>{area_ha:.2f} ha</div>
                    <div class='lbl'>📐 Farm Area<br>({y_area} Acres converted)</div>
                </div>
                """, unsafe_allow_html=True)

                # ── AUTO-FETCHED DATA (collapsed) ─────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📋 Weather & Soil used for this estimate"):
                    st.caption(
                        "These values were fetched automatically from "
                        "Open-Meteo and your location database."
                    )
                    wc = st.columns(5)
                    wc[0].metric("🌡️ Avg Temp",  f"{w['mean_temperature']}°C")
                    wc[1].metric("🌡️ Max Temp",  f"{w['max_temperature']}°C")
                    wc[2].metric("🌡️ Min Temp",  f"{w['min_temperature']}°C")
                    wc[3].metric("🌧️ Rainfall",  f"{w['precipitation']} mm")
                    wc[4].metric("💧 Humidity",   f"{w['relative_humidity']}%")

                    sc = st.columns(5)
                    sc[0].metric("💨 Wind",       f"{w['wind_speed']} km/h")
                    sc[1].metric("💦 ET₀",        f"{w['et0']} mm/day")
                    sc[2].metric("🪣 Soil Moisture",f"{w['soil_moisture']:.3f}")
                    sc[3].metric("🌱 Soil Temp",  f"{w['soil_temperature']}°C")
                    sc[4].metric("📍 Source",     "Open-Meteo API")

                # ── SAVE TO SUPABASE ──────────────────────────────────────────
                try:
                    save_yield(
                        st.session_state.get("farmer_phone", "guest"),
                        y_district,
                        y_state,
                        y_season,
                        y_crop,
                        area_ha,
                        result,
                        result.get("weather"),
                        result.get("soil"),
                    )
                except Exception:
                    pass

        except requests.exceptions.Timeout:
            st.error("⏱️ Weather fetch timed out. Check your internet and try again.")
        except requests.exceptions.ConnectionError:
            st.error("📡 No internet connection. Please connect and try again.")
        except Exception as e:
            st.error(f"❌ Could not estimate yield: {e}")
            st.info(
                "Make sure yield_predict_model.pkl, onehot_encoders.pkl "
                "and state_mapping.pkl are in App/Pickles/Yield/"
            )

# ══════════════════════════════════════════════════════════════════════════════
# MARKET PRICE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰  Market Price":
    render_prediction_breadcrumb(page)
    st.markdown("# 💰 Market Price Prediction")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    st.markdown("""
    <div class='step-hdr'>🗺️ Select State</div>
    """, unsafe_allow_html=True)
    m_state = st.selectbox("🗺️ State", STATES, key="market_state")
    market_districts = MARKET_DISTRICTS_BY_STATE.get(
        m_state, MARKET_DISTRICTS_BY_STATE["Telangana"]
    )
    if "market_district" not in st.session_state or st.session_state.market_district not in market_districts:
        st.session_state.market_district = market_districts[0]

    with st.form("market_form"):
        st.markdown("""
        <div class='step-hdr'>📍 Market Details</div>
        """, unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            m_crop  = st.selectbox("🌾 Commodity (Crop)", CROPS)
            m_district = st.selectbox(
                "📍 District",
                market_districts,
                key="market_district",
            )
        with c2:
            m_qty   = st.number_input("📦 Arrival Quantity (Quintals)",1.0,100000.0,1000.0,100.0)
            m_date  = st.date_input("📅 Market Date",datetime.now())
        submitted5 = st.form_submit_button("💰 Predict Market Price", use_container_width=True)

    if submitted5:
        m_district = _get_valid_district(m_state, m_district, MARKET_DISTRICTS_BY_STATE)
        with st.spinner("Predicting market price..."):
            try:
                quarter = (m_date.month-1)//3+1
                price   = api_predict_market(m_crop, m_state, m_district,
                                             m_date.day, m_date.month, m_date.year,
                                             quarter, m_qty)
                total_val = price * m_qty / 100
                st.markdown(f"""
                <div class='rcard' style='padding:18px; margin-bottom:18px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center; gap:12px;'>

                        <div>
                            <div style='font-size:13px; color:rgba(255,255,255,0.6); margin-bottom:6px;'>📍 {m_district}, {m_state} &nbsp;|&nbsp; 📅 {m_date.strftime('%d %B %Y')}</div>
                            <h2 style='margin:0; font-size:20px;'>💰 Market Price — {m_crop}</h2>
                            <div style='font-size:13px; color:rgba(255,255,255,0.75); margin-top:6px;'>Estimated rate</div>
                        </div>

                        <div style='text-align:right; min-width:220px;'>
                            <div style='font-size:34px; font-weight:800; color:#43e97b;'>₹{price:,.2f}</div>
                            <div style='font-size:13px; color:rgba(255,255,255,0.65);'>per quintal</div>
                        </div>

                    </div>

                    <div style='margin-top:12px; font-size:14px; color:rgba(255,255,255,0.9);'>
                        Expected value of <strong>{m_qty} quintals</strong> at <strong>{m_district}</strong>: <strong style='color:#7cb342;'>₹{total_val:,.2f}</strong>
                    </div>

                </div>
                """,unsafe_allow_html=True)

                # ── SAVE TO SUPABASE ──────────────────────────────────────────
                try:
                    save_market_price(
                        st.session_state.get("farmer_phone", "guest"),
                        m_crop,
                        m_state,
                        m_district,
                        m_date,
                        m_qty,
                        price,
                        quarter,
                    )
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# AGRI FUSION — ALL IN ONE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🚀  Agri Fusion (All-in-One)":

    st.markdown("# 🚜 Agri Fusion — Your Simple Farm Plan")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    # ================================================================
    # HERO
    # ================================================================
    st.markdown(
        """
        <div class="glass" style="
            margin-bottom:28px;
            text-align:center;
            padding:25px;
        ">
            <div style="font-size:45px; margin-bottom:10px;">🌾</div>

                <div style="
                    font-size:21px;
                    font-weight:700;
                    color:#43e97b;
                    margin-bottom:8px;
                ">
                    Your Simple Farm Plan
                </div>

                <div style="
                    color:rgba(255,255,255,0.65);
                    font-size:14px;
                    line-height:1.7;
                ">
                    Tell us where your farm is and how big it is.<br>
                    We will tell you what to plant, when to water,
                    expected harvest and estimated value.
                </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ================================================================
    # FARM INPUT
    # ================================================================
    with st.form("fusion_form"):

        fc1, fc2 = st.columns(2)

        with fc1:
            f_city = st.selectbox(
                "📍 Where is your farm?",
                sorted(list(CITY_COORDS.keys())),
                help="Select the nearest city to your farm",
            )

        with fc2:
            f_acres = st.number_input(
                "🚜 Farm size (Acres)",
                min_value=0.5,
                max_value=5000.0,
                value=2.0,
                step=0.5,
            )

        run_btn = st.form_submit_button(
            "🚀 Create My Farm Plan",
            use_container_width=True,
        )

    # ================================================================
    # RUN FUSION
    # ================================================================
    if run_btn:

        with st.spinner("🔄 Preparing your farm plan..."):

            try:
                R = api_run_fusion(f_city, f_acres)

                if not isinstance(R, dict):
                    st.error("❌ Fusion backend did not return a dictionary.")
                    st.stop()

                # ── SAVE TO SUPABASE ──────────────────────────────────────────
                try:
                    save_fusion_fn = getattr(backend_db, "save_fusion", None)
                    if save_fusion_fn:
                        save_fusion_fn(
                            st.session_state.get("farmer_phone", "guest"),
                            {"city": f_city, "farm_size_acres": f_acres},
                            R
                        )
                except Exception as e:
                    print(f"[Supabase] save_fusion error: {e}")

            except Exception as e:
                st.error(f"❌ Could not create farm plan: {e}")
                st.exception(e)
                st.stop()

        # ============================================================
        # SAFE DATA EXTRACTION
        # ============================================================

        city = R.get("city") or f_city

        season = R.get("season") or "Current season"

        date_value = R.get("date")

        if date_value:
            date_text = str(date_value)
        else:
            date_text = datetime.now().strftime("%d %B %Y")

        farm_ha_value = R.get("farm_ha")

        try:
            farm_ha = (
                float(farm_ha_value)
                if farm_ha_value is not None
                else float(f_acres) * 0.404686
            )
        except (TypeError, ValueError):
            farm_ha = float(f_acres) * 0.404686

        # ============================================================
        # WEATHER
        # ============================================================

        weather = R.get("weather") or {}

        def safe_float(value, default=None):
            try:
                if value is None:
                    return default

                value = float(value)

                if math.isnan(value) or math.isinf(value):
                    return default

                return value

            except (TypeError, ValueError):
                return default

        temperature = safe_float(
            weather.get("temperature"),
            None,
        )

        humidity = safe_float(
            weather.get("relative_humidity"),
            None,
        )

        rainfall = safe_float(
            weather.get("rainfall"),
            0.0,
        )

        wind_speed = safe_float(
            weather.get("wind_speed"),
            None,
        )

        # ============================================================
        # CROP
        # ============================================================

        crop_data = R.get("crop") or {}

        crop = crop_data.get("name") or "Recommendation unavailable"

        confidence = safe_float(
            crop_data.get("confidence"),
            0.0,
        )

        # ============================================================
        # CLIMATE RISK
        # ============================================================

        climate = R.get("climate") or {}

        risk_raw = climate.get("level")

        if risk_raw is None:
            risk = "unknown"
        else:
            risk = str(risk_raw).strip().lower()

            if risk in ("", "none", "nan", "null"):
                risk = "unknown"

        risk_confidence = safe_float(
            climate.get("confidence"),
            0.0,
        )

        rain_7d = safe_float(
            climate.get("rain_7d"),
            0.0,
        )

        dry_days_float = safe_float(
            climate.get("dry_days"),
            0.0,
        )

        dry_days = int(max(0, round(dry_days_float)))

        # ============================================================
        # IRRIGATION
        # ============================================================

        irrigation = R.get("irrigation") or {}

        irrigate_raw = irrigation.get("irrigate", False)

        if isinstance(irrigate_raw, str):
            irrigate = irrigate_raw.strip().lower() in (
                "true",
                "yes",
                "1",
            )
        else:
            irrigate = bool(irrigate_raw)

        mm_day = safe_float(
            irrigation.get("mm_day"),
            0.0,
        )

        liters = safe_float(
            irrigation.get("liters"),
            0.0,
        )

        next_date = irrigation.get("next_date")

        if not next_date:
            next_date = "Not available"

        # 20-litre bucket conversion
        buckets = liters / 20.0

        # ============================================================
        # YIELD
        # ============================================================

        yield_data = R.get("yield") or {}

        yield_per_ha = safe_float(
            yield_data.get("per_ha"),
            0.0,
        )

        total_yield = safe_float(
            yield_data.get("total"),
            0.0,
        )

        total_yield = max(0.0, total_yield)

        total_kg = total_yield * 1000.0
        total_quintals = total_yield * 10.0

        # ============================================================
        # MARKET
        # ============================================================

        market = R.get("market") or {}

        price_per_quintal = safe_float(
            market.get("price_per_quintal"),
            0.0,
        )

        price_per_quintal = max(0.0, price_per_quintal)

        # Always calculate estimated gross value from the displayed
        # harvest and displayed price.
        estimated_value = (
            total_quintals * price_per_quintal
        )

        # ============================================================
        # SOIL PH — SAFE LOOKUP
        # ============================================================

        soil_ph = None

        try:
            if hasattr(CITY_SOIL, "columns") and "City" in CITY_SOIL.columns:

                soil_rows = CITY_SOIL[
                    CITY_SOIL["City"].astype(str).str.strip().str.lower()
                    == str(city).strip().lower()
                ]

                if not soil_rows.empty and "Soil_pH" in soil_rows.columns:

                    soil_ph = safe_float(
                        soil_rows.iloc[0]["Soil_pH"],
                        None,
                    )

        except Exception:
            soil_ph = None

        soil_ph_text = (
            f"{soil_ph:.1f}"
            if soil_ph is not None
            else "Not available"
        )

        # ============================================================
        # DISPLAY TEXT
        # ============================================================

        temperature_text = (
            f"{temperature:.1f}°C"
            if temperature is not None
            else "Not available"
        )

        humidity_text = (
            f"{humidity:.0f}%"
            if humidity is not None
            else "Not available"
        )

        wind_text = (
            f"{wind_speed:.1f} km/h"
            if wind_speed is not None
            else "Not available"
        )

        # ============================================================
        # TOP FARM INFORMATION
        # ============================================================

        st.markdown(
            f"""
            <div class="glass" style="
                text-align:center;
                margin-top:25px;
                margin-bottom:28px;
                padding:20px;
            ">

                <div style="
                    font-size:14px;
                    color:rgba(255,255,255,0.65);
                    margin-bottom:8px;
                ">
                    📍 <b>{city}</b>
                    &nbsp; | &nbsp;
                    🚜 <b>{f_acres:g} Acres</b>
                    &nbsp; | &nbsp;
                    🍂 <b>{season}</b>
                    &nbsp; | &nbsp;
                    📅 <b>{date_text}</b>
                </div>

                <div style="
                    font-size:16px;
                    color:rgba(255,255,255,0.85);
                ">
                    Here is your farm plan for today.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        # ============================================================
        # STEP 1 — CROP
        # ============================================================

        st.markdown("## 1️⃣ 🌱 What should I plant?")

        CROP_EMOJI_MAP = {
            "Rice": "🌾",
            "Maize": "🌽",
            "Cotton": "☁️",
            "Banana": "🍌",
            "Grapes": "🍇",
            "Mango": "🥭",
            "Groundnut": "🥜",
            "Mungbean": "🫘",
            "Mothbeans": "🫘",
            "Watermelon": "🍉",
            "Muskmelon": "🍈",
            "Papaya": "🧡",
            "Chickpea": "🫘",
            "Lentil": "🫘",
            "Sunflower": "🌻",
        }

        crop_emoji = CROP_EMOJI_MAP.get(
            str(crop),
            "🌱",
        )

        crop_col1, crop_col2 = st.columns([1, 2])

        with crop_col1:

            st.markdown(
                f"""
                <div class="rcard" style="text-align:center;">

                    <div style="
                        font-size:60px;
                        margin-bottom:8px;
                    ">
                        {crop_emoji}
                    </div>

                    <div style="
                        font-size:28px;
                        font-weight:800;
                        color:#43e97b;
                    ">
                        {crop}
                    </div>

                    <div style="
                        font-size:13px;
                        color:rgba(255,255,255,0.55);
                        margin-top:6px;
                    ">
                        AI confidence: {confidence:.1f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with crop_col2:

            st.markdown(
                f"""
                <div class="glass">

                    <div style="
                        font-size:17px;
                        font-weight:700;
                        color:#43e97b;
                        margin-bottom:12px;
                    ">
                        🌱 Why this crop?
                    </div>

                    <div style="
                        font-size:14px;
                        line-height:1.9;
                        color:rgba(255,255,255,0.82);
                    ">

                        ✅ Suitable for the
                        <b>{season}</b> season
                        <br>

                        ✅ Suitable for your area:
                        <b>{city}</b>
                        <br>

                        🌡️ Today's temperature:
                        <b>{temperature_text}</b>
                        <br>

                        💧 Today's humidity:
                        <b>{humidity_text}</b>
                        <br>

                        🌱 Soil pH:
                        <b>{soil_ph_text}</b>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )
        # ============================================================
        # FARMER-FRIENDLY WEATHER MESSAGE
        # ============================================================

        # We keep rain_7d and dry_days for backend/model logic,
        # but convert them into a simple message for the farmer.

        if risk in ("high", "extreme"):

            planting_message = (
                f"⚠️ It is better to wait before planting {crop}. "
                "Weather conditions may be risky today."
            )

            planting_status = "WAIT BEFORE PLANTING"

        elif risk == "moderate":

            planting_message = (
                f"⚠️ You can prepare the field for {crop}, "
                "but wait for safer weather before planting."
            )

            planting_status = "BE CAREFUL"

        elif rainfall > 5:

            planting_message = (
                f"🌧️ It is better not to plant {crop} today "
                "because there has been significant rain."
            )

            planting_status = "WAIT TODAY"

        elif rainfall > 0:

            planting_message = (
                f"🌦️ The weather is suitable for planting {crop}, "
                "but check that the soil is not too wet."
            )

            planting_status = "GENERALLY SUITABLE"

        elif rain_7d > 20:

            planting_message = (
                f"🌱 The weather is generally suitable for planting {crop}. "
                "The soil may still have enough moisture from recent rain."
            )

            planting_status = "GOOD TO PLANT"

        elif dry_days >= 5:

            planting_message = (
                f"☀️ The weather has been dry recently. "
                f"Before planting {crop}, check whether the soil has enough moisture."
            )

            planting_status = "CHECK SOIL FIRST"

        else:

            planting_message = (
                f"🌱 Today's weather looks suitable for planting {crop}. "
                "You can prepare and plant if the soil is ready."
            )

            planting_status = "GOOD TO PLANT"
        # ============================================================
        # STEP 2 — FARMER-FRIENDLY WEATHER
        # ============================================================

        st.markdown("---")
        st.markdown("## 2️⃣ 🌦️ Is today a good day to plant?")

        # ------------------------------------------------------------
        # MAIN FARMER MESSAGE
        # ------------------------------------------------------------

        if planting_status == "GOOD TO PLANT":

            st.success(
                f"🌱 **GOOD DAY TO PLANT {crop.upper()}**\n\n"
                f"{planting_message}"
            )

        elif planting_status == "GENERALLY SUITABLE":

            st.info(
                f"🌱 **WEATHER LOOKS SUITABLE**\n\n"
                f"{planting_message}"
            )

        elif planting_status == "CHECK SOIL FIRST":

            st.warning(
                f"🌱 **CHECK THE SOIL BEFORE PLANTING**\n\n"
                f"{planting_message}"
            )

        elif planting_status == "WAIT TODAY":

            st.warning(
                f"🌧️ **WAIT BEFORE PLANTING**\n\n"
                f"{planting_message}"
            )

        elif planting_status == "WAIT BEFORE PLANTING":

            st.error(
                f"⚠️ **WAIT BEFORE PLANTING**\n\n"
                f"{planting_message}"
            )

        else:

            st.warning(
                f"🌦️ **CHECK THE WEATHER BEFORE PLANTING**\n\n"
                f"{planting_message}"
            )


        # ------------------------------------------------------------
        # SIMPLE WEATHER INFORMATION
        # ------------------------------------------------------------

        st.markdown("### Today's weather")

        w1, w2, w3, w4 = st.columns(4)

        with w1:
            st.metric(
                "🌡️ Temperature",
                temperature_text
            )

        with w2:
            st.metric(
                "💧 Humidity",
                humidity_text
            )

        with w3:
            st.metric(
                "🌧️ Rain today",
                f"{rainfall:.1f} mm"
            )

        with w4:
            st.metric(
                "💨 Wind",
                wind_text
            )


        # ------------------------------------------------------------
        # WEATHER EXPLANATION
        # ------------------------------------------------------------

        if rainfall > 5:

            weather_explanation = (
                "There has been noticeable rain today. "
                "Avoid planting in waterlogged soil."
            )

        elif rainfall > 0:

            weather_explanation = (
                "There has been some rain today. "
                "Make sure the soil is not too wet before planting."
            )

        elif dry_days >= 5:

            weather_explanation = (
                "The weather has been dry for several days. "
                "Check the soil moisture before planting."
            )

        elif rain_7d > 20:

            weather_explanation = (
                "Your area received useful rain recently. "
                "The soil may still have enough moisture."
            )

        else:

            weather_explanation = (
                "There is no significant rain today. "
                "The weather looks generally suitable for farm work."
            )


        st.info(
            f"👨‍🌾 **For you:** {weather_explanation}"
        )


        # ------------------------------------------------------------
        # WEATHER RISK
        # ------------------------------------------------------------

        if risk == "unknown":

            risk_display = "Not available"

            st.warning(
                "⚠️ We could not calculate the weather-risk level. "
                "The live weather information is available, "
                "but please also check the local weather forecast."
            )

        elif risk in ("high", "extreme"):

            risk_display = "High"

            st.error(
                "⚠️ **Weather may be risky today.** "
                f"Take extra care of your {crop} crop."
            )

        elif risk == "moderate":

            risk_display = "Moderate"

            st.warning(
                "⚠️ **Weather needs attention today.** "
                "Keep checking your crop and local weather."
            )

        elif risk == "low":

            risk_display = "Low"

            st.success(
                "✅ **Weather conditions look safe today.** "
                f"This is generally suitable for {crop}."
            )

        else:

            risk_display = "Not available"

            st.warning(
                "⚠️ Weather-risk information is currently unavailable."
            )

        # ============================================================
        # STEP 3 — FARMER-FRIENDLY IRRIGATION
        # ============================================================

        st.markdown("---")
        st.markdown("## 3️⃣ 💧 Should I water my crop today?")

        # ------------------------------------------------------------
        # WATER NEEDED
        # ------------------------------------------------------------

        if irrigate and liters > 0:

            # ========================================================
            # YES — WATER TODAY
            # ========================================================

            st.warning(
                f"💧 **YES, WATER YOUR {crop.upper()} CROP TODAY**\n\n"
                f"Your farm needs approximately **{liters:,.0f} litres** of water today."
            )

            # Main water information
            i1, i2 = st.columns(2)

            with i1:

                st.metric(
                    "💧 Water needed",
                    f"{liters:,.0f} litres"
                )

            with i2:

                st.metric(
                    "🪣 20-litre buckets",
                    f"{buckets:,.0f}"
                )

            # --------------------------------------------------------
            # SIMPLE FARMER INSTRUCTION
            # --------------------------------------------------------

            st.info(
                f"👨‍🌾 **What to do:**\n\n"
                f"Give your {crop} crop about **{liters:,.0f} litres of water**.\n\n"
                f"⏰ Best time: **6 AM – 8 AM**"
            )

            # --------------------------------------------------------
            # NEXT WATERING
            # --------------------------------------------------------

            st.success(
                f"📅 **Next planned watering:** {next_date}"
            )

            # --------------------------------------------------------
            # TECHNICAL VALUE — OPTIONAL
            # --------------------------------------------------------

            if mm_day > 0:

                with st.expander("More water information"):

                    st.write(
                        f"Water depth used by the model: **{mm_day:.1f} mm/day**"
                    )

        else:

            # ========================================================
            # NO — DO NOT WATER
            # ========================================================

            st.success(
                f"✅ **NO WATER NEEDED TODAY**\n\n"
                f"Your {crop} crop does not need irrigation today. "
                "Save your water."
            )

            # --------------------------------------------------------
            # SIMPLE INFORMATION
            # --------------------------------------------------------

            i1, i2 = st.columns(2)

            with i1:

                st.metric(
                    "💧 Water needed today",
                    "0 litres"
                )

            with i2:

                st.metric(
                    "🪣 20-litre buckets",
                    "0"
                )

            # --------------------------------------------------------
            # FARMER INSTRUCTION
            # --------------------------------------------------------

            st.info(
                f"👨‍🌾 **What to do:**\n\n"
                f"Do not irrigate your {crop} crop today. "
                "Save the water for the next planned irrigation."
            )

            # --------------------------------------------------------
            # NEXT WATERING
            # --------------------------------------------------------

            st.success(
                f"📅 **Next planned watering:** {next_date}"
            )

        # ============================================================
        # STEP 4 — EXPECTED HARVEST
        # ============================================================

        st.markdown("---")
        st.markdown("## 4️⃣ 🌾 How much harvest can I expect?")

        # ------------------------------------------------------------
        # MAIN HARVEST MESSAGE
        # ------------------------------------------------------------

        st.success(
            f"🌾 **YOUR EXPECTED HARVEST: {total_kg:,.0f} KG**\n\n"
            f"For your **{f_acres:g}-acre farm**, the model estimates "
            f"approximately **{total_kg:,.0f} kg of {crop}**."
        )

        # ------------------------------------------------------------
        # HARVEST CARDS
        # ------------------------------------------------------------

        y1, y2 = st.columns(2)

        with y1:

            st.markdown(
                f"""
                <div class="mcard" style="text-align:center;">

                    <div style="
                        font-size:15px;
                        color:rgba(255,255,255,0.55);
                    ">
                        Expected harvest
                    </div>

                    <div style="
                        font-size:38px;
                        font-weight:800;
                        color:#43e97b;
                        margin:8px 0;
                    ">
                        {total_kg:,.0f} kg
                    </div>

                    <div style="
                        font-size:14px;
                        color:rgba(255,255,255,0.55);
                    ">
                        About {total_quintals:.1f} quintals
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with y2:

            st.markdown(
                f"""
                <div class="mcard" style="text-align:center;">

                    <div style="
                        font-size:15px;
                        color:rgba(255,255,255,0.55);
                    ">
                        Your farm
                    </div>

                    <div style="
                        font-size:38px;
                        font-weight:800;
                        color:#43e97b;
                        margin:8px 0;
                    ">
                        {f_acres:g} acres
                    </div>

                    <div style="
                        font-size:14px;
                        color:rgba(255,255,255,0.55);
                    ">
                        About {farm_ha:.2f} hectares
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ------------------------------------------------------------
        # SIMPLE EXPLANATION
        # ------------------------------------------------------------

        st.info(
            f"👨‍🌾 **What does this mean?**\n\n"
            f"If you grow {crop} on your {f_acres:g}-acre farm, "
            f"the model estimates a harvest of about **{total_kg:,.0f} kg** "
            f"({total_quintals:.1f} quintals)."
        )

        # ------------------------------------------------------------
        # DISCLAIMER
        # ------------------------------------------------------------

        st.warning(
            "🌾 This is only an estimate, not a guaranteed harvest. "
            "Actual harvest can be higher or lower depending on rainfall, "
            "irrigation, seed variety, soil, pests, diseases and farm management."
        )

        # ============================================================
        # STEP 5 — MARKET VALUE
        # ============================================================

        st.markdown("---")
        st.markdown("## 5️⃣ 💰 What could my harvest be worth?")

        m1, m2 = st.columns(2)

        with m1:

            st.markdown(
                f"""
                <div class="mcard" style="text-align:center;">

                    <div style="
                        font-size:15px;
                        color:rgba(255,255,255,0.55);
                    ">
                        Estimated price
                    </div>

                    <div style="
                        font-size:34px;
                        font-weight:800;
                        color:#43e97b;
                        margin:8px 0;
                    ">
                        ₹{price_per_quintal:,.0f}
                    </div>

                    <div style="
                        font-size:13px;
                        color:rgba(255,255,255,0.55);
                    ">
                        per quintal
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:

            st.markdown(
                f"""
                <div class="mcard" style="text-align:center;">

                    <div style="
                        font-size:15px;
                        color:rgba(255,255,255,0.55);
                    ">
                        Estimated harvest value
                    </div>

                    <div style="
                        font-size:34px;
                        font-weight:800;
                        color:#43e97b;
                        margin:8px 0;
                    ">
                        ₹{estimated_value:,.0f}
                    </div>

                    <div style="
                        font-size:13px;
                        color:rgba(255,255,255,0.55);
                    ">
                        before farming expenses
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info(
            f"Calculation: {total_quintals:.1f} quintals × "
            f"₹{price_per_quintal:,.0f} = "
            f"₹{estimated_value:,.0f}"
        )

        st.warning(
            "💰 This is estimated gross value, not profit. "
            "Seeds, fertilizer, labour, irrigation, pesticides, "
            "transport and other expenses are not deducted."
        )

        # ============================================================
        # FINAL FARMER WEATHER DECISION
        # ============================================================

        if planting_status == "GOOD TO PLANT":

            final_weather_summary = (
                f"Good day to plant {crop}."
            )

        elif planting_status == "GENERALLY SUITABLE":

            final_weather_summary = (
                f"Weather is generally suitable for planting {crop}."
            )

        elif planting_status == "CHECK SOIL FIRST":

            final_weather_summary = (
                f"Check soil moisture before planting {crop}."
            )

        elif planting_status == "WAIT TODAY":

            final_weather_summary = (
                f"Wait today before planting {crop}."
            )

        elif planting_status == "WAIT BEFORE PLANTING":

            final_weather_summary = (
                f"Weather is risky. Wait before planting {crop}."
            )

        else:

            final_weather_summary = (
                "Check the local weather before planting."
            )

        # ============================================================
        # WHAT SHOULD THE FARMER DO TODAY?
        # ============================================================

        st.markdown("### 👨‍🌾 What should I do today?")

        st.success(
            f"🌱 Plant: The model recommends {crop}."
        )

        if irrigate and liters > 0:

            st.warning(
                f"💧 Water: Give about "
                f"{liters:,.0f} litres "
                f"({buckets:,.0f} buckets of 20 litres)."
            )

        else:

            st.success(
                "💧 Water: Do not irrigate today. Save your water."
            )

        if risk == "unknown":

            st.warning(
                "🌦️ Weather: Live weather is available, "
                "but the weather-risk prediction is unavailable. "
                "Please also check the local weather forecast."
            )

        elif risk in ("high", "extreme"):

            st.error(
                "🌦️ Weather: Risk is high. "
                "Monitor the crop carefully."
            )

        elif risk == "moderate":

            st.warning(
                "🌦️ Weather: Moderate risk. "
                "Keep checking the weather."
            )

        else:

            st.success(
                "🌦️ Weather: Conditions currently look suitable."
            )

        st.info(
            "🌾 AI predictions are estimates. "
            "Actual crop performance depends on real farm conditions."
        )

        # ============================================================
        # TECHNICAL INFORMATION
        # ============================================================

        with st.expander("🔧 Technical / Developer Information"):

            st.json(R)

# ══════════════════════════════════════════════════════════════════════════════
# FARMER FEEDBACK SECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬  Feedback":
    render_prediction_breadcrumb(page)
    st.markdown("# 💬 Farmer Feedback & Support")
    st.markdown(
        "<div class='disclaimer'>"
        "💡 <strong>We value your feedback!</strong> Your responses help us refine model predictions and improve user experience for farmers across India. "
        "<strong>Note: All fields are optional.</strong>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class='glass' style='margin-bottom:24px;'>
        <div style='display:flex; align-items:center; gap:14px;'>
            <div style='font-size:36px;'>📝</div>
            <div>
                <div style='color:#7cb342; font-weight:700; font-size:16px; margin-bottom:4px;'>
                    Share Your Experience or Report an Issue
                </div>
                <div style='color:rgba(255,255,255,0.7); font-size:13.5px;'>
                    Tell us if you faced any difficulty, saw an incorrect prediction, or have suggestions.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("feedback_form"):
        st.markdown("### ⭐ 1. Overall System Rating")
        rating = st.select_slider(
            "How satisfied are you with Agri Fusion?",
            options=[1, 2, 3, 4, 5],
            value=5,
            format_func=lambda x: {
                1: "⭐ 1 — Needs Improvement",
                2: "⭐⭐ 2 — Fair",
                3: "⭐⭐⭐ 3 — Good",
                4: "⭐⭐⭐⭐ 4 — Very Good",
                5: "⭐⭐⭐⭐⭐ 5 — Excellent"
            }[x],
            help="Select a rating score from 1 to 5 stars (optional)"
        )

        st.markdown("---")
        st.markdown("### ⚡ 2. Difficulties or Usability Issues")
        difficulties_selected = st.multiselect(
            "Did you face any difficulties while using the application?",
            [
                "Slow loading speed / timeout",
                "Difficulty selecting city / location",
                "Understanding scientific terms / water units",
                "Pump time calculation unclear",
                "Mobile layout / screen rendering",
                "Weather data not updating",
            ],
            help="Select any difficulties you encountered (optional)"
        )
        difficulties_text = st.text_area(
            "Details on difficulties faced (optional):",
            placeholder="e.g. Weather data took longer than expected to fetch...",
            height=80
        )

        st.markdown("---")
        st.markdown("### 🤖 3. Model Accuracy Feedback")
        incorrect_model = st.selectbox("Did any specific model provide unexpected or incorrect outputs?", [
            "None / All models worked great",
            "🌾 Crop Recommendation",
            "🌡️ Climate Risk",
            "💧 Irrigation Advisor",
            "📈 Yield Estimator",
            "💰 Market Price",
            "Multiple Models",
        ])
        incorrect_text = st.text_area(
            "If yes, please describe what output was unexpected or incorrect (optional):",
            placeholder="e.g. Crop recommendation suggested Rice, but my soil pH is too low for Rice...",
            height=100
        )

        st.markdown("---")
        st.markdown("### 📱 4. Contact Information (Optional)")
        contact_phone = st.text_input(
            "Your Mobile / WhatsApp Number (if you'd like our support team to contact you):",
            value=st.session_state.get("farmer_phone", ""),
            placeholder="e.g. 9876543210",
            max_chars=12,
            help="Optional — enter your mobile number if you would like us to follow up with you."
        )

        st.markdown("---")
        st.markdown("### 💬 5. Additional Comments & Suggestions")
        general_comments = st.text_area(
            "Any other suggestions or features you would like to see in Agri Fusion (optional):",
            placeholder="e.g. Please add support for regional languages like Telugu / Hindi...",
            height=100
        )

        submit_fb = st.form_submit_button("💬 Submit Feedback", use_container_width=True)

    if submit_fb:
        diff_summary = ", ".join(difficulties_selected)
        if difficulties_text:
            diff_summary = f"{diff_summary} | {difficulties_text}" if diff_summary else difficulties_text

        saved = save_feedback(
            farmer_phone=st.session_state.get("farmer_phone", "guest"),
            rating=rating,
            difficulties=diff_summary,
            incorrect_model=incorrect_model if incorrect_model != "None / All models worked great" else "",
            incorrect_outputs=incorrect_text,
            contact_phone=contact_phone,
            general_comments=general_comments,
        )

        st.balloons()
        st.success("🎉 Thank you! Your feedback has been recorded successfully. We appreciate your input!")