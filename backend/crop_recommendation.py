"""
Crop Recommendation Backend — Enhanced Multi-Feature Version (v4)
===================================================================
Uses universal agronomic & meteorological features:
  temperature, humidity, rainfall, wind_speed,
  soil_ph, nitrogen, organic_carbon, clay, sand, silt, cec,
  season_Monsoon, season_Post-Monsoon, season_Summer, season_Winter

Supports ALL 28+ cities across Telangana & Andhra Pradesh, with graceful fallback
and full support for custom weather & soil overrides.
"""

import os
import sys
import numpy as np
import pandas as pd
import requests
import joblib
from datetime import datetime

# ── Automatically ensure enhanced pickles exist ──────────────────────────────
def _ensure_model_trained():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pik_dir = os.path.join(base_dir, "Pickles", "Crop")
    model_path = os.path.join(pik_dir, "model.pkl")
    cols_path = os.path.join(pik_dir, "feature_columns.pkl")
    enc_path = os.path.join(pik_dir, "crop_encoders.pkl")
    
    need_retrain = False
    if not os.path.exists(model_path) or not os.path.exists(cols_path) or not os.path.exists(enc_path):
        need_retrain = True
    else:
        try:
            encs = joblib.load(enc_path)
            if encs.get("force_id") != "v9_unscaled_raw_features_fix":
                need_retrain = True
        except Exception:
            need_retrain = True

    if need_retrain:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(base_dir), "Data", "Crop"))
        if data_dir not in sys.path:
            sys.path.insert(0, data_dir)
        import importlib
        import generate_crop_dataset
        importlib.reload(generate_crop_dataset)
        df = generate_crop_dataset.generate_dataset(samples_per_crop=200)
        df.to_csv(os.path.join(data_dir, "crop_recommendation_enhanced.csv"), index=False)
        df.to_csv(os.path.join(data_dir, "crop_recommendation_raw.csv"), index=False)
        
        import train_crop_model
        importlib.reload(train_crop_model)
        train_crop_model.run_pipeline(verbose=False)


try:
    _ensure_model_trained()
except Exception:
    pass


def get_latest_crop_model_artifacts():
    _ensure_model_trained()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pik_dir = os.path.join(base_dir, "Pickles", "Crop")
    return {
        "model": joblib.load(os.path.join(pik_dir, "model.pkl")),
        "feature_columns": joblib.load(os.path.join(pik_dir, "feature_columns.pkl")),
        "label_encoder": joblib.load(os.path.join(pik_dir, "label_encoder.pkl")),
        "crop_encoders": joblib.load(os.path.join(pik_dir, "crop_encoders.pkl")),
    }


# ── Full 28 City Metadata with Coordinates & Seasonal Rain Baselines ─────────
CITY_DATA = {
    "Adilabad":      {"lat": 19.6641, "lon": 78.5320, "state": "Telangana", "temp": 27.4, "humidity": 58.3, "rain_monthly": 160.0, "wind": 8.4},
    "Anakapalli":    {"lat": 17.6913, "lon": 83.0039, "state": "Andhra Pradesh", "temp": 27.3, "humidity": 75.8, "rain_monthly": 180.0, "wind": 9.9},
    "Bapatla":       {"lat": 15.9042, "lon": 80.4675, "state": "Andhra Pradesh", "temp": 28.2, "humidity": 75.0, "rain_monthly": 170.0, "wind": 10.6},
    "Chittoor":      {"lat": 13.2172, "lon": 79.1003, "state": "Andhra Pradesh", "temp": 27.3, "humidity": 66.3, "rain_monthly": 110.0, "wind": 10.1},
    "Eluru":         {"lat": 16.7107, "lon": 81.0952, "state": "Andhra Pradesh", "temp": 28.3, "humidity": 72.7, "rain_monthly": 165.0, "wind": 9.2},
    "Hanumakonda":   {"lat": 18.0058, "lon": 79.5570, "state": "Telangana", "temp": 27.6, "humidity": 62.7, "rain_monthly": 135.0, "wind": 9.2},
    "Hyderabad":     {"lat": 17.3850, "lon": 78.4867, "state": "Telangana", "temp": 26.6, "humidity": 59.9, "rain_monthly": 115.0, "wind": 9.4},
    "Jagtial":       {"lat": 18.7947, "lon": 78.9166, "state": "Telangana", "temp": 27.4, "humidity": 60.4, "rain_monthly": 140.0, "wind": 9.0},
    "Jangaon":       {"lat": 17.7214, "lon": 79.1601, "state": "Telangana", "temp": 27.5, "humidity": 61.5, "rain_monthly": 120.0, "wind": 9.1},
    "Kakinada":      {"lat": 16.9891, "lon": 82.2475, "state": "Andhra Pradesh", "temp": 28.0, "humidity": 74.6, "rain_monthly": 175.0, "wind": 10.6},
    "Kamareddy":     {"lat": 18.3200, "lon": 78.3400, "state": "Telangana", "temp": 25.7, "humidity": 63.0, "rain_monthly": 125.0, "wind": 10.5},
    "Karimnagar":    {"lat": 18.4386, "lon": 79.1288, "state": "Telangana", "temp": 27.1, "humidity": 63.9, "rain_monthly": 145.0, "wind": 9.1},
    "Khammam":       {"lat": 17.2473, "lon": 80.1514, "state": "Telangana", "temp": 28.2, "humidity": 66.0, "rain_monthly": 140.0, "wind": 8.9},
    "Kurnool":       {"lat": 15.8281, "lon": 78.0373, "state": "Andhra Pradesh", "temp": 28.5, "humidity": 57.0, "rain_monthly": 75.0, "wind": 10.9},
    "Mahabubabad":   {"lat": 17.5977, "lon": 80.0032, "state": "Telangana", "temp": 27.8, "humidity": 64.0, "rain_monthly": 130.0, "wind": 9.0},
    "Mahabubnagar":  {"lat": 16.7488, "lon": 77.9856, "state": "Telangana", "temp": 27.2, "humidity": 60.5, "rain_monthly": 110.0, "wind": 9.3},
    "Mancherial":    {"lat": 18.8679, "lon": 79.4639, "state": "Telangana", "temp": 27.6, "humidity": 61.0, "rain_monthly": 150.0, "wind": 8.8},
    "Medak":         {"lat": 18.0454, "lon": 78.2618, "state": "Telangana", "temp": 26.5, "humidity": 62.0, "rain_monthly": 120.0, "wind": 9.5},
    "Mulugu":        {"lat": 18.1914, "lon": 79.9431, "state": "Telangana", "temp": 27.7, "humidity": 65.0, "rain_monthly": 155.0, "wind": 9.0},
    "Nagarkurnool":  {"lat": 16.4854, "lon": 78.3047, "state": "Telangana", "temp": 27.0, "humidity": 61.0, "rain_monthly": 105.0, "wind": 9.2},
    "Nalgonda":      {"lat": 17.0575, "lon": 79.2684, "state": "Telangana", "temp": 27.9, "humidity": 62.8, "rain_monthly": 115.0, "wind": 9.4},
    "Nandyal":       {"lat": 15.4781, "lon": 78.4836, "state": "Andhra Pradesh", "temp": 28.0, "humidity": 63.0, "rain_monthly": 85.0, "wind": 9.8},
    "Narayanpet":    {"lat": 16.7410, "lon": 77.4984, "state": "Telangana", "temp": 27.1, "humidity": 59.5, "rain_monthly": 100.0, "wind": 9.5},
    "Nirmal":        {"lat": 19.0964, "lon": 78.3426, "state": "Telangana", "temp": 27.3, "humidity": 59.0, "rain_monthly": 145.0, "wind": 8.7},
    "Srikakulam":    {"lat": 18.2949, "lon": 83.8938, "state": "Andhra Pradesh", "temp": 27.8, "humidity": 76.0, "rain_monthly": 175.0, "wind": 10.2},
    "Tirupati":      {"lat": 13.6288, "lon": 79.4192, "state": "Andhra Pradesh", "temp": 27.9, "humidity": 66.6, "rain_monthly": 115.0, "wind": 8.0},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh", "temp": 27.5, "humidity": 76.6, "rain_monthly": 185.0, "wind": 10.1},
    "Vizianagaram":  {"lat": 18.1124, "lon": 83.3978, "state": "Andhra Pradesh", "temp": 27.6, "humidity": 75.0, "rain_monthly": 170.0, "wind": 10.0},
}
CITIES = sorted(list(CITY_DATA.keys()))

_DEFAULT_CITY = {
    "lat": 17.3850, "lon": 78.4867, "state": "Telangana", 
    "temp": 27.0, "humidity": 65.0, "rain_monthly": 120.0, "wind": 9.0
}

STATE_SOIL_DEFAULTS = {
    "Telangana":      {"soil_ph": 7.0, "nitrogen": 85.0, "organic_carbon": 6.5, "clay": 35.0, "sand": 38.0, "silt": 27.0, "cec": 22.0},
    "Andhra Pradesh": {"soil_ph": 6.7, "nitrogen": 90.0, "organic_carbon": 7.8, "clay": 30.0, "sand": 42.0, "silt": 28.0, "cec": 19.0},
}
_DEFAULT_SOIL = {"soil_ph": 6.8, "nitrogen": 85.0, "organic_carbon": 6.5, "clay": 30.0, "sand": 42.0, "silt": 28.0, "cec": 20.0}

CROP_EMOJI = {
    "Rice": "🌾", "Maize": "🌽", "Chickpea": "🫘", "Kidneybeans": "🫘",
    "Pigeonpeas": "🫘", "Mothbeans": "🫘", "Mungbean": "🫘", "Blackgram": "🌰",
    "Lentil": "🫘", "Pomegranate": "🔴", "Banana": "🍌", "Mango": "🥭",
    "Grapes": "🍇", "Watermelon": "🍉", "Muskmelon": "🍈", "Apple": "🍎",
    "Orange": "🍊", "Papaya": "🟠", "Coconut": "🥥", "Cotton": "☁️",
    "Jute": "🌿", "Coffee": "☕",
}

CROP_INFO = {
    "Rice":        {"care": "Keep 5–10 cm standing water during vegetative stage; transplant 20–25 day old seedlings; apply N in 3 splits."},
    "Maize":       {"care": "Sow at 60x20 cm spacing; earth-up at knee-high stage; watch for fall armyworm."},
    "Chickpea":    {"care": "Avoid waterlogging; light irrigation at flowering & pod-filling only; seed-treat with Rhizobium."},
    "Kidneybeans": {"care": "Provide staking for pole types; keep soil moist but not waterlogged during flowering."},
    "Pigeonpeas":  {"care": "Wide spacing (60–90 cm rows); very drought-hardy once established; watch for pod borer."},
    "Mothbeans":   {"care": "Ideal for sandy, low-rainfall tracts; minimal irrigation needed; good as an intercrop."},
    "Mungbean":    {"care": "Short duration (60–65 days); irrigate at flowering and pod formation; avoid excess N."},
    "Blackgram":   {"care": "Good for rotation after rice; light, frequent irrigation; harvest as pods mature."},
    "Lentil":      {"care": "Sow as a winter (rabi) crop; one irrigation at pre-flowering is usually enough."},
    "Pomegranate": {"care": "Drip irrigation preferred; prune annually; watch for bacterial blight in humid spells."},
    "Banana":      {"care": "Needs consistent moisture — mulch to retain it; heavy feeder, fertilize monthly."},
    "Mango":       {"care": "Withhold irrigation before flowering to induce blooming; deep but infrequent watering."},
    "Grapes":      {"care": "Needs trellising; prune twice a year; drip irrigation avoids fungal disease."},
    "Watermelon":  {"care": "Wide spacing for vine spread; reduce watering as fruit ripens to boost sweetness."},
    "Muskmelon":   {"care": "Needs warm soil to germinate; mulch to conserve moisture; avoid overhead watering."},
    "Apple":       {"care": "Needs winter chilling hours; prune for open canopy; suited to cooler hill regions only."},
    "Orange":      {"care": "Avoid waterlogging (root rot risk); regular but moderate irrigation; watch for citrus canker."},
    "Papaya":      {"care": "Very sensitive to waterlogging — raised beds help; stake young plants against wind."},
    "Coconut":     {"care": "Deep watering weekly if no rain; mulch basin; apply organic manure twice a year."},
    "Cotton":      {"care": "Needs a dry spell at boll-opening; scout regularly for bollworm and whitefly."},
    "Jute":        {"care": "Needs standing water tolerance; retting after harvest needs a clean water source nearby."},
    "Coffee":      {"care": "Needs shade trees and cool elevation; mulch to retain moisture; prune after harvest."},
}
_DEFAULT_CARE = "General good practice: monitor soil moisture and watch for pests weekly."


def _season_for_today() -> str:
    month = datetime.now().month
    if month in (6, 7, 8, 9):
        return "Monsoon"
    if month in (10, 11):
        return "Post-Monsoon"
    if month in (3, 4, 5):
        return "Summer"
    return "Winter"


def _fetch_live_weather(lat: float, lon: float):
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "timezone": "auto",
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "forecast_days": 1,
        }, timeout=8)
        if not r.ok:
            return None
        c = r.json().get("current", {})
        vals = {
            "temperature": c.get("temperature_2m"),
            "humidity": c.get("relative_humidity_2m"),
            "rainfall": c.get("precipitation"),
            "wind_speed": c.get("wind_speed_10m"),
        }
        if any(v is None for v in vals.values()):
            return None
        return {k: round(float(v), 2) for k, v in vals.items()}
    except Exception:
        return None


def _get_weather(city: str) -> dict:
    info = CITY_DATA.get(city, _DEFAULT_CITY)
    live = _fetch_live_weather(info["lat"], info["lon"])
    base_rain = info.get("rain_monthly", 120.0)
    if live is not None:
        daily_p = live["rainfall"]
        monthly_p = base_rain + (daily_p * 10.0)
        return {
            "temperature": live["temperature"],
            "humidity": live["humidity"],
            "rainfall": round(monthly_p, 1),
            "wind_speed": live["wind_speed"],
            "source": "live weather + seasonal baseline",
        }
    return {
        "temperature": info["temp"],
        "humidity": info["humidity"],
        "rainfall": base_rain,
        "wind_speed": info["wind"],
        "source": "historical regional average",
    }


def _get_soil(city: str, city_soil_df: pd.DataFrame = None) -> dict:
    # 1. Try loading from city_soil_lookup.csv if available
    if city_soil_df is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "city_soil_lookup.csv")
        if os.path.exists(csv_path):
            try:
                city_soil_df = pd.read_csv(csv_path)
            except Exception:
                city_soil_df = None

    if city_soil_df is not None and not city_soil_df.empty:
        match = city_soil_df[city_soil_df["City"].astype(str).str.lower() == str(city).strip().lower()]
        if not match.empty:
            row = match.iloc[0].to_dict()
            n_raw = float(row.get("Nitrogen", 1.5))
            n_val = n_raw * 45.0 if n_raw < 10.0 else n_raw
            oc_raw = float(row.get("Organic_Carbon", 12.0))
            oc_val = float(row.get("Organic_Carbon", 12.0))

            return {
                "soil_ph": round(float(row.get("Soil_pH", 6.8)), 2),
                "nitrogen": round(float(n_val), 2),
                "organic_carbon": round(float(oc_val), 2),
                "clay": round(float(row.get("Clay_Percentage", 30.0)), 2),
                "sand": round(float(row.get("Sand_Percentage", 40.0)), 2),
                "silt": round(float(row.get("Silt_Percentage", 30.0)), 2),
                "cec": round(float(row.get("CEC", 20.0)), 2),
                "source": f"city lookup ({row.get('Soil_Type', 'Soil')})",
            }

    info = CITY_DATA.get(city, _DEFAULT_CITY)
    defaults = STATE_SOIL_DEFAULTS.get(info["state"], _DEFAULT_SOIL)
    soil = dict(defaults)
    soil["source"] = "regional state default"
    return soil


def _build_explanation(crop: str, season: str, soil_ph: float, temperature: float, rainfall: float) -> str:
    parts = [
        f"For {season.lower()} conditions in your location (soil pH {soil_ph:.1f}, temperature {temperature:.1f}°C), "
        f"{crop} is exceptionally well suited to the current soil composition and climate profile."
    ]
    if rainfall and rainfall >= 100:
        parts.append(f"Abundant water availability (~{rainfall:.0f} mm/month) fully supports its growth cycle.")
    elif rainfall and rainfall >= 50:
        parts.append(f"Moderate water availability (~{rainfall:.0f} mm/month) satisfies its growth requirements.")
    else:
        parts.append(f"Light water availability (~{rainfall:.0f} mm/month) makes drought-resilient crops ideal.")
    return " ".join(parts)


def recommend_crop(city: str, model, feature_columns, label_encoder,
                    crop_encoders=None, city_soil_df=None,
                    weather_override: dict = None, soil_override: dict = None,
                    season_override: str = None) -> dict:
    """
    Enhanced Multi-Feature Crop Recommendation Engine.
    Accurately recommends crops based on temperature, humidity, rainfall, soil pH,
    nitrogen, soil texture (clay/sand/silt), CEC, organic carbon, and season.
    """
    city_info = CITY_DATA.get(city, _DEFAULT_CITY)
    weather = _get_weather(city)
    soil = _get_soil(city, city_soil_df=city_soil_df)
    season = season_override if season_override else _season_for_today()

    # Apply overrides if provided (for custom weather/soil input UI or API)
    if weather_override:
        for k, v in weather_override.items():
            if v is not None:
                weather[k] = float(v)
    if soil_override:
        for k, v in soil_override.items():
            if v is not None:
                soil[k] = float(v)

    # Scale daily rainfall override to monthly equivalent if user passed < 20 mm
    model_rainfall = weather["rainfall"]
    if model_rainfall < 20.0:
        base_rain = city_info.get("rain_monthly", 120.0)
        model_rainfall = base_rain + (model_rainfall * 10.0)

    # Map numeric values matching model trained feature names
    numeric_values = {
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "rainfall": model_rainfall,
        "wind_speed": weather["wind_speed"],
        "soil_ph": soil["soil_ph"],
        "nitrogen": soil["nitrogen"],
        "organic_carbon": soil["organic_carbon"],
        "clay": soil["clay"],
        "sand": soil["sand"],
        "silt": soil["silt"],
        "cec": soil["cec"],
    }

    # Build input vector matching feature_columns
    ordered_values = []
    for col in feature_columns:
        if col in numeric_values:
            ordered_values.append(numeric_values[col])
        elif col.startswith("season_"):
            ordered_values.append(1.0 if col == f"season_{season}" else 0.0)
        elif col.startswith("city_"):
            ordered_values.append(1.0 if col == f"city_{city}" else 0.0)
        else:
            ordered_values.append(0.0)

    features = np.array([ordered_values], dtype=float)

    pred = model.predict(features)[0]
    crop_name = str(label_encoder.inverse_transform([pred])[0]).strip().title() \
        if hasattr(label_encoder, "inverse_transform") else str(pred)

    conf = None
    alternatives = []
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        conf = round(float(max(proba)) * 100, 1)
        
        # Get top alternatives
        top_indices = np.argsort(proba)[::-1]
        for idx in top_indices[1:4]:
            alt_name = str(label_encoder.inverse_transform([idx])[0]).strip().title() \
                if hasattr(label_encoder, "inverse_transform") else str(idx)
            alt_conf = round(float(proba[idx]) * 100, 1)
            if alt_conf > 0.5:
                alternatives.append({"crop": alt_name, "confidence": alt_conf, "emoji": CROP_EMOJI.get(alt_name, "🌱")})

    emoji = CROP_EMOJI.get(crop_name, "🌱")
    care = CROP_INFO.get(crop_name, {}).get("care", _DEFAULT_CARE)
    explanation = _build_explanation(crop_name, season, soil["soil_ph"], weather["temperature"], model_rainfall)

    return {
        "recommended_crop": crop_name,
        "crop_emoji": emoji,
        "confidence_percent": conf if conf is not None else "N/A",
        "recommendation": explanation,
        "care_tips": care,
        "season": season,
        "alternative_crops": alternatives,
        "fetched_at": datetime.now().strftime("%d %B %Y, %I:%M %p"),
        "location": {"city": city, "latitude": city_info["lat"], "longitude": city_info["lon"]},
        "weather": weather,
        "soil": soil,
    }