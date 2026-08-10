"""
Crop Recommendation Backend — Location-Only Version (v3)
===========================================================
Matches the ACTUAL feature_columns.pkl for this model:
  temperature, humidity, rainfall, wind_speed,
  soil_ph, nitrogen, organic_carbon, clay, sand, silt, cec,
  season_Monsoon, season_Post-Monsoon, season_Summer, season_Winter,
  city_<16 specific cities>

Farmer input : ONLY a city (restricted to the 16 cities the model was
               actually trained on — a city outside this list can't be
               represented by the one-hot city_* columns).
Weather      : live Open-Meteo (temperature, humidity, rainfall, wind_speed).
               Any missing/failed field falls back to that city's historical
               average from weather.csv. Never raises, never returns None.
Soil         : live SoilGrids (soil_ph, nitrogen, organic_carbon, clay,
               sand, silt, cec) at the city's coordinates. Any missing/failed
               property falls back to a state-level regional default
               (Telangana vs Andhra Pradesh). Never raises, never returns None.
Season       : derived from today's date (Indian meteorological seasons).
"""

import numpy as np
import requests
from datetime import datetime

# ── The 16 cities the model was actually trained on (from feature_columns.pkl
# city_* dummy columns). The city dropdown MUST be restricted to these —
# picking anything else can't be one-hot encoded correctly.
CITY_DATA = {
    "Adilabad":      {"lat": 19.6641, "lon": 78.5320, "state": "Telangana",
                       "temperature_c": 27.4, "humidity_percent": 58.3, "rainfall_mm": 0.14, "wind_speed": 8.4},
    "Anakapalli":    {"lat": 17.6913, "lon": 83.0039, "state": "Andhra Pradesh",
                       "temperature_c": 27.3, "humidity_percent": 75.8, "rainfall_mm": 0.18, "wind_speed": 9.9},
    "Bapatla":       {"lat": 15.9042, "lon": 80.4675, "state": "Andhra Pradesh",
                       "temperature_c": 28.2, "humidity_percent": 75.0, "rainfall_mm": 0.15, "wind_speed": 10.6},
    "Chittoor":      {"lat": 13.2172, "lon": 79.1003, "state": "Andhra Pradesh",
                       "temperature_c": 27.3, "humidity_percent": 66.3, "rainfall_mm": 0.12, "wind_speed": 10.1},
    "Eluru":         {"lat": 16.7107, "lon": 81.0952, "state": "Andhra Pradesh",
                       "temperature_c": 28.3, "humidity_percent": 72.7, "rainfall_mm": 0.16, "wind_speed": 9.2},
    "Hanumakonda":   {"lat": 18.0058, "lon": 79.5570, "state": "Telangana",
                       "temperature_c": 27.6, "humidity_percent": 62.7, "rainfall_mm": 0.13, "wind_speed": 9.2},
    "Hyderabad":     {"lat": 17.3850, "lon": 78.4867, "state": "Telangana",
                       "temperature_c": 26.6, "humidity_percent": 59.9, "rainfall_mm": 0.10, "wind_speed": 9.4},
    "Jagtial":       {"lat": 18.7947, "lon": 78.9166, "state": "Telangana",
                       "temperature_c": 27.4, "humidity_percent": 60.4, "rainfall_mm": 0.13, "wind_speed": 9.0},
    "Kakinada":      {"lat": 16.9891, "lon": 82.2475, "state": "Andhra Pradesh",
                       "temperature_c": 28.0, "humidity_percent": 74.6, "rainfall_mm": 0.14, "wind_speed": 10.6},
    "Kamareddy":     {"lat": 18.3200, "lon": 78.3400, "state": "Telangana",
                       "temperature_c": 25.7, "humidity_percent": 63.0, "rainfall_mm": 0.11, "wind_speed": 10.5},
    "Karimnagar":    {"lat": 18.4386, "lon": 79.1288, "state": "Telangana",
                       "temperature_c": 27.1, "humidity_percent": 63.9, "rainfall_mm": 0.13, "wind_speed": 9.1},
    "Khammam":       {"lat": 17.2473, "lon": 80.1514, "state": "Telangana",
                       "temperature_c": 28.2, "humidity_percent": 66.0, "rainfall_mm": 0.13, "wind_speed": 8.9},
    "Kurnool":       {"lat": 15.8281, "lon": 78.0373, "state": "Andhra Pradesh",
                       "temperature_c": 28.5, "humidity_percent": 57.0, "rainfall_mm": 0.09, "wind_speed": 10.9},
    "Nalgonda":      {"lat": 17.0575, "lon": 79.2684, "state": "Telangana",
                       "temperature_c": 27.9, "humidity_percent": 62.8, "rainfall_mm": 0.11, "wind_speed": 9.4},
    "Tirupati":      {"lat": 13.6288, "lon": 79.4192, "state": "Andhra Pradesh",
                       "temperature_c": 27.9, "humidity_percent": 66.6, "rainfall_mm": 0.11, "wind_speed": 8.0},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh",
                       "temperature_c": 27.5, "humidity_percent": 76.6, "rainfall_mm": 0.17, "wind_speed": 10.1},
}
CITIES = sorted(CITY_DATA.keys())

# ── State-level soil defaults — used per-property whenever SoilGrids omits
# or fails on that specific property, so one missing field never blocks
# the whole request. Typical ranges for AP (coastal alluvial) vs Telangana
# (red/black soils); refine against your real 59-location dataset if you
# want tighter accuracy.
STATE_SOIL_DEFAULTS = {
    "Telangana":      {"soil_ph": 7.2, "nitrogen": 45.0, "organic_carbon": 6.0, "clay": 28.0, "sand": 42.0, "silt": 30.0, "cec": 18.0},
    "Andhra Pradesh": {"soil_ph": 6.6, "nitrogen": 50.0, "organic_carbon": 7.5, "clay": 24.0, "sand": 46.0, "silt": 30.0, "cec": 15.0},
}
_DEFAULT_SOIL = {"soil_ph": 6.8, "nitrogen": 45.0, "organic_carbon": 6.5, "clay": 26.0, "sand": 44.0, "silt": 30.0, "cec": 16.0}

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
    return "Winter"  # Dec, Jan, Feb


def _fetch_live_weather(lat: float, lon: float):
    """Returns dict or None — never raises, never contains None values."""
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
    info = CITY_DATA[city]
    live = _fetch_live_weather(info["lat"], info["lon"])
    if live is not None:
        live["source"] = "live"
        return live
    return {
        "temperature": info["temperature_c"],
        "humidity": info["humidity_percent"],
        "rainfall": info["rainfall_mm"],
        "wind_speed": info["wind_speed"],
        "source": "historical average",
    }


# SoilGrids property name -> our feature name
_SOILGRIDS_MAP = {
    "phh2o": "soil_ph", "nitrogen": "nitrogen", "soc": "organic_carbon",
    "clay": "clay", "sand": "sand", "silt": "silt", "cec": "cec",
}
# SoilGrids returns tenths/hundredths for some properties — divisor to
# convert each to the plain unit the model expects.
_SOILGRIDS_DIVISOR = {
    "soil_ph": 10.0,        # pH*10
    "organic_carbon": 10.0,  # dg/kg -> g/kg-ish scale
    "nitrogen": 100.0,       # cg/kg -> g/kg-ish scale
    # clay, sand, silt, cec are already in usable units (g/kg, cmol/kg)
}


def _fetch_live_soil(lat: float, lon: float) -> dict:
    """Returns whatever properties SoilGrids successfully provided —
    missing/null ones are simply absent from the returned dict, never None."""
    result = {}
    try:
        r = requests.get(
            "https://rest.isric.org/soilgrids/v2.0/properties/query",
            params={
                "lon": lon, "lat": lat,
                "property": list(_SOILGRIDS_MAP.keys()),
                "depth": "0-5cm", "value": "mean",
            },
            timeout=15,
        )
        if not r.ok:
            return result
        layers = r.json().get("properties", {}).get("layers", [])
        for layer in layers:
            name = layer.get("name")
            feature_name = _SOILGRIDS_MAP.get(name)
            if not feature_name:
                continue
            try:
                raw = layer["depths"][0]["values"]["mean"]
            except (KeyError, IndexError, TypeError):
                raw = None
            if raw is None:
                continue  # leave missing — caller fills from regional default
            divisor = _SOILGRIDS_DIVISOR.get(feature_name, 1.0)
            result[feature_name] = round(raw / divisor, 2)
    except Exception:
        pass
    return result


def _get_soil(city: str) -> dict:
    info = CITY_DATA[city]
    defaults = STATE_SOIL_DEFAULTS.get(info["state"], _DEFAULT_SOIL)
    live = _fetch_live_soil(info["lat"], info["lon"])
    soil = dict(defaults)
    soil.update(live)  # only overwrite fields SoilGrids actually returned
    soil["source"] = "live" if len(live) == len(defaults) else (
        "partial live + regional estimate" if live else "regional estimate"
    )
    return soil


def _build_explanation(crop: str, season: str, soil_ph: float, temperature: float, rainfall: float) -> str:
    parts = [
        f"For {season.lower()} conditions in your area (soil pH {soil_ph}, temperature {temperature}°C), "
        f"{crop} is well suited to the current soil and climate profile."
    ]
    if rainfall and rainfall > 0:
        parts.append(f"Rainfall today ({rainfall} mm) supports its growth stage.")
    else:
        parts.append(f"{crop} can be managed with irrigation in the current dry conditions.")
    return " ".join(parts)


def recommend_crop(city: str, model, feature_columns, label_encoder,
                    crop_encoders=None, city_soil_df=None) -> dict:
    """
    model            : M["crop_model"]     — model.pkl
    feature_columns  : M["crop_columns"]   — feature_columns.pkl (ordered list,
                        includes season_* and city_* one-hot columns)
    label_encoder    : M["crop_label_enc"] — label_encoder.pkl
    crop_encoders    : M["crop_encoders"]  — accepted for call-site compatibility;
                        not needed here since season/city are handled as
                        one-hot columns directly from feature_columns.pkl
    city_soil_df     : unused; kept for call-site compatibility
    """
    if city not in CITY_DATA:
        raise ValueError(
            f"Location '{city}' isn't one this model was trained on. "
            f"Please pick one of: {', '.join(CITIES)}"
        )

    weather = _get_weather(city)
    soil = _get_soil(city)
    season = _season_for_today()

    numeric_values = {
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "rainfall": weather["rainfall"],
        "wind_speed": weather["wind_speed"],
        "soil_ph": soil["soil_ph"],
        "nitrogen": soil["nitrogen"],
        "organic_carbon": soil["organic_carbon"],
        "clay": soil["clay"],
        "sand": soil["sand"],
        "silt": soil["silt"],
        "cec": soil["cec"],
    }

    ordered_values = []
    for col in feature_columns:
        if col in numeric_values:
            ordered_values.append(numeric_values[col])
        elif col.startswith("season_"):
            ordered_values.append(1.0 if col == f"season_{season}" else 0.0)
        elif col.startswith("city_"):
            ordered_values.append(1.0 if col == f"city_{city}" else 0.0)
        else:
            raise ValueError(
                f"feature_columns.pkl has a column '{col}' the backend doesn't know how "
                f"to fill. feature_columns.pkl contains: {list(feature_columns)}"
            )

    features = np.array([ordered_values], dtype=float)

    pred = model.predict(features)[0]
    crop_name = str(label_encoder.inverse_transform([pred])[0]).strip().title() \
        if hasattr(label_encoder, "inverse_transform") else str(pred)

    conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        conf = round(float(max(proba)) * 100, 1)

    emoji = CROP_EMOJI.get(crop_name, "🌱")
    care = CROP_INFO.get(crop_name, {}).get("care", _DEFAULT_CARE)
    explanation = _build_explanation(crop_name, season, soil["soil_ph"], weather["temperature"], weather["rainfall"])

    return {
        "recommended_crop": crop_name,
        "crop_emoji": emoji,
        "confidence_percent": conf if conf is not None else "N/A",
        "recommendation": explanation,
        "care_tips": care,
        "season": season,
        "fetched_at": datetime.now().strftime("%d %B %Y, %I:%M %p"),
        "location": {"city": city, "latitude": CITY_DATA[city]["lat"], "longitude": CITY_DATA[city]["lon"]},
        "weather": weather,
        "soil": soil,
    }