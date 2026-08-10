"""
Yield Estimator Backend
========================
Farmer inputs : State, District, Season, Crop, Farm Area
Auto-fetched  : All weather (Open-Meteo) + soil (city_soil_lookup)
Model expects : 114 features (21 numeric + 93 one-hot)
"""

import requests
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime

# ── District → lat/lon ───────────────────────────────────────────────────────
DISTRICT_COORDS = {
    "Adilabad":       (19.66, 78.53), "Anantapur":     (14.68, 77.60),
    "Chittoor":       (13.22, 79.10), "East Godavari": (16.98, 82.24),
    "Guntur":         (16.30, 80.43), "Hyderabad":     (17.38, 78.47),
    "Kadapa":         (14.47, 78.82), "Karimnagar":    (18.43, 79.13),
    "Khammam":        (17.25, 80.15), "Krishna":       (16.71, 81.09),
    "Kurnool":        (15.83, 78.04), "Mahbubnagar":   (16.74, 77.98),
    "Medak":          (18.04, 78.26), "Nalgonda":      (17.05, 79.27),
    "Nizamabad":      (18.67, 78.10), "Prakasam":      (15.33, 79.57),
    "Rangareddi":     (17.40, 78.50), "SPSR Nellore":  (14.44, 79.99),
    "Srikakulam":     (18.30, 83.90), "Visakhapatnam": (17.68, 83.22),
    "Vizianagaram":   (18.10, 83.40), "Warangal":      (17.99, 79.59),
    "West Godavari":  (16.91, 81.34),
}

DISTRICT_ELEVATION = {
    "Adilabad":271,"Anantapur":331,"Chittoor":270,"East Godavari":9,
    "Guntur":32,"Hyderabad":536,"Kadapa":154,"Karimnagar":230,
    "Khammam":85,"Krishna":12,"Kurnool":303,"Mahbubnagar":503,
    "Medak":508,"Nalgonda":371,"Nizamabad":388,"Prakasam":18,
    "Rangareddi":540,"SPSR Nellore":15,"Srikakulam":12,
    "Visakhapatnam":45,"Vizianagaram":66,"Warangal":255,"West Godavari":12,
}

STATE_MAP = {"Andhra Pradesh": 0, "Telangana": 1}
# encoder order: season, district, crop (from notebook cell 153)
SEASON_MAP  = {"Kharif": "kharif", "Rabi": "rabi", "Whole Year": "whole year"}

# Soil fallback per district if not in city_soil_lookup
SOIL_DEFAULTS = {"soil_ph":6.5,"organic_carbon":12.0,"clay":30.0,"sand":40.0,"silt":30.0}


def _fetch_weather(lat, lon):
    """Fetch weather from Open-Meteo — all weather features the model needs.
    Falls back to a safe regional default on any network failure (no
    internet, DNS failure, timeout) instead of crashing the yield pipeline."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "timezone": "auto",
            "current": (
                "temperature_2m,relative_humidity_2m,precipitation,"
                "shortwave_radiation,wind_speed_10m"
            ),
            "hourly": "et0_fao_evapotranspiration,soil_moisture_0_to_1cm,soil_temperature_0cm",
            "daily":  "temperature_2m_max,temperature_2m_min",
            "forecast_days": 1,
        }, timeout=12)
        r.raise_for_status()
        d = r.json()
        c = d["current"]; h = d["hourly"]; dy = d["daily"]
        return {
            "mean_temperature":  round(c["temperature_2m"], 1),
            "max_temperature":   round(dy["temperature_2m_max"][0], 1),
            "min_temperature":   round(dy["temperature_2m_min"][0], 1),
            "precipitation":     round(c["precipitation"], 2),
            "shortwave_radiation": round(c["shortwave_radiation"], 1),
            "wind_speed":        round(c["wind_speed_10m"], 1),
            "relative_humidity": round(c["relative_humidity_2m"], 1),
            "et0":               round(h["et0_fao_evapotranspiration"][0] or 0, 3),
            "soil_moisture":     round(h["soil_moisture_0_to_1cm"][0] or 0.25, 4),
            "soil_temperature":  round(h["soil_temperature_0cm"][0] or c["temperature_2m"]-3, 2),
        }
    except (requests.exceptions.RequestException, KeyError, IndexError, TypeError):
        return {
            "mean_temperature": 27.5, "max_temperature": 31.5, "min_temperature": 23.5,
            "precipitation": 0.13, "shortwave_radiation": 550.0, "wind_speed": 9.5,
            "relative_humidity": 65.0, "et0": 3.5, "soil_moisture": 0.25, "soil_temperature": 24.5,
        }


def predict_yield(district, state, season, crop, area, model, encoder,
                  state_map, city_soil_df):
    """
    Full pipeline — farmer provides district, state, season, crop, area.
    Returns dict with yield + all intermediate values for display.
    """
    district_key = district
    lat, lon = DISTRICT_COORDS.get(district_key, (17.38, 78.47))
    elev     = DISTRICT_ELEVATION.get(district_key, 200)
    year     = datetime.now().year

    # Step 1: Fetch weather automatically
    weather = _fetch_weather(lat, lon)

    # Step 2: Get soil from lookup (match by district name or closest city)
    soil = SOIL_DEFAULTS.copy()
    match = city_soil_df[city_soil_df["City"].str.lower() == district_key.lower()]
    if len(match) == 0:
        # try partial match
        match = city_soil_df[city_soil_df["City"].str.lower().str.contains(
            district_key.lower().split()[0])]
    if len(match) > 0:
        row = match.iloc[0]
        soil = {
            "soil_ph":        row.get("Soil_pH", 6.5),
            "organic_carbon": row.get("Organic_Carbon", 12.0),
            "clay":           row.get("Clay_Percentage", 30.0),
            "sand":           row.get("Sand_Percentage", 40.0),
            "silt":           row.get("Silt_Percentage", 30.0),
        }

    # Step 3: Build numeric feature row (21 columns, exact notebook order)
    numeric = {
        "state":              state_map.get(state, 0),
        "latitude":           lat,
        "longitude":          lon,
        "year":               year,
        "area":               area,
        "mean_temperature":   weather["mean_temperature"],
        "max_temperature":    weather["max_temperature"],
        "min_temperature":    weather["min_temperature"],
        "precipitation":      weather["precipitation"],
        "shortwave_radiation":weather["shortwave_radiation"],
        "wind_speed":         weather["wind_speed"],
        "relative_humidity":  weather["relative_humidity"],
        "et0":                weather["et0"],
        "soil_moisture":      weather["soil_moisture"],
        "soil_temperature":   weather["soil_temperature"],
        "soil_ph":            soil["soil_ph"],
        "organic_carbon":     soil["organic_carbon"],
        "clay":               soil["clay"],
        "sand":               soil["sand"],
        "silt":               soil["silt"],
        "elevation":          elev,
    }
    X_num = pd.DataFrame([numeric])

    # Step 4: One-hot encode season, district, crop (exact training order)
    season_enc   = SEASON_MAP.get(season, season.lower())
    district_enc = district_key.lower()
    crop_enc     = crop.lower()

    cat_df  = pd.DataFrame([[season_enc, district_enc, crop_enc]],
                            columns=["season", "district", "crop"])
    ohe_arr = encoder.transform(cat_df)
    ohe_cols= encoder.get_feature_names_out(["season", "district", "crop"])
    X_ohe   = pd.DataFrame(ohe_arr, columns=ohe_cols)

    # Step 5: Combine and align to 114 features
    X = pd.concat([X_num.reset_index(drop=True), X_ohe.reset_index(drop=True)], axis=1)
    X = X.reindex(columns=list(range(model.n_features_in_)), fill_value=0)

    # Step 6: Predict
    yld_per_ha   = float(model.predict(X)[0])
    total_tonnes = yld_per_ha * area

    return {
        "yield_per_hectare": round(yld_per_ha, 2),
        "total_tonnes":      round(total_tonnes, 2),
        "district":          district,
        "state":             state,
        "crop":              crop,
        "season":            season,
        "area":              area,
        "weather":           weather,
        "soil":              soil,
    }