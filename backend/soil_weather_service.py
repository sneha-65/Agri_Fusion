"""
Soil and Weather Service
Provides auto-fetching of live weather data via Open-Meteo API
and regional soil characteristics from city_soil_lookup.csv.
Automates all technical input fields for farmers.
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITY_SOIL_PATH = os.path.join(BASE_DIR, "backend", "city_soil_lookup.csv")

CITY_SOIL_DF = pd.read_csv(CITY_SOIL_PATH)

CITY_COORDS = {
    "Adilabad": (19.66, 78.53), "Anakapalli": (17.69, 83.00),
    "Bapatla": (15.90, 80.47), "Chittoor": (13.22, 79.10),
    "Eluru": (16.71, 81.09), "Hanumakonda": (17.99, 79.59),
    "Hyderabad": (17.38, 78.47), "Jagtial": (18.79, 78.91),
    "Jangaon": (17.72, 79.15), "Kakinada": (16.98, 82.24),
    "Kamareddy": (18.32, 78.34), "Karimnagar": (18.43, 79.13),
    "Khammam": (17.25, 80.15), "Kurnool": (15.83, 78.04),
    "Mahabubabad": (17.60, 80.00), "Mahabubnagar": (16.74, 77.98),
    "Mancherial": (18.87, 79.46), "Medak": (18.04, 78.26),
    "Mulugu": (18.19, 80.00), "Nagarkurnool": (16.48, 78.32),
    "Nalgonda": (17.05, 79.27), "Nandyal": (15.47, 78.48),
    "Narayanpet": (16.74, 77.49), "Nirmal": (19.10, 78.35),
    "Srikakulam": (18.30, 83.90), "Tirupati": (13.63, 79.42),
    "Visakhapatnam": (17.68, 83.22), "Vizianagaram": (18.10, 83.40),
}

SOIL_NPK_DEFAULTS = {
    "Black Soil (Clayey)": {"N": 75, "P": 45, "K": 55},
    "Alluvial Soil":        {"N": 65, "P": 40, "K": 45},
    "Loamy Soil":           {"N": 60, "P": 42, "K": 40},
    "Default":              {"N": 65, "P": 42, "K": 45},
}

def get_city_soil(city_name: str) -> dict:
    """Retrieve soil parameters for a given city."""
    df_match = CITY_SOIL_DF[CITY_SOIL_DF["City"].str.lower() == city_name.lower()]
    if df_match.empty:
        row = CITY_SOIL_DF[CITY_SOIL_DF["City"] == "Hyderabad"].iloc[0].to_dict()
    else:
        row = df_match.iloc[0].to_dict()
    
    soil_type = row.get("Soil_Type", "Default")
    npk_base = SOIL_NPK_DEFAULTS.get(soil_type, SOIL_NPK_DEFAULTS["Default"])
    
    n_val = round(row.get("Nitrogen", 1.5) * 40, 1) if row.get("Nitrogen") else npk_base["N"]
    
    row["N_est"] = n_val
    row["P_est"] = npk_base["P"]
    row["K_est"] = npk_base["K"]
    return row

def fetch_live_weather(lat: float, lon: float) -> dict:
    """Fetch live weather data from Open-Meteo API with fallback."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "current": "temperature_2m,relative_humidity_2m,precipitation,surface_pressure,cloud_cover,wind_speed_10m,wind_direction_10m,shortwave_radiation",
            "hourly": "et0_fao_evapotranspiration,soil_moisture_0_to_1cm,soil_temperature_0cm",
            "forecast_days": 1,
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        
        curr = data.get("current", {})
        hourly = data.get("hourly", {})
        
        temp = curr.get("temperature_2m", 28.0)
        humid = curr.get("relative_humidity_2m", 65.0)
        precip = curr.get("precipitation", 0.0)
        wind = curr.get("wind_speed_10m", 10.0)
        solar = curr.get("shortwave_radiation", 350.0)
        
        et0_list = hourly.get("et0_fao_evapotranspiration", [4.0])
        et0 = et0_list[0] if et0_list and et0_list[0] is not None else 4.0
        
        sm_list = hourly.get("soil_moisture_0_to_1cm", [0.3])
        soil_moist = sm_list[0] if sm_list and sm_list[0] is not None else 0.3
        
        st_list = hourly.get("soil_temperature_0cm", [temp - 3])
        soil_temp = st_list[0] if st_list and st_list[0] is not None else (temp - 3)
        
        return {
            "temperature": round(temp, 1),
            "relative_humidity": round(humid, 1),
            "precipitation": round(precip, 1),
            "rainfall": round(precip, 1),
            "wind_speed": round(wind, 1),
            "solar_radiation": round(solar, 1),
            "et0": round(et0, 2),
            "soil_moisture": round(soil_moist, 3),
            "soil_temperature": round(soil_temp, 1),
            "heat_index": round(temp + (humid / 100) * 2, 1),
        }
    except Exception as e:
        print(f"Weather API fallback used due to: {e}")
        return {
            "temperature": 28.0,
            "relative_humidity": 65.0,
            "precipitation": 0.0,
            "rainfall": 0.0,
            "wind_speed": 12.0,
            "solar_radiation": 400.0,
            "et0": 4.2,
            "soil_moisture": 0.3,
            "soil_temperature": 25.0,
            "heat_index": 30.0,
        }

def get_farmer_unified_payload(city_name: str, crop: str = "Rice", farm_size_acres: float = 2.0, growth_stage: str = "Development") -> dict:
    """
    Builds a complete, auto-filled payload for all 5 ML models using just city, crop, farm size, and growth stage.
    """
    lat, lon = CITY_COORDS.get(city_name, (17.38, 78.47))
    soil = get_city_soil(city_name)
    weather = fetch_live_weather(lat, lon)
    
    return {
        "city": city_name,
        "state": soil.get("State", "Telangana"),
        "crop": crop,
        "farm_size_acres": farm_size_acres,
        "growth_stage": growth_stage,
        "latitude": lat,
        "longitude": lon,
        "soil": soil,
        "weather": weather,
    }
