import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKLE_DIR = os.path.join(BASE_DIR, "Pickles", "Ir")

model           = joblib.load(os.path.join(PICKLE_DIR, "best_irrigation_model.pkl"))
onehot_encoders = joblib.load(os.path.join(PICKLE_DIR, "onehot_encoders.pkl"))
feature_columns = joblib.load(os.path.join(PICKLE_DIR, "feature_columns.pkl"))

city_soil = pd.read_csv(os.path.join(BASE_DIR, "backend", "city_soil_lookup.csv"))
crop_kc   = pd.read_csv(os.path.join(BASE_DIR, "backend", "crop_kc_lookup.csv"))

STATE_MAP        = {"Telangana": 0, "Andhra Pradesh": 1}
CLIMATE_RISK_MAP = {"Low": 0, "Medium": 1}
GROWTH_STAGE_MAP = {"Initial": 1, "Development": 2, "Mid-season": 3, "Late-season": 4}
KC_BINS          = [0, 0.5, 0.9, 1.2, 2.0]
KC_LABELS        = ["low", "medium", "high", "very_high"]
EFFICIENCY       = {"Drip": 0.90, "Sprinkler": 0.75, "Flood": 0.55}
ACRE_TO_HA       = 0.4047

def get_soil_data(city):
    match = city_soil[city_soil["City"].str.lower() == city.lower()]
    if match.empty:
        match = city_soil[city_soil["City"] == "Hyderabad"]  # safe regional fallback
    return match.iloc[0].to_dict()

# Crops outside crop_kc_lookup.csv's 14-crop coverage (e.g. Chickpea,
# Kidneybeans, Watermelon) fall back to a generic medium-water-demand
# profile instead of crashing the irrigation stage.
_DEFAULT_KC = {"kc": 0.85, "root_depth_m": 0.6}

def get_kc_data(crop, growth_stage):
    row = crop_kc[(crop_kc["Crop"] == crop) & (crop_kc["Growth_Stage"] == growth_stage)]
    if row.empty:
        return dict(_DEFAULT_KC)
    row = row.iloc[0]
    return {"kc": row["Kc"], "root_depth_m": row["Root_Depth_m"]}

def next_irrigation_days(water_mm):
    if water_mm > 10: return 1
    elif water_mm >= 5: return 2
    else: return 3

def predict(user_input, weather):
    soil    = get_soil_data(user_input["city"])
    kc_data = get_kc_data(user_input["crop"], user_input["growth_stage"])

    row = {
        "state":              STATE_MAP[soil["State"]],
        "city":               user_input["city"],
        "temperature":        weather["temperature"],
        "relative_humidity":  weather["relative_humidity"],
        "rainfall":           weather["rainfall"],
        "wind_speed":         weather["wind_speed"],
        "solar_radiation":    weather["solar_radiation"],
        "et0":                weather["et0"],
        "climate_risk_score": soil["Climate_Risk_Score"],
        "climate_risk":       CLIMATE_RISK_MAP[soil["Climate_Risk"]],
        "soil_ph":            soil["Soil_pH"],
        "organic_carbon":     soil["Organic_Carbon"],
        "sand_percentage":    soil["Sand_Percentage"],
        "silt_percentage":    soil["Silt_Percentage"],
        "clay_percentage":    soil["Clay_Percentage"],
        "cec":                soil["CEC"],
        "bulk_density":       soil["Bulk_Density"],
        "field_capacity":     soil["Field_Capacity"],
        "wilting_point":      soil["Wilting_Point"],
        "available_water":    soil["Available_Water"],
        "nitrogen":           soil["Nitrogen"],
        "soil_type":          soil["Soil_Type"],
        "crop":               user_input["crop"],
        "growth_stage":       GROWTH_STAGE_MAP[user_input["growth_stage"]],
        "root_depth_m":       kc_data["root_depth_m"],
        "kc":                 kc_data["kc"],
    }

    df = pd.DataFrame([row])
    df["kc_band"] = pd.cut(df["kc"], bins=KC_BINS, labels=KC_LABELS)
    df = df.drop(columns=["kc"])

    for col, enc in onehot_encoders.items():
        encoded = enc.transform(df[[col]])
        enc_df  = pd.DataFrame(encoded, columns=enc.get_feature_names_out([col]), index=df.index)
        df = pd.concat([df.drop(columns=[col]), enc_df], axis=1)

    df = df.reindex(columns=feature_columns, fill_value=0)

    water_mm   = float(model.predict(df)[0])
    hectares   = user_input["farm_size"] * ACRE_TO_HA if user_input["farm_size_unit"] == "Acres" else user_input["farm_size"]
    liters     = water_mm * hectares * 10000
    eff        = EFFICIENCY.get(user_input.get("irrigation_method"), 1.0)
    adj_liters = liters / eff
    motor_mins = round(adj_liters / user_input["pump_lpm"]) if user_input.get("pump_lpm") else None

    return {
        "water_requirement_mm_day": round(water_mm, 2),
        "total_liters":             round(adj_liters, 1),
        "irrigation_required":      water_mm > 1.5,
        "best_irrigation_time":     "6:00 AM – 8:00 AM",
        "next_irrigation_days":     next_irrigation_days(water_mm),
        "motor_minutes":            motor_mins,
    }