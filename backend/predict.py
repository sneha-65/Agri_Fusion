import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, timezone

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
ACRE_TO_M2       = 4046.8564224
IST              = timezone(timedelta(hours=5, minutes=30))

def get_soil_data(city):
    match = city_soil[city_soil["City"].astype(str).str.lower() == str(city).strip().lower()]
    if match.empty:
        match = city_soil[city_soil["City"] == "Hyderabad"]
    return match.iloc[0].to_dict()

_DEFAULT_KC = {"kc": 0.85, "root_depth_m": 0.6}

def get_kc_data(crop, growth_stage):
    row = crop_kc[(crop_kc["Crop"] == crop) & (crop_kc["Growth_Stage"] == growth_stage)]
    if row.empty:
        return dict(_DEFAULT_KC)
    row = row.iloc[0]
    return {"kc": row["Kc"], "root_depth_m": row["Root_Depth_m"]}

def calculate_effective_rainfall(rainfall_mm: float) -> float:
    """
    Agricultural Effective Rainfall calculation (USDA SCS / FAO method).
    Rainfall below 1mm is mostly intercepted by canopy or evaporated.
    """
    p = max(0.0, float(rainfall_mm))
    if p <= 1.0:
        return 0.0
    elif p <= 8.3:
        return p * 0.65
    else:
        return 5.4 + (p - 8.3) * 0.75

def next_irrigation_days(net_water_mm: float, rainfall_mm: float = 0.0) -> int:
    eff_rain = calculate_effective_rainfall(rainfall_mm)
    if net_water_mm <= 0.5:
        return 3 if eff_rain > 10.0 else 2
    elif net_water_mm > 6.0:
        return 1
    elif net_water_mm >= 3.0:
        return 2
    else:
        return 3

def format_motor_runtime(minutes: float) -> str:
    if minutes is None or minutes <= 0:
        return None
    mins = int(round(minutes))
    if mins < 1:
        mins = 1
    h = mins // 60
    m = mins % 60
    if h > 0 and m > 0:
        return f"{h} hours {m} minutes" if h > 1 else f"1 hour {m} minutes"
    elif h > 0:
        return f"{h} hours" if h > 1 else "1 hour"
    else:
        return f"{m} minutes"

def _safe_float(val, default):
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default

def predict(user_input: dict, weather: dict) -> dict:
    """
    Scientific Irrigation Backend Engine.
    Timezone aware (Asia/Kolkata - IST).
    Separates Gross Crop Water Requirement (ETc) from Net Irrigation Requirement.
    Deducts effective rainfall.
    """
    city = user_input.get("city", "Hyderabad")
    crop = user_input.get("crop", "Rice")
    growth_stage = user_input.get("growth_stage", "Development")
    farm_size = _safe_float(user_input.get("farm_size"), 2.0)
    farm_unit = user_input.get("farm_size_unit", "Acres")
    pump_lpm_raw = user_input.get("pump_lpm")
    already_irrigated = bool(user_input.get("already_irrigated_today") or user_input.get("already_watered", False))
    irrigation_method = user_input.get("irrigation_method")

    pump_lpm = _safe_float(pump_lpm_raw, 0.0)
    if pump_lpm <= 0:
        pump_lpm = None

    soil    = get_soil_data(city)
    kc_data = get_kc_data(crop, growth_stage)

    temp_val = _safe_float(weather.get("temperature"), 27.5)
    humidity_val = _safe_float(weather.get("relative_humidity"), 65.0)
    rain_val = _safe_float(weather.get("rainfall"), 0.0)

    row = {
        "state":              STATE_MAP.get(soil.get("State"), 0),
        "city":               city,
        "temperature":        temp_val,
        "relative_humidity":  humidity_val,
        "rainfall":           rain_val,
        "wind_speed":         _safe_float(weather.get("wind_speed"), 9.5),
        "solar_radiation":    _safe_float(weather.get("solar_radiation"), 550.0),
        "et0":                _safe_float(weather.get("et0"), 3.5),
        "climate_risk_score": soil.get("Climate_Risk_Score", 0),
        "climate_risk":       CLIMATE_RISK_MAP.get(soil.get("Climate_Risk"), 0),
        "soil_ph":            soil.get("Soil_pH", 6.5),
        "organic_carbon":     soil.get("Organic_Carbon", 0.5),
        "sand_percentage":    soil.get("Sand_Percentage", 40.0),
        "silt_percentage":    soil.get("Silt_Percentage", 30.0),
        "clay_percentage":    soil.get("Clay_Percentage", 30.0),
        "cec":                soil.get("CEC", 15.0),
        "bulk_density":       soil.get("Bulk_Density", 1.3),
        "field_capacity":     soil.get("Field_Capacity", 25.0),
        "wilting_point":      soil.get("Wilting_Point", 12.0),
        "available_water":    soil.get("Available_Water", 13.0),
        "nitrogen":           soil.get("Nitrogen", 120.0),
        "soil_type":          soil.get("Soil_Type", "Loam"),
        "crop":               crop,
        "growth_stage":       GROWTH_STAGE_MAP.get(growth_stage, 2),
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

    # 1. Gross Crop Water Requirement (ETc predicted by ML model in mm/day)
    crop_water_req_mm = float(model.predict(df)[0])

    # 2. Effective Rainfall Calculation (mm)
    eff_rainfall_mm = calculate_effective_rainfall(rain_val)

    # 3. Net Irrigation Requirement (mm/day)
    net_irrigation_mm = max(0.0, crop_water_req_mm - eff_rainfall_mm)

    # 4. Gross Irrigation Depth accounting for irrigation method efficiency
    eff = EFFICIENCY.get(irrigation_method, 1.0) if (irrigation_method and irrigation_method != "Not specified") else 1.0
    gross_irrigation_mm = net_irrigation_mm / eff if eff > 0 else net_irrigation_mm

    # 5. Farm Area Conversion (m²)
    # 1 acre = 4046.8564224 m² | 1 hectare = 10,000 m²
    if farm_unit == "Acres":
        area_m2 = farm_size * ACRE_TO_M2
    else:
        area_m2 = farm_size * 10000.0

    # 6. Irrigation Water Volume in Liters
    # 1 mm over 1 m² = 1 Liter
    irrigation_liters = gross_irrigation_mm * area_m2
    total_crop_demand_liters = crop_water_req_mm * area_m2  # Total gross crop ETc demand for comparison

    # 7. Motor Running Time (calculated for user pump_lpm or 100 L/min reference)
    if pump_lpm and pump_lpm > 0 and irrigation_liters > 0:
        motor_mins = int(round(irrigation_liters / pump_lpm))
        motor_running_time = format_motor_runtime(motor_mins)
        ref_motor_running_time = motor_running_time
        motor_time_exact = True
    elif irrigation_liters > 0:
        motor_mins = int(round(irrigation_liters / 100.0))
        ref_motor_running_time = format_motor_runtime(motor_mins)
        motor_running_time = None
        motor_time_exact = False
    else:
        motor_mins = None
        ref_motor_running_time = None
        motor_running_time = None
        motor_time_exact = False

    # ── DECISION ENGINE (Asia/Kolkata IST Timezone Aware) ─────────────────────
    now_ist = datetime.now(IST)
    current_h = now_ist.hour + now_ist.minute / 60.0

    # Determine if irrigation is actually required today (Net requirement > 0.5 mm)
    is_irrigation_needed = net_irrigation_mm > 0.5

    MORNING_WINDOW = "06:00 AM – 08:00 AM"
    EVENING_WINDOW = "04:30 PM – 06:30 PM"

    if already_irrigated:
        irrigation_required_today = False
        irrigation_status = "WATERING COMPLETED TODAY"
        farmer_message = "✅ Already Irrigated Today. No more irrigation is recommended today."
        best_time = MORNING_WINDOW
        best_date = now_ist.strftime("%d %B %Y")
        next_days = 2
        reason = "You indicated that watering is already completed for today."

    elif not is_irrigation_needed:
        irrigation_required_today = False
        irrigation_status = "NO IRRIGATION NEEDED TODAY"
        if rain_val > 1.0:
            farmer_message = f"🌧️ Rainfall is expected to provide sufficient water ({rain_val} mm rain). Irrigation can be skipped today."
            reason = f"Effective rainfall ({round(eff_rainfall_mm, 1)} mm) covers the crop water requirement ({round(crop_water_req_mm, 1)} mm/day)."
        else:
            farmer_message = "🌧️ No Irrigation Needed Today. Your crop has sufficient water."
            reason = f"Estimated crop water need ({round(crop_water_req_mm, 1)} mm/day) is low for current soil/weather conditions."
        best_time = "—"
        best_date = now_ist.strftime("%d %B %Y")
        next_days = next_irrigation_days(net_irrigation_mm, rain_val)

    else: # Irrigation is required today and farmer has not irrigated today
        irrigation_required_today = True

        if current_h < 8.0:
            irrigation_status = "IRRIGATION REQUIRED TODAY"
            farmer_message = f"💧 Irrigation Required Today. Water your crop during the morning window ({MORNING_WINDOW})."
            best_time = MORNING_WINDOW
            best_date = now_ist.strftime("%d %B %Y")
            next_days = next_irrigation_days(net_irrigation_mm, rain_val)
            reason = "Optimal morning irrigation window."

        elif 8.0 <= current_h < 16.5:
            irrigation_status = "IRRIGATION REQUIRED TODAY"
            farmer_message = f"💧 Today's morning window has passed. Water your crop during the evening window ({EVENING_WINDOW})."
            best_time = EVENING_WINDOW
            best_date = now_ist.strftime("%d %B %Y")
            next_days = next_irrigation_days(net_irrigation_mm, rain_val)
            reason = "Today's recommended morning window passed; evening window is available."

        elif 16.5 <= current_h <= 18.5:
            irrigation_status = "IRRIGATION REQUIRED TODAY"
            farmer_message = f"💧 You are inside the evening window ({EVENING_WINDOW}) — irrigate now."
            best_time = EVENING_WINDOW
            best_date = now_ist.strftime("%d %B %Y")
            next_days = next_irrigation_days(net_irrigation_mm, rain_val)
            reason = "Optimal evening irrigation window."

        else: # Past 06:30 PM (18.5)
            irrigation_required_today = False
            irrigation_status = "IRRIGATE TOMORROW MORNING"
            farmer_message = f"🌅 Today's best irrigation time has passed. Water first thing tomorrow morning ({MORNING_WINDOW})."
            best_time = f"Tomorrow, {MORNING_WINDOW}"
            next_days = 1
            best_date = (now_ist + timedelta(days=1)).strftime("%d %B %Y")
            reason = "Today's recommended morning and evening irrigation windows have passed."

    next_date_dt = now_ist + timedelta(days=next_days)
    next_irrigation_date_str = next_date_dt.strftime("%d %B %Y")
    next_irrigation_display = "In 1 day" if next_days == 1 else f"In {next_days} days"

    # Farmer-friendly display formatting
    if irrigation_liters > 0:
        water_disp_val = round(irrigation_liters, -3)
        if water_disp_val == 0:
            water_disp_val = round(irrigation_liters)
        water_display_str = f"Approximately {water_disp_val:,.0f} liters"
    else:
        water_display_str = "0 liters (Skip irrigation)"

    reason_explanation = (
        f"Your crop is estimated to need approximately {round(crop_water_req_mm, 1)} mm of water today. "
        f"{f'Rainfall ({rain_val} mm) provides ~{round(eff_rainfall_mm, 1)} mm of effective water. ' if rain_val > 0 else 'No effective rainfall was recorded today. '}"
        f"{f'After subtracting rainfall, the estimated additional irrigation depth needed is {round(net_irrigation_mm, 1)} mm (~{round(irrigation_liters):,.0f} L for your {farm_size} {farm_unit} farm).' if is_irrigation_needed else 'Available rainfall is sufficient to cover crop water needs today.'}"
    )

    weather_summary = {
        "temperature": f"{temp_val}°C",
        "humidity": f"{humidity_val}%",
        "rainfall_forecast": f"{rain_val} mm"
    }

    return {
        "crop":                         crop,
        "growth_stage":                 growth_stage,
        "location":                     city,
        "crop_water_requirement_mm":    round(crop_water_req_mm, 2),
        "water_requirement_mm_day":     round(crop_water_req_mm, 2),  # backward compatibility
        "effective_rainfall_mm":        round(eff_rainfall_mm, 2),
        "net_irrigation_mm":            round(net_irrigation_mm, 2),
        "gross_irrigation_mm":          round(gross_irrigation_mm, 2),
        "total_water_liters":           round(irrigation_liters),
        "total_liters":                 round(irrigation_liters, 1),   # backward compatibility
        "crop_demand_liters":           round(total_crop_demand_liters),
        "irrigation_required_today":    irrigation_required_today,
        "irrigation_required":          irrigation_required_today,  # backward compatibility
        "irrigation_status":            irrigation_status,
        "farmer_message":               farmer_message,
        "water_display":                water_display_str,
        "best_irrigation_time":         best_time,
        "best_irrigation_date":         best_date,
        "next_irrigation_days":         next_days,
        "days_until_next_irrigation":   next_days,
        "next_irrigation_date":         next_irrigation_date_str,
        "next_irrigation":              next_irrigation_date_str,
        "next_irrigation_display":      next_irrigation_display,
        "motor_running_time":           motor_running_time,
        "reference_motor_running_time": ref_motor_running_time,
        "motor_minutes":                motor_mins,
        "motor_time_exact":             motor_time_exact,
        "pump_flow_lpm":                pump_lpm,
        "already_irrigated_today":      already_irrigated,
        "reason":                       reason,
        "reason_explanation":           reason_explanation,
        "weather_summary":              weather_summary,
        "recommendation":               farmer_message,
        "time_status":                  "done_today" if already_irrigated else ("not_needed" if not is_irrigation_needed else ("wait_tomorrow" if "TOMORROW" in irrigation_status else "irrigate_now")),
        "irrigation_recommendation": {
            "crop_water_need_mm": f"{round(crop_water_req_mm, 2)} mm/day",
            "net_irrigation_mm": f"{round(net_irrigation_mm, 2)} mm",
            "water_required_liters_per_day": f"{round(irrigation_liters):,} Liters",
            "pumping_duration_hours_per_day": motor_running_time if motor_running_time else "Enter pump flow rate to calculate motor running time.",
            "next_irrigation_recommendation": f"{next_irrigation_date_str} ({next_irrigation_display})"
        }
    }