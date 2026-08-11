"""
Agri Fusion Pipeline v2
=========================
Farmer inputs : City + Farm Size only
Chain         : Crop Rec -> Climate Risk -> Irrigation -> Yield -> Market Price
Each model's output feeds the next as input.

Unlike v1, this version calls your REAL, dedicated backend modules for each
model (crop_recommendation.py, climate_risk.py, predict.py, yield_estimator.py)
instead of re-implementing their encoding/feature-alignment logic inline —
one proven pipeline per model, so fusion can't silently drift out of sync
with what each model was actually trained on.

Every network-dependent stage (weather/soil fetches inside the imported
modules) already has its own safe fallback — see the patched versions of
climate_risk.py, weather.py, and yield_estimator.py. A stage failing for
any OTHER reason (bad lookup, missing mapping) is caught here so the whole
fusion run doesn't crash — you get partial results with a clear "unavailable"
note instead of a stack trace.
"""

import os
import joblib
import pandas as pd
from datetime import datetime

import backend.weather as weather_mod
import backend.predict as irrigation_mod
import backend.yield_estimator as yield_mod
import backend.climate_risk as climate_mod
import backend.crop_recommendation as crop_mod

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIK = os.path.join(BASE, "Pickles")


def _load():
    return {
        "crop_model":     joblib.load(os.path.join(PIK, "Crop", "model.pkl")),
        "crop_columns":   joblib.load(os.path.join(PIK, "Crop", "feature_columns.pkl")),
        "crop_label_enc": joblib.load(os.path.join(PIK, "Crop", "label_encoder.pkl")),
        "crop_encoders":  joblib.load(os.path.join(PIK, "Crop", "crop_encoders.pkl")),

        "climate_model":  joblib.load(os.path.join(PIK, "Climate", "climate_risk_model.pkl")),
        "climate_enc":    joblib.load(os.path.join(PIK, "Climate", "encoders.pkl")),

        "yield_model":    joblib.load(os.path.join(PIK, "Yield", "yield_predict_model.pkl")),
        "yield_enc":      joblib.load(os.path.join(PIK, "Yield", "onehot_encoders.pkl")),
        "yield_state_map":joblib.load(os.path.join(PIK, "Yield", "state_mapping.pkl")),

        "market_model":   joblib.load(os.path.join(PIK, "Market", "market_price_rf_model.pkl")),
        "market_maps":    joblib.load(os.path.join(PIK, "Market", "Market_Label_Mappings.pkl")),
    }


_MODELS = None
def get_models():
    global _MODELS
    if _MODELS is None:
        _MODELS = _load()
    return _MODELS


# ── City metadata — shared across all stages ────────────────────────────────
CITY_META = {
    "Adilabad":      {"state": "Telangana",      "lat": 19.66, "lon": 78.53, "elev": 271, "district": "Adilabad"},
    "Anakapalli":    {"state": "Andhra Pradesh",  "lat": 17.69, "lon": 83.00, "elev": 27,  "district": "Visakhapatnam"},
    "Bapatla":       {"state": "Andhra Pradesh",  "lat": 15.90, "lon": 80.47, "elev": 8,   "district": "Guntur"},
    "Chittoor":      {"state": "Andhra Pradesh",  "lat": 13.22, "lon": 79.10, "elev": 270, "district": "Chittor"},
    "Eluru":         {"state": "Andhra Pradesh",  "lat": 16.71, "lon": 81.09, "elev": 20,  "district": "West Godavari"},
    "Hanumakonda":   {"state": "Telangana",       "lat": 17.99, "lon": 79.59, "elev": 255, "district": "Hanumakonda"},
    "Hyderabad":     {"state": "Telangana",       "lat": 17.38, "lon": 78.47, "elev": 536, "district": "Nalgonda"},
    "Jagtial":       {"state": "Telangana",       "lat": 18.79, "lon": 78.91, "elev": 223, "district": "Karimnagar"},
    "Jangaon":       {"state": "Telangana",       "lat": 17.72, "lon": 79.16, "elev": 240, "district": "Hanumakonda"},
    "Kakinada":      {"state": "Andhra Pradesh",  "lat": 16.98, "lon": 82.24, "elev": 9,   "district": "SPSR Nellore"},
    "Kamareddy":     {"state": "Telangana",       "lat": 18.32, "lon": 78.34, "elev": 388, "district": "Nalgonda"},
    "Karimnagar":    {"state": "Telangana",       "lat": 18.43, "lon": 79.13, "elev": 230, "district": "Karimnagar"},
    "Khammam":       {"state": "Telangana",       "lat": 17.25, "lon": 80.15, "elev": 85,  "district": "Khammam"},
    "Kurnool":       {"state": "Andhra Pradesh",  "lat": 15.83, "lon": 78.04, "elev": 303, "district": "Kurnool"},
    "Mahabubabad":   {"state": "Telangana",       "lat": 17.60, "lon": 80.00, "elev": 170, "district": "Khammam"},
    "Mahabubnagar":  {"state": "Telangana",       "lat": 16.75, "lon": 77.99, "elev": 498, "district": "Nalgonda"},
    "Mancherial":    {"state": "Telangana",       "lat": 18.87, "lon": 79.46, "elev": 150, "district": "Adilabad"},
    "Medak":         {"state": "Telangana",       "lat": 18.05, "lon": 78.26, "elev": 442, "district": "Nalgonda"},
    "Mulugu":        {"state": "Telangana",       "lat": 18.19, "lon": 79.94, "elev": 180, "district": "Hanumakonda"},
    "Nagarkurnool":  {"state": "Telangana",       "lat": 16.49, "lon": 78.30, "elev": 450, "district": "Nalgonda"},
    "Nalgonda":      {"state": "Telangana",       "lat": 17.05, "lon": 79.27, "elev": 371, "district": "Nalgonda"},
    "Nandyal":       {"state": "Andhra Pradesh",  "lat": 15.48, "lon": 78.48, "elev": 203, "district": "Kurnool"},
    "Narayanpet":    {"state": "Telangana",       "lat": 16.74, "lon": 77.50, "elev": 430, "district": "Nalgonda"},
    "Nirmal":        {"state": "Telangana",       "lat": 19.10, "lon": 78.34, "elev": 240, "district": "Adilabad"},
    "Srikakulam":    {"state": "Andhra Pradesh",  "lat": 18.29, "lon": 83.89, "elev": 32,  "district": "Visakhapatnam"},
    "Tirupati":      {"state": "Andhra Pradesh",  "lat": 13.63, "lon": 79.42, "elev": 182, "district": "Chittor"},
    "Visakhapatnam": {"state": "Andhra Pradesh",  "lat": 17.68, "lon": 83.22, "elev": 45,  "district": "Visakhapatnam"},
    "Vizianagaram":  {"state": "Andhra Pradesh",  "lat": 18.11, "lon": 83.40, "elev": 63,  "district": "Visakhapatnam"},
}
_DEFAULT_CITY_META = {"state": "Telangana", "lat": 17.38, "lon": 78.47, "elev": 536, "district": "Nalgonda"}


ACRE_HA = 0.4047
MARKET_CROP_MAP = {"Rice": "Rice", "Maize": "Maize", "Cotton": "Cotton", "Banana": "Banana",
                    "Grapes": "Grapes", "Mango": "Mango", "Watermelon": "Water Melon",
                    "Muskmelon": "Karbuja(Musk Melon)", "Papaya": "Papaya",
                    "Mothbeans": "Black Gram Dal(Urd Dal)", "Mungbean": "Black Gram Dal(Urd Dal)",
                    "Groundnut": "Black Gram Dal(Urd Dal)", "Chickpea": "Black Gram Dal(Urd Dal)"}


def _predict_market_price(city, crop, arrival_qty_quintals, M):
    """No dedicated market backend module was provided, so this stays here.
    Mirrors exactly how Market_prd.ipynb built its training rows."""
    maps = M["market_maps"]
    meta = CITY_META[city]
    today = datetime.now()
    mkt_crop = MARKET_CROP_MAP.get(crop, "Rice")

    commodity_id = maps["commodity_mapping"].get(mkt_crop)
    district_id = maps["district_mapping"].get(meta["district"])
    state_id = maps["state_mapping"].get(meta["state"])
    if commodity_id is None or district_id is None or state_id is None:
        raise ValueError(
            f"Market model has no price data for crop '{mkt_crop}' / district "
            f"'{meta['district']}' — it only covers the 14 commodities and 46 "
            f"districts it was trained on."
        )

    row = pd.DataFrame([{
        "Commodity": commodity_id, "State": state_id, "District": district_id,
        "Day": today.day, "Month": today.month, "Year": today.year,
        "Quarter": (today.month - 1) // 3 + 1, "Arrival_Quantity": arrival_qty_quintals,
    }])
    price = float(M["market_model"].predict(row)[0])
    return {
        "market_crop": mkt_crop,
        "price_per_quintal": round(price, 2),
        "arrival_qty_quintals": arrival_qty_quintals,
        "total_value_inr": round(price * arrival_qty_quintals / 100, 2),
    }


def run_fusion(city: str, farm_size_acres: float, city_soil_df: pd.DataFrame) -> dict:
    """
    Complete Agri Fusion pipeline.

    Frontend contract:
        city
        state
        season
        date
        farm_size_acres
        farm_ha
        crop
        climate
        irrigation
        yield
        market
        weather
        errors

    Pipeline:
        1. Crop Recommendation
        2. Climate Risk
        3. Irrigation
        4. Yield
        5. Market Price
    """

    # ─────────────────────────────────────────────────────────────────────
    # 0. Validate city
    # ─────────────────────────────────────────────────────────────────────

    if city not in CITY_META:
        raise ValueError(
            f"'{city}' isn't one of the supported cities."
        )

    meta = CITY_META[city]

    # Convert acres -> hectares
    farm_ha = farm_size_acres * ACRE_HA

    # Load models
    M = get_models()

    # Current date
    now = datetime.now()

    # ─────────────────────────────────────────────────────────────────────
    # Base result
    # ─────────────────────────────────────────────────────────────────────

    result = {
        "city": city,
        "state": meta["state"],
        "district": meta["district"],

        "farm_size_acres": float(farm_size_acres),
        "farm_ha": round(farm_ha, 4),

        "date": now.strftime("%d %B %Y"),
        "fetched_at": now.strftime("%d %B %Y, %I:%M %p"),

        "season": None,

        "crop": None,
        "climate": None,
        "irrigation": None,
        "yield": None,
        "market": None,
        "weather": None,

        "errors": {}
    }

    # ─────────────────────────────────────────────────────────────────────
    # 1. CROP RECOMMENDATION
    # ─────────────────────────────────────────────────────────────────────

    try:

        crop_result = crop_mod.recommend_crop(
            city=city,
            model=M["crop_model"],
            feature_columns=M["crop_columns"],
            label_encoder=M["crop_label_enc"],
            crop_encoders=M["crop_encoders"],
        )

        crop = crop_result["recommended_crop"]

        season = crop_result["season"]

        result["crop"] = {
            "name": crop,
            "confidence": crop_result.get(
                "confidence",
                crop_result.get("confidence_percent", 0)
            )
        }

        result["season"] = season

    except Exception as e:

        result["errors"]["crop_recommendation"] = str(e)

        crop = "Rice"
        season = crop_mod._season_for_today()

        result["season"] = season

        result["crop"] = {
            "name": crop,
            "confidence": 0
        }

    # ─────────────────────────────────────────────────────────────────────
    # 2. WEATHER
    # ─────────────────────────────────────────────────────────────────────

    try:

        weather_data = weather_mod.get_weather(
            meta["lat"],
            meta["lon"]
        )

        # Keep the complete weather dictionary available to frontend
        result["weather"] = {
            "temperature": weather_data.get(
                "temperature",
                weather_data.get("temperature_2m", 0)
            ),

            "relative_humidity": weather_data.get(
                "relative_humidity",
                weather_data.get("relative_humidity_2m", 0)
            ),

            "rainfall": weather_data.get(
                "rainfall",
                weather_data.get("rain", 0)
            ),

            "wind_speed": weather_data.get(
                "wind_speed",
                weather_data.get("wind_speed_10m", 0)
            ),

            "solar_radiation": weather_data.get(
                "solar_radiation",
                weather_data.get("shortwave_radiation", 0)
            ),

            "et0": weather_data.get(
                "et0",
                weather_data.get(
                    "et0_fao_evapotranspiration",
                    0
                )
            )
        }

    except Exception as e:

        result["errors"]["weather"] = str(e)

        result["weather"] = {
            "temperature": 0,
            "relative_humidity": 0,
            "rainfall": 0,
            "wind_speed": 0,
            "solar_radiation": 0,
            "et0": 0
        }

        weather_data = {
            "temperature": 0,
            "relative_humidity": 0,
            "rainfall": 0,
            "wind_speed": 0,
            "solar_radiation": 0,
            "et0": 0
        }

    # ─────────────────────────────────────────────────────────────────────
    # 3. CLIMATE RISK
    # ─────────────────────────────────────────────────────────────────────

    try:

        climate_result = climate_mod.predict_climate_risk(
            city=city,
            crop=crop,
            model=M["climate_model"],
            encoders=M["climate_enc"],
            city_soil_df=city_soil_df,
        )

        # Your frontend expects:
        #
        # R["climate"]["level"]
        # R["climate"]["color"]
        # R["climate"]["icon"]
        # R["climate"]["confidence"]
        # R["climate"]["rain_7d"]
        # R["climate"]["dry_days"]

        result["climate"] = {
            "level": climate_result.get(
                "level",
                climate_result.get(
                    "risk",
                    climate_result.get(
                        "climate_risk",
                        "unknown"
                    )
                )
            ),

            "color": climate_result.get(
                "color",
                "#43e97b"
            ),

            "icon": climate_result.get(
                "icon",
                "🌤️"
            ),

            "confidence": climate_result.get(
                "confidence",
                climate_result.get(
                    "confidence_percent",
                    0
                )
            ),

            "rain_7d": climate_result.get(
                "rain_7d",
                climate_result.get(
                    "rainfall_7d",
                    0
                )
            ),

            "dry_days": climate_result.get(
                "dry_days",
                climate_result.get(
                    "consecutive_dry_days",
                    0
                )
            )
        }

    except Exception as e:

        result["errors"]["climate_risk"] = str(e)

        result["climate"] = {
            "level": "unknown",
            "color": "#9e9e9e",
            "icon": "⚠️",
            "confidence": 0,
            "rain_7d": 0,
            "dry_days": 0
        }

    # ─────────────────────────────────────────────────────────────────────
    # 4. IRRIGATION
    # ─────────────────────────────────────────────────────────────────────

    try:

        irrigation_input = {
            "city": city,
            "crop": crop,
            "growth_stage": "Mid-season",
            "farm_size": farm_size_acres,
            "farm_size_unit": "Acres",
            "irrigation_method": "Drip",
            "pump_lpm": None,
        }

        irrigation_result = irrigation_mod.predict(
            irrigation_input,
            weather_data
        )

        # Frontend expects:
        #
        # irrigate
        # mm_day
        # liters
        # next_days
        # next_date

        mm_day = float(
            irrigation_result.get(
                "mm_day",
                irrigation_result.get(
                    "water_requirement_mm",
                    irrigation_result.get(
                        "daily_water_mm",
                        0
                    )
                )
            )
        )

        liters = float(
            irrigation_result.get(
                "liters",
                irrigation_result.get(
                    "total_liters",
                    mm_day * farm_ha * 10000
                )
            )
        )

        irrigate = bool(
            irrigation_result.get(
                "irrigate",
                irrigation_result.get(
                    "irrigation_needed",
                    False
                )
            )
        )

        next_days = int(
            irrigation_result.get(
                "next_days",
                1
            )
        )

        # Calculate next irrigation date
        from datetime import timedelta

        next_date = (
            now + timedelta(days=next_days)
        ).strftime("%d %B %Y")

        result["irrigation"] = {
            "irrigate": irrigate,
            "mm_day": round(mm_day, 2),
            "liters": round(liters, 0),
            "next_days": next_days,
            "next_date": next_date
        }

    except Exception as e:

        result["errors"]["irrigation"] = str(e)

        result["irrigation"] = {
            "irrigate": False,
            "mm_day": 0,
            "liters": 0,
            "next_days": 0,
            "next_date": "Not available"
        }

    # ─────────────────────────────────────────────────────────────────────
    # 5. YIELD
    # ─────────────────────────────────────────────────────────────────────

    try:

        yield_result = yield_mod.predict_yield(
            district=meta["district"],
            state=meta["state"],
            season=season,
            crop=crop,
            area=farm_ha,
            model=M["yield_model"],
            encoder=M["yield_enc"],
            state_map=M["yield_state_map"],
            city_soil_df=city_soil_df,
        )

        # Support the names returned by the existing backend.
        per_ha = float(
            yield_result.get(
                "per_ha",
                yield_result.get(
                    "yield_per_ha",
                    yield_result.get(
                        "predicted_yield_per_ha",
                        0
                    )
                )
            )
        )

        total = float(
            yield_result.get(
                "total",
                yield_result.get(
                    "total_tonnes",
                    per_ha * farm_ha
                )
            )
        )

        result["yield"] = {
            "per_ha": round(per_ha, 2),
            "total": round(total, 2),
            "total_tonnes": round(total, 2)
        }

    except Exception as e:

        result["errors"]["yield"] = str(e)

        result["yield"] = {
            "per_ha": 0,
            "total": 0,
            "total_tonnes": 0
        }

    # ─────────────────────────────────────────────────────────────────────
    # 6. MARKET PRICE
    # ─────────────────────────────────────────────────────────────────────

    try:

        total_tonnes = result["yield"]["total_tonnes"]

        arrival_quintals = round(
            total_tonnes * 10,
            1
        )

        market_result = _predict_market_price(
            city,
            crop,
            arrival_quintals,
            M
        )

        result["market"] = {
            "price_per_quintal": market_result.get(
                "price_per_quintal",
                0
            ),

            "total_value": market_result.get(
                "total_value_inr",
                0
            ),

            "arrival_qty": market_result.get(
                "arrival_qty_quintals",
                arrival_quintals
            ),

            # Keep original names too
            "total_value_inr": market_result.get(
                "total_value_inr",
                0
            ),

            "arrival_qty_quintals": market_result.get(
                "arrival_qty_quintals",
                arrival_quintals
            ),

            "market_crop": market_result.get(
                "market_crop",
                crop
            )
        }

    except Exception as e:

        result["errors"]["market"] = str(e)

        result["market"] = {
            "price_per_quintal": 0,
            "total_value": 0,
            "arrival_qty": 0,
            "total_value_inr": 0,
            "arrival_qty_quintals": 0,
            "market_crop": crop
        }

    # ─────────────────────────────────────────────────────────────────────
    # FINAL RESULT
    # ─────────────────────────────────────────────────────────────────────

    return result