"""
database.py — Supabase integration for all models.
Provides per-model save helpers that are safe to call even when
Supabase is not configured.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import requests

from backend.config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_ROLE_KEY


def _get_auth_headers() -> Dict[str, str]:
    key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _rest_request(method: str, table: str, payload: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, str]] = None):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}"
    headers = _get_auth_headers()
    try:
        response = requests.request(method, url, headers=headers, json=payload, params=params, timeout=20)
    except Exception as exc:
        print(f"[Supabase] {method} {table} request failed: {exc}")
        return None

    if response.status_code in (200, 201, 204):
        try:
            return response.json()
        except ValueError:
            return None

    print(f"[Supabase] {method} {table} failed: {response.status_code} {response.text[:500]}")
    return None


# ── Generic safe insert ───────────────────────────────────────────────────────
def _insert(table: str, data: dict) -> bool:
    """Insert a row. Silently skips if Supabase is not connected."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    try:
        result = _rest_request("POST", table, payload=data)
        return result is not None
    except Exception as exc:
        print(f"[Supabase] {table} insert failed: {exc}")
        return False


# ── Farmer Auth ───────────────────────────────────────────────────────────────
def register_farmer(full_name: str, phone: str, password_hash: str, password_salt: Optional[str] = None) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    try:
        payload = {
            "full_name": full_name,
            "phone": phone,
            "password_hash": password_hash,
            "created_at": datetime.now().isoformat(),
        }
        return _insert("farmers", payload)
    except Exception as e:
        print(f"[Supabase] register_farmer failed: {e}")
        return False


def get_farmer(phone: str) -> dict | None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        result = _rest_request("GET", "farmers", params={"phone": f"eq.{phone}"})
        if isinstance(result, list) and result:
            return result[0]
        return None
    except Exception as e:
        print(f"[Supabase] get_farmer failed: {e}")
        return None


def update_last_login(phone: str):
    if not SUPABASE_URL or not SUPABASE_KEY:
        return
    try:
        _rest_request("PATCH", "farmers", payload={"last_login_at": datetime.now().isoformat()}, params={"phone": f"eq.{phone}"})
    except Exception as e:
        print(f"[Supabase] update_last_login failed: {e}")


# ── Climate Risk ──────────────────────────────────────────────────────────────
def save_climate_risk(farmer_phone: str, result: dict, soil_row: Optional[Dict[str, Any]] = None) -> bool:
    payload = {
        "farmer_phone": farmer_phone or "guest",
        "city": result.get("city") or "",
        "crop": result.get("crop") or "",
        "state": result.get("state") or "",
        "season": result.get("season") or "",
        "risk_level": result.get("risk_level") or "",
        "risk_score": result.get("risk_score"),
        "confidence_pct": result.get("confidence_pct"),
        "climate_summary": result.get("climate_summary") or "",
    }

    weather = result.get("weather") if isinstance(result.get("weather"), dict) else {}
    if weather:
        payload.update({
            "temperature_2m": weather.get("temperature_2m") or weather.get("temperature"),
            "relative_humidity_2m": weather.get("relative_humidity_2m") or weather.get("humidity"),
            "precipitation": weather.get("precipitation") or weather.get("rainfall"),
            "wind_speed_10m": weather.get("wind_speed_10m") or weather.get("wind_speed"),
            "surface_pressure": weather.get("surface_pressure"),
            "cloud_cover": weather.get("cloud_cover"),
            "shortwave_radiation": weather.get("shortwave_radiation"),
            "et0": weather.get("et0"),
            "soil_moisture": weather.get("soil_moisture"),
            "soil_temperature": weather.get("soil_temperature"),
            "heat_index": weather.get("heat_index"),
            "rainfall_last_7_days": weather.get("rainfall_last_7_days"),
            "rainfall_last_30_days": weather.get("rainfall_last_30_days"),
            "consecutive_dry_days": weather.get("consecutive_dry_days"),
        })

    if isinstance(soil_row, dict):
        payload.update({
            "soil_ph": soil_row.get("Soil_pH") or soil_row.get("soil_ph"),
            "organic_carbon": soil_row.get("Organic_Carbon") or soil_row.get("organic_carbon"),
            "clay_percentage": soil_row.get("Clay_Percentage") or soil_row.get("clay_percentage"),
            "sand_percentage": soil_row.get("Sand_Percentage") or soil_row.get("sand_percentage"),
            "silt_percentage": soil_row.get("Silt_Percentage") or soil_row.get("silt_percentage"),
            "elevation": soil_row.get("Elevation") or soil_row.get("elevation"),
        })

    return _insert("climate_risk_predictions", payload)


# ── Crop Recommendation ───────────────────────────────────────────────────────
def save_crop_recommendation(farmer_phone: str, city: str, result: dict, weather: Optional[Dict[str, Any]] = None) -> bool:
    payload = {
        "farmer_phone": farmer_phone or "guest",
        "city": city or "",
        "season": result.get("season", ""),
        "recommended_crop": result.get("recommended_crop") or "",
        "confidence_percent": result.get("confidence_percent"),
        "recommendation": result.get("recommendation") or "",
        "care_tips": result.get("care_tips") or "",
    }

    if isinstance(weather, dict):
        payload.update({
            "temperature": weather.get("temperature") or weather.get("temperature_c"),
            "humidity": weather.get("humidity") or weather.get("humidity_percent"),
            "rainfall": weather.get("rainfall") or weather.get("rainfall_mm"),
            "wind_speed": weather.get("wind_speed"),
        })

    soil = result.get("soil") if isinstance(result.get("soil"), dict) else {}
    if soil:
        payload.update({
            "soil_ph": soil.get("soil_ph") or soil.get("ph"),
            "nitrogen": soil.get("nitrogen") or soil.get("nitrogen_N"),
            "organic_carbon": soil.get("organic_carbon"),
            "clay": soil.get("clay"),
            "sand": soil.get("sand"),
            "silt": soil.get("silt"),
            "cec": soil.get("cec"),
        })

    return _insert("crop_recommendation_predictions", payload)


# ── Irrigation ────────────────────────────────────────────────────────────────
def save_irrigation(farmer_phone: str, user_input: dict, result: dict, weather: Optional[Dict[str, Any]] = None, soil_row: Optional[Dict[str, Any]] = None) -> bool:
    payload = {
        "farmer_phone": farmer_phone or "guest",
        "city": user_input.get("city") or "",
        "crop": user_input.get("crop") or "",
        "growth_stage": user_input.get("growth_stage") or "",
        "farm_size": user_input.get("farm_size"),
        "farm_size_unit": user_input.get("farm_size_unit") or "",
        "irrigation_source": user_input.get("irrigation_source") or "",
        "irrigation_method": user_input.get("irrigation_method"),
        "water_requirement_mm_day": result.get("water_requirement_mm_day"),
        "total_liters": result.get("total_liters"),
        "irrigation_required": result.get("irrigation_required"),
        "best_irrigation_time": result.get("best_irrigation_time") or "",
        "next_irrigation_days": result.get("next_irrigation_days"),
        "motor_minutes": result.get("motor_minutes"),
    }

    if isinstance(weather, dict):
        payload.update({
            "weather_temperature": weather.get("temperature") or weather.get("temperature_c"),
            "weather_relative_humidity": weather.get("relative_humidity") or weather.get("humidity"),
            "weather_rainfall": weather.get("rainfall"),
            "weather_wind_speed": weather.get("wind_speed"),
            "weather_solar_radiation": weather.get("solar_radiation"),
            "weather_et0": weather.get("et0"),
        })

    if isinstance(soil_row, dict):
        payload.update({
            "soil_ph": soil_row.get("Soil_pH") or soil_row.get("soil_ph"),
            "organic_carbon": soil_row.get("Organic_Carbon") or soil_row.get("organic_carbon"),
            "sand_percentage": soil_row.get("Sand_Percentage") or soil_row.get("sand_percentage"),
            "silt_percentage": soil_row.get("Silt_Percentage") or soil_row.get("silt_percentage"),
            "clay_percentage": soil_row.get("Clay_Percentage") or soil_row.get("clay_percentage"),
            "cec": soil_row.get("CEC") or soil_row.get("cec"),
            "bulk_density": soil_row.get("Bulk_Density") or soil_row.get("bulk_density"),
            "field_capacity": soil_row.get("Field_Capacity") or soil_row.get("field_capacity"),
            "wilting_point": soil_row.get("Wilting_Point") or soil_row.get("wilting_point"),
            "available_water": soil_row.get("Available_Water") or soil_row.get("available_water"),
            "nitrogen": soil_row.get("Nitrogen") or soil_row.get("nitrogen"),
            "soil_type": soil_row.get("Soil_Type") or soil_row.get("soil_type"),
        })

    return _insert("irrigation_predictions", payload)


# ── Yield ─────────────────────────────────────────────────────────────────────
def save_yield(farmer_phone: str, district: str, state: str, season: str, crop: str, area_ha: float, result: dict, weather: Optional[Dict[str, Any]] = None, soil: Optional[Dict[str, Any]] = None) -> bool:
    payload = {
        "farmer_phone": farmer_phone or "guest",
        "state": state or "",
        "district": district or "",
        "season": season or "",
        "crop": crop or "",
        "area": area_ha,
        "yield_per_hectare": result.get("yield_per_hectare"),
        "total_tonnes": result.get("total_tonnes"),
    }

    if isinstance(weather, dict):
        payload.update({
            "mean_temperature": weather.get("mean_temperature"),
            "max_temperature": weather.get("max_temperature"),
            "min_temperature": weather.get("min_temperature"),
            "precipitation": weather.get("precipitation"),
            "shortwave_radiation": weather.get("shortwave_radiation"),
            "wind_speed": weather.get("wind_speed"),
            "relative_humidity": weather.get("relative_humidity"),
            "et0": weather.get("et0"),
            "soil_moisture": weather.get("soil_moisture"),
            "soil_temperature": weather.get("soil_temperature"),
        })

    if isinstance(soil, dict):
        payload.update({
            "soil_ph": soil.get("soil_ph"),
            "organic_carbon": soil.get("organic_carbon"),
            "clay": soil.get("clay"),
            "sand": soil.get("sand"),
            "silt": soil.get("silt"),
        })

    return _insert("yield_predictions", payload)


# ── Market Price ──────────────────────────────────────────────────────────────
def save_market_price(farmer_phone: str, commodity: str, state: str, district: str, market_date: Any, arrival_quantity: float, predicted_price: float, quarter: int) -> bool:
    payload = {
        "farmer_phone": farmer_phone or "guest",
        "commodity": commodity or "",
        "state": state or "",
        "district": district or "",
        "market_date": market_date.isoformat() if hasattr(market_date, "isoformat") else market_date,
        "arrival_quantity": arrival_quantity,
        "predicted_price": predicted_price,
        "quarter": quarter,
    }

    if hasattr(market_date, "day"):
        payload.update({
            "day": market_date.day,
            "month": market_date.month,
            "year": market_date.year,
        })

    return _insert("market_price_predictions", payload)


# ── Fusion ────────────────────────────────────────────────────────────────────
def save_fusion(farmer_phone: str, inputs: dict, results: dict):
    return _insert("fusion_predictions", {
        "farmer_phone": farmer_phone,
        "city": inputs.get("city"),
        "crop": inputs.get("crop"),
        "farm_size": inputs.get("farm_size"),
        "farm_size_unit": inputs.get("farm_size_unit"),
        "state": inputs.get("state"),
        "district": inputs.get("district"),
        "season": inputs.get("season"),
        "climate_risk_level": results.get("climate_risk_level"),
        "recommended_crop": results.get("recommended_crop"),
        "irrigation_required": results.get("irrigation_required"),
        "predicted_yield_tonnes": results.get("predicted_yield_tonnes"),
        "predicted_market_price": results.get("predicted_market_price"),
    })


# ── Existing generic fallback (keeps old code working) ───────────────────────
def save_prediction(data: dict, table: str = "irrigation_predictions") -> bool:
    """Legacy generic function."""
    return _insert(table, data)


# ── Feedback ──────────────────────────────────────────────────────────────────
def save_feedback(
    farmer_phone: Optional[str] = None,
    rating: Optional[int] = None,
    difficulties: Optional[str] = None,
    incorrect_model: Optional[str] = None,
    incorrect_outputs: Optional[str] = None,
    contact_phone: Optional[str] = None,
    general_comments: Optional[str] = None,
) -> bool:
    payload = {
        "farmer_phone": farmer_phone or "guest",
        "rating": rating,
        "difficulties": difficulties or "",
        "incorrect_model": incorrect_model or "",
        "incorrect_outputs": incorrect_outputs or "",
        "contact_phone": contact_phone or "",
        "general_comments": general_comments or "",
        "submitted_at": datetime.now().isoformat(),
    }
    return _insert("feedback", payload)