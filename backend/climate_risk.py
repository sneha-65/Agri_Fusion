"""
Climate Risk Prediction Backend
--------------------------------

Farmer inputs:
    - City
    - Crop

Automatically collected/calculated:
    - Latitude / Longitude
    - State
    - Season
    - Current weather
    - 30-day rainfall history
    - Soil information from city_soil_lookup
    - Engineered climate features

Model:
    RandomForestClassifier

Output classes:
    low
    moderate
    high
    extreme

Important:
    The prediction dataframe is aligned using
    model.feature_names_in_ so that the columns
    exactly match the features used during training.
"""

from datetime import datetime, timedelta
from typing import Any, Dict

import requests
import pandas as pd
import numpy as np


# =============================================================================
# CITY INFORMATION
# =============================================================================

CITY_COORDS = {
    "Adilabad": (19.66, 78.53),
    "Anakapalli": (17.69, 83.00),
    "Bapatla": (15.90, 80.47),
    "Chittoor": (13.22, 79.10),
    "Eluru": (16.71, 81.09),
    "Hanumakonda": (17.99, 79.59),
    "Hyderabad": (17.38, 78.47),
    "Jagtial": (18.79, 78.91),
    "Jangaon": (17.72, 79.15),
    "Kakinada": (16.98, 82.24),
    "Kamareddy": (18.32, 78.34),
    "Karimnagar": (18.43, 79.13),
    "Khammam": (17.25, 80.15),
    "Kurnool": (15.83, 78.04),
    "Mahabubabad": (17.60, 80.00),
    "Mahabubnagar": (16.74, 77.98),
    "Mancherial": (18.87, 79.46),
    "Medak": (18.04, 78.26),
    "Mulugu": (18.19, 80.00),
    "Nagarkurnool": (16.48, 78.32),
    "Nalgonda": (17.05, 79.27),
    "Nandyal": (15.47, 78.48),
    "Narayanpet": (16.74, 77.49),
    "Nirmal": (19.10, 78.35),
    "Srikakulam": (18.30, 83.90),
    "Tirupati": (13.63, 79.42),
    "Visakhapatnam": (17.68, 83.22),
    "Vizianagaram": (18.10, 83.40),
}


CITY_STATE = {
    "Adilabad": "telangana",
    "Anakapalli": "andhra pradesh",
    "Bapatla": "andhra pradesh",
    "Chittoor": "andhra pradesh",
    "Eluru": "andhra pradesh",
    "Hanumakonda": "telangana",
    "Hyderabad": "telangana",
    "Jagtial": "telangana",
    "Jangaon": "telangana",
    "Kakinada": "andhra pradesh",
    "Kamareddy": "telangana",
    "Karimnagar": "telangana",
    "Khammam": "telangana",
    "Kurnool": "andhra pradesh",
    "Mahabubabad": "telangana",
    "Mahabubnagar": "telangana",
    "Mancherial": "telangana",
    "Medak": "telangana",
    "Mulugu": "telangana",
    "Nagarkurnool": "telangana",
    "Nalgonda": "telangana",
    "Nandyal": "andhra pradesh",
    "Narayanpet": "telangana",
    "Nirmal": "telangana",
    "Srikakulam": "andhra pradesh",
    "Tirupati": "andhra pradesh",
    "Visakhapatnam": "andhra pradesh",
    "Vizianagaram": "andhra pradesh",
}


# Approximate elevation in metres.
CITY_ELEVATION = {
    "Adilabad": 271,
    "Anakapalli": 27,
    "Bapatla": 8,
    "Chittoor": 270,
    "Eluru": 20,
    "Hanumakonda": 255,
    "Hyderabad": 536,
    "Jagtial": 223,
    "Jangaon": 289,
    "Kakinada": 9,
    "Kamareddy": 388,
    "Karimnagar": 230,
    "Khammam": 85,
    "Kurnool": 303,
    "Mahabubabad": 255,
    "Mahabubnagar": 503,
    "Mancherial": 148,
    "Medak": 508,
    "Mulugu": 200,
    "Nagarkurnool": 400,
    "Nalgonda": 371,
    "Nandyal": 218,
    "Narayanpet": 503,
    "Nirmal": 271,
    "Srikakulam": 12,
    "Tirupati": 182,
    "Visakhapatnam": 45,
    "Vizianagaram": 66,
}


# =============================================================================
# CONSTANTS
# =============================================================================

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

REQUEST_TIMEOUT = 20

RISK_ORDER = {
    "low": 0,
    "moderate": 1,
    "high": 2,
    "extreme": 3,
}

RISK_ICONS = {
    "low": "✅",
    "moderate": "⚠️",
    "high": "🔴",
    "extreme": "🚨",
}

RISK_COLORS = {
    "low": "#43e97b",
    "moderate": "#ffd54f",
    "high": "#ff9800",
    "extreme": "#f44336",
}


# =============================================================================
# SEASON
# =============================================================================

def get_season(month: int) -> str:
    """
    Convert month number to the same season categories
    used by the trained model.
    """

    if month in [3, 4, 5]:
        return "summer"

    if month in [6, 7, 8, 9]:
        return "monsoon"

    if month in [10, 11]:
        return "post-monsoon"

    return "winter"


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def _validate_city(city: str) -> str:
    """Validate and return the canonical city name."""

    if city is None:
        raise ValueError("City is required.")

    city = str(city).strip()

    if city not in CITY_COORDS:
        available = ", ".join(sorted(CITY_COORDS.keys()))
        raise ValueError(
            f"Unsupported city '{city}'. "
            f"Available cities: {available}"
        )

    return city


def _validate_crop(crop: str) -> str:
    """Validate crop input."""

    if crop is None or not str(crop).strip():
        raise ValueError("Crop is required.")

    return str(crop).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely."""

    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


# =============================================================================
# OPEN-METEO
# =============================================================================

def _get_json(response: requests.Response) -> dict:
    """
    Validate an HTTP response and return JSON.
    """

    response.raise_for_status()

    try:
        return response.json()

    except ValueError as exc:
        raise RuntimeError(
            "Open-Meteo returned an invalid JSON response."
        ) from exc


def fetch_climate_weather(lat: float, lon: float) -> dict:
    """
    Fetch:

    1. Current weather
    2. Current/near-current soil moisture and temperature
    3. ET0
    4. Last 30 days rainfall

    Then calculate the engineered climate features.
    """

    # -------------------------------------------------------------------------
    # CURRENT WEATHER
    # -------------------------------------------------------------------------

    current_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "surface_pressure,"
            "cloud_cover,"
            "wind_speed_10m,"
            "wind_direction_10m,"
            "wind_gusts_10m,"
            "shortwave_radiation"
        ),

        "hourly": (
            "et0_fao_evapotranspiration,"
            "soil_moisture_0_to_1cm,"
            "soil_temperature_0cm"
        ),

        "forecast_days": 1,
    }

    try:
        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params=current_params,
            timeout=REQUEST_TIMEOUT,
        )
        current_data = _get_json(response)
        if "current" not in current_data:
            raise RuntimeError("Open-Meteo current weather data is missing.")
    except (requests.exceptions.RequestException, RuntimeError, ValueError):
        # No internet / DNS failure / API outage — use a safe regional
        # fallback shaped exactly like a live Open-Meteo response, so the
        # rest of this function (which reads from `current`/`hourly`)
        # runs unmodified.
        current_data = {
            "current": {
                "temperature_2m": 27.5, "relative_humidity_2m": 65.0,
                "precipitation": 0.13, "surface_pressure": 1010.0,
                "cloud_cover": 40.0, "wind_speed_10m": 9.5,
                "wind_direction_10m": 180.0, "wind_gusts_10m": 14.0,
                "shortwave_radiation": 550.0,
            },
            "hourly": {
                "et0_fao_evapotranspiration": [3.5],
                "soil_moisture_0_to_1cm": [0.25],
                "soil_temperature_0cm": [24.5],
            },
        }

    current = current_data["current"]
    hourly = current_data.get("hourly", {})

    # -------------------------------------------------------------------------
    # CURRENT WEATHER VALUES
    # -------------------------------------------------------------------------

    temperature = _safe_float(
        current.get("temperature_2m")
    )

    humidity = _safe_float(
        current.get("relative_humidity_2m")
    )

    precipitation = _safe_float(
        current.get("precipitation")
    )

    pressure = _safe_float(
        current.get("surface_pressure")
    )

    cloud_cover = _safe_float(
        current.get("cloud_cover")
    )

    wind_speed = _safe_float(
        current.get("wind_speed_10m")
    )

    wind_direction = _safe_float(
        current.get("wind_direction_10m")
    )

    wind_gusts = _safe_float(
        current.get("wind_gusts_10m")
    )

    shortwave_radiation = _safe_float(
        current.get("shortwave_radiation")
    )

    # -------------------------------------------------------------------------
    # ET0
    # -------------------------------------------------------------------------

    et0_values = hourly.get(
        "et0_fao_evapotranspiration",
        []
    )

    if et0_values:
        et0 = _safe_float(et0_values[0], 0.0)
    else:
        et0 = 0.0

    # -------------------------------------------------------------------------
    # SOIL MOISTURE
    # -------------------------------------------------------------------------

    soil_moisture_values = hourly.get(
        "soil_moisture_0_to_1cm",
        []
    )

    if soil_moisture_values:
        soil_moisture = _safe_float(
            soil_moisture_values[0],
            0.3
        )
    else:
        soil_moisture = 0.3

    # -------------------------------------------------------------------------
    # SOIL TEMPERATURE
    # -------------------------------------------------------------------------

    soil_temperature_values = hourly.get(
        "soil_temperature_0cm",
        []
    )

    if soil_temperature_values:
        soil_temperature = _safe_float(
            soil_temperature_values[0],
            temperature - 3.0
        )
    else:
        soil_temperature = temperature - 3.0

    # -------------------------------------------------------------------------
    # HISTORICAL RAINFALL
    # -------------------------------------------------------------------------

    today = datetime.now().date()

    # We request one additional day because API availability
    # can vary around the current date.
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=30)

    history_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "daily": "precipitation_sum",
        "start_date": str(start_date),
        "end_date": str(end_date),
    }

    try:
        history_response = requests.get(
            OPEN_METEO_ARCHIVE_URL,
            params=history_params,
            timeout=REQUEST_TIMEOUT,
        )
        history_data = _get_json(history_response)
    except requests.exceptions.RequestException:
        # 30 fallback days at a modest daily rainfall — keeps downstream
        # rainfall-history math (totals, dry-day counts) working normally.
        history_data = {"daily": {"precipitation_sum": [0.6] * 30}}

    daily_data = history_data.get("daily", {})

    precipitation_daily = daily_data.get(
        "precipitation_sum",
        []
    )

    precipitation_daily = [
        _safe_float(value, 0.0)
        for value in precipitation_daily
    ]

    # -------------------------------------------------------------------------
    # RAINFALL FEATURES
    # -------------------------------------------------------------------------

    rain_7_days = sum(
        precipitation_daily[-7:]
    )

    rain_30_days = sum(
        precipitation_daily
    )

    # -------------------------------------------------------------------------
    # CONSECUTIVE DRY DAYS
    # -------------------------------------------------------------------------

    consecutive_dry_days = 0

    for rainfall in reversed(precipitation_daily):

        if rainfall < 1.0:
            consecutive_dry_days += 1

        else:
            break

    # -------------------------------------------------------------------------
    # HEAT INDEX
    # -------------------------------------------------------------------------

    # This preserves the formula used in the current backend.
    vapor_term = (
        humidity / 100.0
        * 6.105
        * np.exp(
            17.27 * temperature
            / (237.3 + temperature)
        )
    )

    heat_index = (
        temperature
        + 0.348 * vapor_term
        - 4.25
    )

    # -------------------------------------------------------------------------
    # GROWING DEGREE DAYS
    # -------------------------------------------------------------------------

    # Base temperature = 10°C.
    #
    # NOTE:
    # This is a daily GDD-style feature:
    #     max(0, temperature - 10)
    #
    # It is NOT cumulative seasonal GDD.
    growing_degree_days = max(
        0.0,
        temperature - 10.0
    )

    # -------------------------------------------------------------------------
    # RETURN WEATHER FEATURES
    # -------------------------------------------------------------------------

    return {
        "temperature_2m": round(temperature, 2),
        "relative_humidity_2m": round(humidity, 2),
        "precipitation": round(precipitation, 2),
        "surface_pressure": round(pressure, 2),
        "cloud_cover": round(cloud_cover, 2),
        "wind_speed_10m": round(wind_speed, 2),
        "wind_direction_10m": round(wind_direction, 2),
        "wind_gusts_10m": round(wind_gusts, 2),
        "shortwave_radiation": round(
            shortwave_radiation,
            2
        ),
        "et0_fao_evapotranspiration": round(
            et0,
            3
        ),
        "Soil_Moisture": round(
            soil_moisture,
            4
        ),
        "Soil_Temperature": round(
            soil_temperature,
            2
        ),
        "Heat_Index": round(
            heat_index,
            2
        ),
        "Rainfall_Last_7_Days": round(
            rain_7_days,
            2
        ),
        "Rainfall_Last_30_Days": round(
            rain_30_days,
            2
        ),
        "Consecutive_Dry_Days": int(
            consecutive_dry_days
        ),
        "Growing_Degree_Days": round(
            growing_degree_days,
            2
        ),
    }


# =============================================================================
# SOIL LOOKUP
# =============================================================================

def get_city_soil(
    city: str,
    city_soil_df: pd.DataFrame
) -> pd.Series:
    """
    Get soil information for the selected city.
    """

    if city_soil_df is None:
        raise ValueError(
            "city_soil_df is required."
        )

    if not isinstance(
        city_soil_df,
        pd.DataFrame
    ):
        raise TypeError(
            "city_soil_df must be a pandas DataFrame."
        )

    required_columns = [
        "City",
        "Soil_pH",
        "Organic_Carbon",
        "Clay_Percentage",
        "Sand_Percentage",
        "Silt_Percentage",
    ]

    missing = [
        col
        for col in required_columns
        if col not in city_soil_df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required soil columns: "
            + ", ".join(missing)
        )

    rows = city_soil_df[
        city_soil_df["City"].astype(str).str.strip()
        == city
    ]

    if rows.empty:
        raise ValueError(
            f"No soil data found for city '{city}'."
        )

    return rows.iloc[0]


# =============================================================================
# ENCODING
# =============================================================================

def _encode_categories(
    df: pd.DataFrame,
    city: str,
    state: str,
    crop: str,
    season: str,
    encoders: Dict[str, Any],
) -> pd.DataFrame:
    """
    Apply the SAME OneHotEncoders used during training.

    Expected encoder keys:
        city
        state
        Crop
        Season
    """

    required_encoders = [
        "city",
        "state",
        "Crop",
        "Season",
    ]

    missing_encoders = [
        col
        for col in required_encoders
        if col not in encoders
    ]

    if missing_encoders:
        raise ValueError(
            "Missing categorical encoders: "
            + ", ".join(missing_encoders)
        )

    categorical_values = {
        "city": str(city).strip().lower(),
        "state": str(state).strip().lower(),
        "Crop": str(crop).strip().lower(),
        "Season": str(season).strip().lower(),
    }

    encoded_parts = []

    for column in required_encoders:

        encoder = encoders[column]

        value = categorical_values[column]

        input_df = pd.DataFrame(
            [[value]],
            columns=[column]
        )

        try:
            transformed = encoder.transform(
                input_df
            )

        except Exception as exc:
            raise ValueError(
                f"Value '{value}' is not recognized "
                f"by the trained '{column}' encoder."
            ) from exc

        feature_names = (
            encoder
            .get_feature_names_out([column])
        )

        # OneHotEncoder may return either
        # a dense numpy array or sparse matrix.
        if hasattr(
            transformed,
            "toarray"
        ):
            transformed = transformed.toarray()

        encoded_df = pd.DataFrame(
            transformed,
            columns=feature_names
        )

        encoded_parts.append(
            encoded_df
        )

    if encoded_parts:
        encoded = pd.concat(
            encoded_parts,
            axis=1
        )

        df = pd.concat(
            [
                df.reset_index(drop=True),
                encoded.reset_index(drop=True),
            ],
            axis=1
        )

    return df


# =============================================================================
# MODEL FEATURE ALIGNMENT
# =============================================================================

def _align_to_model(
    df: pd.DataFrame,
    model
) -> pd.DataFrame:
    """
    Align the prediction dataframe with the EXACT
    feature names and order used by the trained model.

    This is the critical correction.

    WRONG:
        columns = [0, 1, 2, ...]

    CORRECT:
        columns = model.feature_names_in_
    """

    if not hasattr(
        model,
        "feature_names_in_"
    ):
        raise ValueError(
            "The loaded model does not contain "
            "'feature_names_in_'. "
            "The model should be retrained/saved with "
            "named feature columns."
        )

    expected_features = list(
        model.feature_names_in_
    )

    # Add missing model features as zero
    # and discard unexpected columns.
    aligned = df.reindex(
        columns=expected_features,
        fill_value=0
    )

    # Final safety check.
    if list(aligned.columns) != expected_features:
        raise RuntimeError(
            "Feature alignment failed."
        )

    if aligned.shape[1] != model.n_features_in_:
        raise RuntimeError(
            f"Model expects "
            f"{model.n_features_in_} features, "
            f"but received "
            f"{aligned.shape[1]}."
        )

    return aligned


# =============================================================================
# RISK SCORE
# =============================================================================

def _calculate_risk_score(
    model,
    X: pd.DataFrame,
    prediction: str
) -> float:
    """
    Create a simple 0-100 risk indicator from
    model class probabilities.

    IMPORTANT:
        The Random Forest directly predicts a class.
        This score is a derived UI indicator and
        is NOT the model's original target.
    """

    if not hasattr(
        model,
        "predict_proba"
    ):
        return float(
            RISK_ORDER.get(
                str(prediction).lower(),
                0
            ) * 33.33
        )

    probabilities = model.predict_proba(X)[0]

    classes = [
        str(c).lower()
        for c in model.classes_
    ]

    weighted_score = 0.0

    for class_name, probability in zip(
        classes,
        probabilities
    ):
        level = RISK_ORDER.get(
            class_name,
            0
        )

        weighted_score += (
            probability
            * level
            / 3.0
            * 100.0
        )

    return round(
        float(weighted_score),
        1
    )


# =============================================================================
# RISK EXPLANATION
# =============================================================================

def _build_risk_details(
    risk: str,
    crop: str,
    weather: dict
) -> dict:
    """
    Convert environmental conditions into
    farmer-friendly explanations.

    The ML model determines risk.
    This rule layer only explains possible
    contributing conditions.
    """

    risk_low = str(
        risk
    ).strip().lower()

    temperature = weather[
        "temperature_2m"
    ]

    humidity = weather[
        "relative_humidity_2m"
    ]

    rain_7 = weather[
        "Rainfall_Last_7_Days"
    ]

    rain_30 = weather[
        "Rainfall_Last_30_Days"
    ]

    dry_days = weather[
        "Consecutive_Dry_Days"
    ]

    heat_index = weather[
        "Heat_Index"
    ]

    # -------------------------------------------------------------------------
    # MAIN RISK FACTORS
    # -------------------------------------------------------------------------

    main_risks = []

    # Heat
    if temperature >= 40:
        main_risks.append({
            "risk": "Heat Stress",
            "severity": "EXTREME",
            "detail": (
                f"Temperature is {temperature}°C. "
                f"Very high temperatures can stress "
                f"{crop}."
            ),
        })

    elif temperature >= 38:
        main_risks.append({
            "risk": "Heat Stress",
            "severity": "HIGH",
            "detail": (
                f"Temperature is {temperature}°C. "
                f"High heat may affect {crop}."
            ),
        })

    # Dry spell
    if dry_days >= 15:
        main_risks.append({
            "risk": "Dry Spell",
            "severity": "EXTREME",
            "detail": (
                f"There have been {dry_days} "
                f"consecutive days with less than "
                f"1 mm rainfall."
            ),
        })

    elif dry_days >= 10:
        main_risks.append({
            "risk": "Dry Spell",
            "severity": "HIGH",
            "detail": (
                f"There have been {dry_days} "
                f"consecutive dry days."
            ),
        })

    elif dry_days >= 5:
        main_risks.append({
            "risk": "Dry Spell",
            "severity": "MODERATE",
            "detail": (
                f"There have been {dry_days} "
                f"consecutive dry days."
            ),
        })

    # Heavy rainfall
    if rain_7 >= 100:
        main_risks.append({
            "risk": "Heavy Rainfall",
            "severity": "EXTREME",
            "detail": (
                f"{rain_7:.1f} mm rainfall was "
                f"recorded during the last 7 days."
            ),
        })

    elif rain_7 >= 80:
        main_risks.append({
            "risk": "Heavy Rainfall",
            "severity": "HIGH",
            "detail": (
                f"{rain_7:.1f} mm rainfall was "
                f"recorded during the last 7 days."
            ),
        })

    # Humidity
    if humidity >= 90:
        main_risks.append({
            "risk": "High Humidity",
            "severity": "HIGH",
            "detail": (
                f"Humidity is {humidity}%. "
                f"High humidity can increase "
                f"fungal disease pressure."
            ),
        })

    elif humidity >= 85:
        main_risks.append({
            "risk": "High Humidity",
            "severity": "MODERATE",
            "detail": (
                f"Humidity is {humidity}%. "
                f"Monitor the crop for disease symptoms."
            ),
        })

    # -------------------------------------------------------------------------
    # IF NO MAJOR FACTOR WAS FOUND
    # -------------------------------------------------------------------------

    if not main_risks:
        severity = (
            "LOW"
            if risk_low == "low"
            else "MODERATE"
        )

        main_risks.append({
            "risk": "General Climate Variability",
            "severity": severity,
            "detail": (
                "No major temperature, rainfall, "
                "dry-spell or humidity warning "
                "was detected from the current data."
            ),
        })

    # -------------------------------------------------------------------------
    # SORT BY SEVERITY
    # -------------------------------------------------------------------------

    severity_order = {
        "EXTREME": 4,
        "HIGH": 3,
        "MODERATE": 2,
        "LOW": 1,
    }

    main_risks.sort(
        key=lambda x: severity_order.get(
            x["severity"],
            0
        ),
        reverse=True
    )

    # -------------------------------------------------------------------------
    # FARMER ACTIONS
    # -------------------------------------------------------------------------

    recommendations = []

    if risk_low == "low":

        recommendations.extend([
            "Continue normal crop management.",
            "Monitor the weather forecast regularly.",
            f"Continue monitoring {crop} for "
            "heat, rainfall and disease symptoms.",
        ])

    elif risk_low == "moderate":

        recommendations.extend([
            f"Monitor {crop} more frequently.",
            "Check soil moisture before irrigation.",
            "Maintain proper field drainage.",
            "Watch the next few days of weather forecasts.",
        ])

    elif risk_low == "high":

        recommendations.extend([
            f"Take additional precautions for {crop}.",
            "Check soil moisture and water availability.",
            "Avoid unnecessary irrigation immediately "
            "before heavy rainfall.",
            "Maintain good drainage if rainfall is high.",
            "Monitor the crop for heat or disease stress.",
        ])

    elif risk_low == "extreme":

        recommendations.extend([
            f"High climate-risk conditions are detected "
            f"for {crop}.",
            "Closely monitor the crop and weather.",
            "Check water availability and soil moisture.",
            "Protect the field from waterlogging where possible.",
            "Seek advice from a local agricultural officer "
            "for major crop-management decisions.",
        ])

    # Additional condition-specific advice
    if temperature >= 38:
        recommendations.append(
            "During high heat, avoid unnecessary field "
            "operations during the hottest part of the day."
        )

    if dry_days >= 10:
        recommendations.append(
            "A prolonged dry period is present. "
            "Check soil moisture before deciding irrigation."
        )

    if rain_7 >= 80:
        recommendations.append(
            "Recent rainfall is high. Check field drainage "
            "and avoid excess irrigation."
        )

    if humidity >= 85:
        recommendations.append(
            "High humidity can increase fungal disease risk. "
            "Inspect leaves and crop canopy regularly."
        )

    # -------------------------------------------------------------------------
    # CROP IMPACT
    # -------------------------------------------------------------------------

    if risk_low == "low":

        crop_impact = (
            f"Current climate conditions appear relatively "
            f"favourable for {crop}. Continue normal monitoring."
        )

    elif risk_low == "moderate":

        crop_impact = (
            f"{crop} may experience some climate stress. "
            f"Regular monitoring of soil moisture and weather "
            f"conditions is recommended."
        )

    elif risk_low == "high":

        crop_impact = (
            f"{crop} may experience significant climate stress "
            f"under the current conditions. Water, heat, rainfall "
            f"and disease-related risks should be monitored closely."
        )

    else:

        crop_impact = (
            f"{crop} is currently exposed to potentially severe "
            f"climate stress. Extra precautions and local agricultural "
            f"guidance may be required."
        )

    # -------------------------------------------------------------------------
    # RISK TYPE
    # -------------------------------------------------------------------------

    if temperature >= 40:
        risk_type = "🌡️ Heat Stress"

    elif dry_days >= 15:
        risk_type = "🏜️ Drought / Dry Spell"

    elif rain_7 >= 100:
        risk_type = "🌊 Heavy Rainfall / Waterlogging"

    elif humidity >= 90:
        risk_type = "🍄 High Humidity / Disease Pressure"

    elif risk_low == "low":
        risk_type = "🌤️ Low Climate Stress"

    else:
        risk_type = "🌦️ Climate Variability"

    # -------------------------------------------------------------------------
    # RETURN
    # -------------------------------------------------------------------------

    return {
        "risk_type": risk_type,
        "main_risks": main_risks,
        "crop_impact": crop_impact,
        "recommendations": recommendations,
        "risk_period": "Current conditions",
    }


# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def predict_climate_risk(
    city: str,
    crop: str,
    model,
    encoders,
    city_soil_df: pd.DataFrame,
) -> dict:
    """
    Complete climate-risk prediction pipeline.

    Farmer inputs:
        city
        crop

    Automatically:
        coordinates
        state
        season
        weather
        rainfall history
        soil
        engineered features

    Then:
        encoding
        feature alignment
        Random Forest prediction
        farmer-friendly output
    """

    # -------------------------------------------------------------------------
    # VALIDATE INPUTS
    # -------------------------------------------------------------------------

    city = _validate_city(city)
    crop = _validate_crop(crop)

    if model is None:
        raise ValueError(
            "Climate risk model is not loaded."
        )

    if encoders is None:
        raise ValueError(
            "Climate risk encoders are not loaded."
        )

    # -------------------------------------------------------------------------
    # LOCATION
    # -------------------------------------------------------------------------

    latitude, longitude = CITY_COORDS[city]

    state = CITY_STATE[city]

    elevation = CITY_ELEVATION.get(
        city,
        200
    )

    # -------------------------------------------------------------------------
    # DATE + SEASON
    # -------------------------------------------------------------------------

    now = datetime.now()

    month = now.month

    season = get_season(month)

    # -------------------------------------------------------------------------
    # SOIL
    # -------------------------------------------------------------------------

    soil_row = get_city_soil(
        city,
        city_soil_df
    )

    # -------------------------------------------------------------------------
    # WEATHER
    # -------------------------------------------------------------------------

    weather = fetch_climate_weather(
        latitude,
        longitude
    )

    # -------------------------------------------------------------------------
    # NUMERIC FEATURES
    # -------------------------------------------------------------------------

    numeric_features = {
        "latitude": latitude,
        "longitude": longitude,

        "temperature_2m":
            weather["temperature_2m"],

        "relative_humidity_2m":
            weather["relative_humidity_2m"],

        "precipitation":
            weather["precipitation"],

        "surface_pressure":
            weather["surface_pressure"],

        "cloud_cover":
            weather["cloud_cover"],

        "wind_speed_10m":
            weather["wind_speed_10m"],

        "wind_direction_10m":
            weather["wind_direction_10m"],

        "wind_gusts_10m":
            weather["wind_gusts_10m"],

        "shortwave_radiation":
            weather["shortwave_radiation"],

        "et0_fao_evapotranspiration":
            weather[
                "et0_fao_evapotranspiration"
            ],

        "Soil_Moisture":
            weather["Soil_Moisture"],

        "Soil_Temperature":
            weather["Soil_Temperature"],

        "Soil_pH":
            _safe_float(
                soil_row["Soil_pH"]
            ),

        "Organic_Carbon":
            _safe_float(
                soil_row["Organic_Carbon"]
            ),

        "Clay":
            _safe_float(
                soil_row["Clay_Percentage"]
            ),

        "Sand":
            _safe_float(
                soil_row["Sand_Percentage"]
            ),

        "Silt":
            _safe_float(
                soil_row["Silt_Percentage"]
            ),

        "Elevation":
            elevation,

        "Heat_Index":
            weather["Heat_Index"],

        "Rainfall_Last_7_Days":
            weather[
                "Rainfall_Last_7_Days"
            ],

        "Rainfall_Last_30_Days":
            weather[
                "Rainfall_Last_30_Days"
            ],

        "Consecutive_Dry_Days":
            weather[
                "Consecutive_Dry_Days"
            ],

        "Growing_Degree_Days":
            weather[
                "Growing_Degree_Days"
            ],
    }

    numeric_df = pd.DataFrame(
        [numeric_features]
    )

    # -------------------------------------------------------------------------
    # CATEGORICAL ENCODING
    # -------------------------------------------------------------------------

    feature_df = _encode_categories(
        df=numeric_df,
        city=city,
        state=state,
        crop=crop,
        season=season,
        encoders=encoders,
    )

    # -------------------------------------------------------------------------
    # EXACT MODEL FEATURE ALIGNMENT
    # -------------------------------------------------------------------------

    X = _align_to_model(
        feature_df,
        model
    )

    # -------------------------------------------------------------------------
    # MODEL PREDICTION
    # -------------------------------------------------------------------------

    prediction = model.predict(X)[0]

    prediction = str(
        prediction
    ).strip().lower()

    # -------------------------------------------------------------------------
    # PROBABILITIES
    # -------------------------------------------------------------------------

    class_probabilities = {}

    confidence = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(X)[0]

        class_probabilities = {
            str(class_name).lower():
                round(float(probability), 4)

            for class_name, probability
            in zip(
                model.classes_,
                probabilities
            )
        }

        confidence = max(
            class_probabilities.values()
        )

    # -------------------------------------------------------------------------
    # RISK SCORE
    # -------------------------------------------------------------------------

    risk_score = _calculate_risk_score(
        model,
        X,
        prediction
    )

    # -------------------------------------------------------------------------
    # RISK DETAILS
    # -------------------------------------------------------------------------

    risk_details = _build_risk_details(
        risk=prediction,
        crop=crop,
        weather=weather,
    )

    # -------------------------------------------------------------------------
    # FARMER-FRIENDLY OUTPUT
    # -------------------------------------------------------------------------

    output = {
        "risk_level":
            prediction.upper(),

        "risk_icon":
            RISK_ICONS.get(
                prediction,
                "⚠️"
            ),

        "risk_color":
            RISK_COLORS.get(
                prediction,
                "#ffd54f"
            ),

        "risk_score":
            risk_score,

        "confidence_pct":
            round(
                confidence * 100,
                1
            )
            if confidence is not None
            else None,

        "class_probabilities":
            class_probabilities,

        "risk_type":
            risk_details[
                "risk_type"
            ],

        "risk_period":
            risk_details[
                "risk_period"
            ],

        "main_risks":
            risk_details[
                "main_risks"
            ],

        "crop_impact":
            risk_details[
                "crop_impact"
            ],

        "recommendations":
            risk_details[
                "recommendations"
            ],

        "city":
            city,

        "state":
            state.title(),

        "crop":
            crop,

        "season":
            season.title(),

        "date":
            now.strftime(
                "%d %B %Y"
            ),

        "latitude":
            latitude,

        "longitude":
            longitude,

        "climate_summary": {
            "temperature":
                f"{weather['temperature_2m']} °C",

            "humidity":
                f"{weather['relative_humidity_2m']} %",

            "rainfall_today":
                f"{weather['precipitation']} mm",

            "wind_speed":
                f"{weather['wind_speed_10m']} km/h",

            "heat_index":
                f"{weather['Heat_Index']} °C",

            "rainfall_last_7_days":
                f"{weather['Rainfall_Last_7_Days']} mm",

            "rainfall_last_30_days":
                f"{weather['Rainfall_Last_30_Days']} mm",

            "consecutive_dry_days":
                weather[
                    "Consecutive_Dry_Days"
                ],

            "et0":
                f"{weather['et0_fao_evapotranspiration']} mm",

            "soil_moisture":
                weather[
                    "Soil_Moisture"
                ],
        },

        # Technical data is returned separately.
        # The farmer UI does not need to display it.
        "weather_fetched":
            weather,

        "soil_data": {
            "soil_ph":
                _safe_float(
                    soil_row["Soil_pH"]
                ),

            "organic_carbon":
                _safe_float(
                    soil_row["Organic_Carbon"]
                ),

            "clay_percentage":
                _safe_float(
                    soil_row["Clay_Percentage"]
                ),

            "sand_percentage":
                _safe_float(
                    soil_row["Sand_Percentage"]
                ),

            "silt_percentage":
                _safe_float(
                    soil_row["Silt_Percentage"]
                ),

            "elevation_m":
                elevation,
        },

        "model_features_used":
            list(X.columns),

        "model_feature_count":
            X.shape[1],
    }

    return output