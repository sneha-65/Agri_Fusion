import requests

# Fallback used whenever Open-Meteo is unreachable (no internet, DNS
# failure, timeout, firewall block) — keeps every caller working with a
# plausible regional value instead of crashing.
_FALLBACK_CURRENT = {
    "temperature": 27.5, "relative_humidity": 65.0,
    "wind_speed": 9.5, "solar_radiation": 550.0,
    "et0": 3.5, "rainfall": 0.13,
}


def get_weather(lat: float, lon: float) -> dict:
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation",
                "hourly": "et0_fao_evapotranspiration,precipitation",
                "timezone": "auto",
            }, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        c = data.get("current", {}) or {}
        h = data.get("hourly", {}) or {}
        et0_arr = h.get("et0_fao_evapotranspiration") or [3.5]
        rain_arr = h.get("precipitation") or [0.13]
        return {
            "temperature":       float(c.get("temperature_2m")) if c.get("temperature_2m") is not None else 27.5,
            "relative_humidity": float(c.get("relative_humidity_2m")) if c.get("relative_humidity_2m") is not None else 65.0,
            "wind_speed":        float(c.get("wind_speed_10m")) if c.get("wind_speed_10m") is not None else 9.5,
            "solar_radiation":   float(c.get("shortwave_radiation")) if c.get("shortwave_radiation") is not None else 550.0,
            "et0":               float(et0_arr[0]) if et0_arr and et0_arr[0] is not None else 3.5,
            "rainfall":          float(rain_arr[0]) if rain_arr and rain_arr[0] is not None else 0.13,
        }
    except (requests.exceptions.RequestException, KeyError, IndexError, TypeError):
        return dict(_FALLBACK_CURRENT)


def get_forecast(lat: float, lon: float, days: int = 7) -> list:
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,precipitation_sum,et0_fao_evapotranspiration,shortwave_radiation_sum",
                "hourly": "relative_humidity_2m,wind_speed_10m",
                "forecast_days": days,
                "timezone": "auto",
            }, timeout=10
        )
        resp.raise_for_status()
        payload = resp.json()
        d = payload["daily"]
        h = payload["hourly"]
        return [
            {
                "date":              d["time"][i],
                "temperature":       d["temperature_2m_max"][i],
                "rainfall":          d["precipitation_sum"][i],
                "et0":               d["et0_fao_evapotranspiration"][i],
                "solar_radiation":   d["shortwave_radiation_sum"][i] * 1000 if d["shortwave_radiation_sum"][i] else 0,
                "relative_humidity": h["relative_humidity_2m"][i * 24],
                "wind_speed":        h["wind_speed_10m"][i * 24],
            }
            for i in range(len(d["time"]))
        ]
    except (requests.exceptions.RequestException, KeyError, IndexError, TypeError):
        # Flat fallback forecast — same shape as the live response, so
        # callers that iterate over N days don't need special-case handling.
        return [
            {
                "date": f"day+{i}", **_FALLBACK_CURRENT,
                "temperature": _FALLBACK_CURRENT["temperature"],
                "rainfall": _FALLBACK_CURRENT["rainfall"],
            }
            for i in range(days)
        ]


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