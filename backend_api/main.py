from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
import sys
import os
import traceback
import pandas as pd
from typing import Optional, Dict, Any, List

# Ensure parent directory is in sys.path so `backend` package can be imported
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import backend.weather as weather_mod
import backend.predict as irrigation_mod
import backend.yield_estimator as yield_mod
import backend.climate_risk as climate_mod
import backend.crop_recommendation as crop_mod
import backend.fusion as fusion_mod
import backend.soil_weather_service as soil_weather_mod
import backend.database as db_mod

app = FastAPI(
    title="Agri Fusion Backend API",
    description="Unified REST API for Agricultural AI Analysis, Predictions, Weather, Soil & Auth",
    version="2.0"
)

# Allow local Streamlit frontend & web clients to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    full_name: str
    phone: str
    password: str

class LoginRequest(BaseModel):
    phone: str
    password: str

class IrrigationRequest(BaseModel):
    city: str
    crop: str
    growth_stage: str = "Development"
    farm_size_acres: float = 2.0
    pump_lpm: Optional[float] = 100.0
    irrigation_method: str = "Drip"
    already_irrigated_today: bool = False

class ClimateRiskRequest(BaseModel):
    city: str
    crop: str = "Rice"
    month: Optional[int] = None

class CropRecommendationRequest(BaseModel):
    city: str

class YieldRequest(BaseModel):
    district: str
    state: str
    season: str
    crop: str
    area_acres: float

class MarketRequest(BaseModel):
    commodity: str
    state: str
    district: str
    day: int = 1
    month: int = 1
    year: int = 2026
    quarter: int = 1
    arrival_qty: float = 100.0

class FusionRequest(BaseModel):
    city: str
    farm_size_acres: float = 2.0

# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "Agri Fusion Backend API",
        "version": "2.0"
    }

# ── Auth Endpoints ──

@app.post("/api/auth/register")
async def register_farmer_api(req: RegisterRequest):
    try:
        success = db_mod.register_farmer(
            full_name=req.full_name,
            phone=req.phone,
            password_hash=req.password
        )
        if success:
            return {"status": "success", "message": "Farmer registered successfully"}
        else:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Registration failed or user already exists"})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/login")
async def login_farmer_api(req: LoginRequest):
    try:
        farmer = db_mod.get_farmer(req.phone)
        if farmer:
            db_mod.update_last_login(req.phone)
            return {"status": "success", "farmer": farmer}
        else:
            raise HTTPException(status_code=404, detail="Farmer account not found")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── Weather & Soil Endpoints ──

@app.get("/api/weather")
async def get_weather_api(lat: float = Query(..., description="Latitude"), lon: float = Query(..., description="Longitude")):
    try:
        w_data = weather_mod.get_weather(lat, lon)
        return w_data
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/weather/forecast")
async def get_forecast_api(lat: float = Query(...), lon: float = Query(...), days: int = Query(7)):
    try:
        fc = weather_mod.get_forecast(lat, lon, days)
        return fc
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/soil")
async def get_soil_api(city: str = Query(...)):
    try:
        soil = soil_weather_mod.get_city_soil(city)
        return soil
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── Prediction Endpoints ──

@app.post("/api/predict/irrigation")
@app.post("/predict/irrigation")
async def predict_irrigation_api(req: IrrigationRequest):
    try:
        coords = weather_mod.CITY_COORDS.get(req.city, (17.38, 78.47))
        weather = weather_mod.get_weather(coords[0], coords[1])
        
        user_input = {
            "city": req.city,
            "crop": req.crop,
            "growth_stage": req.growth_stage,
            "farm_size": req.farm_size_acres,
            "farm_size_unit": "Acres",
            "pump_lpm": req.pump_lpm or 100.0,
            "irrigation_method": req.irrigation_method,
            "already_irrigated_today": req.already_irrigated_today,
        }
        res = irrigation_mod.predict(user_input, weather)
        return JSONResponse(content=res)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/climate-risk")
@app.post("/predict/climate-risk")
async def predict_climate_risk_api(req: ClimateRiskRequest):
    try:
        M = fusion_mod.get_models()
        res = climate_mod.predict_climate_risk(
            city=req.city,
            crop=req.crop,
            month=req.month,
            models=M
        )
        return JSONResponse(content=res)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/crop-recommendation")
@app.post("/predict/crop-recommendation")
async def predict_crop_recommendation_api(req: CropRecommendationRequest):
    try:
        M = fusion_mod.get_models()
        res = crop_mod.recommend_crop(
            city=req.city,
            model=M["crop_model"],
            feature_columns=M["crop_columns"],
            label_encoder=M["crop_label_enc"],
            crop_encoders=M["crop_encoders"]
        )
        return JSONResponse(content=res)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/yield")
@app.post("/predict/yield")
async def predict_yield_api(req: YieldRequest):
    try:
        M = fusion_mod.get_models()
        area_ha = float(req.area_acres) * 0.4047
        
        city_soil_df = None
        csv_path = BASE_DIR / "backend" / "city_soil_lookup.csv"
        if csv_path.exists():
            city_soil_df = pd.read_csv(csv_path)

        res = yield_mod.predict_yield(
            district=req.district,
            state=req.state,
            season=req.season,
            crop=req.crop,
            area=area_ha,
            model=M["yield_model"],
            encoder=M["yield_enc"],
            state_map=M["yield_state_map"],
            city_soil_df=city_soil_df
        )
        return JSONResponse(content=res)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/market")
@app.post("/predict/market")
async def predict_market_api(req: MarketRequest):
    try:
        M = fusion_mod.get_models()
        res = fusion_mod._predict_market_price(
            city=req.district,
            crop=req.commodity,
            arrival_qty_quintals=req.arrival_qty,
            M=M
        )
        total_val = float(res.get("predicted_price", 2000.0)) * req.arrival_qty
        return JSONResponse(content={
            "price_per_quintal": res.get("predicted_price", 2000.0),
            "total_value_inr": total_val,
            "details": res
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/fusion")
@app.post("/predict/fusion")
async def predict_fusion_api(req: FusionRequest):
    try:
        city_soil_df = pd.read_csv(BASE_DIR / "backend" / "city_soil_lookup.csv")
        res = fusion_mod.run_fusion(
            city=req.city,
            farm_size_acres=req.farm_size_acres,
            city_soil_df=city_soil_df
        )
        return JSONResponse(content=res)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
