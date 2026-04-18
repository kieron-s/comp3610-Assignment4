import os
import time
import uuid
import joblib
import numpy as np
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


ml_model = None
scaler = None
start_time = time.time()

MODEL_VERSION = "1.0.0"
MODEL_NAME = "taxi-tip-regressor"
FEATURE_NAMES = [
    "pickup_hour", "pickup_day_of_week", "is_weekend",
    "trip_duration_minutes", "trip_speed_mph", "log_trip_distance",
    "trip_distance", "passenger_count", "fare_amount",
    "fare_per_mile", "fare_per_minute", "extra", "mta_tax",
    "tolls_amount", "improvement_surcharge",
    "pickup_borough_enc", "dropoff_borough_enc", "RatecodeID",
]

_model_path = os.getenv("MODEL_PATH", "models/linear_regression.pkl")
_scaler_path = os.getenv("SCALER_PATH", "models/scaler.pkl")
ml_model = joblib.load(_model_path)
scaler = joblib.load(_scaler_path)
print(f"Model loaded from {_model_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global start_time
    start_time = time.time()
    print("Lifespan startup complete.")
    yield
    print("Shutting down.")


app = FastAPI(title="Taxi Tip Predictor", version=MODEL_VERSION, lifespan=lifespan)


class TripInput(BaseModel):
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of pickup (0-23)")
    pickup_day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if weekend, 0 otherwise")
    trip_duration_minutes: float = Field(..., gt=0, description="Trip duration in minutes")
    trip_speed_mph: float = Field(..., gt=0, description="Average trip speed in mph")
    log_trip_distance: float = Field(..., ge=0, description="Natural log of trip distance")
    trip_distance: float = Field(..., gt=0, le=200, description="Trip distance in miles")
    passenger_count: int = Field(..., ge=1, le=9, description="Number of passengers")
    fare_amount: float = Field(..., gt=0, le=1000, description="Fare amount in dollars")
    fare_per_mile: float = Field(..., gt=0, description="Fare per mile")
    fare_per_minute: float = Field(..., gt=0, description="Fare per minute")
    extra: float = Field(default=0.0, ge=0, description="Extra charges")
    mta_tax: float = Field(default=0.5, ge=0, description="MTA tax")
    tolls_amount: float = Field(default=0.0, ge=0, description="Tolls amount")
    improvement_surcharge: float = Field(default=0.3, ge=0, description="Improvement surcharge")
    pickup_borough_enc: int = Field(..., ge=0, description="Encoded pickup borough")
    dropoff_borough_enc: int = Field(..., ge=0, description="Encoded dropoff borough")
    RatecodeID: int = Field(..., ge=1, le=6, description="Rate code ID (1-6)")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "pickup_hour": 14,
                "pickup_day_of_week": 2,
                "is_weekend": 0,
                "trip_duration_minutes": 12.5,
                "trip_speed_mph": 18.4,
                "log_trip_distance": 1.35,
                "trip_distance": 3.2,
                "passenger_count": 1,
                "fare_amount": 13.5,
                "fare_per_mile": 4.22,
                "fare_per_minute": 1.08,
                "extra": 0.5,
                "mta_tax": 0.5,
                "tolls_amount": 0.0,
                "improvement_surcharge": 0.3,
                "pickup_borough_enc": 3,
                "dropoff_borough_enc": 2,
                "RatecodeID": 1
            }]
        }
    }


class PredictionResponse(BaseModel):
    tip_amount: float
    prediction_id: str
    model_version: str


class BatchInput(BaseModel):
    records: List[TripInput] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    processing_time_ms: float


def _predict_one(record: TripInput) -> float:
    features = np.array([[getattr(record, f) for f in FEATURE_NAMES]])
    features_scaled = scaler.transform(features)
    pred = ml_model.predict(features_scaled)[0]
    return round(float(max(pred, 0.0)), 2)  


@app.get("/")
def root():
    return {"message": "Taxi Tip Predictor API is running", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: TripInput):
    tip = _predict_one(input_data)
    return PredictionResponse(
        tip_amount=tip,
        prediction_id=str(uuid.uuid4()),
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput):
    t0 = time.time()
    predictions = []
    for record in batch.records:
        tip = _predict_one(record)
        predictions.append(PredictionResponse(
            tip_amount=tip,
            prediction_id=str(uuid.uuid4()),
            model_version=MODEL_VERSION,
        ))
    elapsed = (time.time() - t0) * 1000
    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        processing_time_ms=round(elapsed, 2),
    )


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "model_version": MODEL_VERSION,
        "uptime_seconds": round(time.time() - start_time, 1),
    }


@app.get("/model/info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "features": FEATURE_NAMES,
        "metrics": {
            "MAE": 1.21,
            "RMSE": 2.39,
            "R2": 0.62,
        },
        "trained_on": "NYC Yellow Taxi - January 2024 (credit card payments)",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again.",
        },
    )
