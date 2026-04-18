"""
test_app.py  –  pytest suite for the Taxi Tip Predictor API
Run with:  pytest test_app.py -v
"""
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared valid payload
# ---------------------------------------------------------------------------
VALID_TRIP = {
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
    "RatecodeID": 1,
}


# ---------------------------------------------------------------------------
# 1. Root endpoint
# ---------------------------------------------------------------------------
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# ---------------------------------------------------------------------------
# 2. Health check
# ---------------------------------------------------------------------------
def test_health_returns_healthy():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data


# ---------------------------------------------------------------------------
# 3. Single prediction – happy path
# ---------------------------------------------------------------------------
def test_predict_valid_input():
    response = client.post("/predict", json=VALID_TRIP)
    assert response.status_code == 200
    data = response.json()
    assert "tip_amount" in data
    assert "prediction_id" in data
    assert "model_version" in data
    assert isinstance(data["tip_amount"], float)
    assert data["tip_amount"] >= 0


# ---------------------------------------------------------------------------
# 4. Single prediction – missing required field → 422
# ---------------------------------------------------------------------------
def test_predict_missing_field():
    bad = {k: v for k, v in VALID_TRIP.items() if k != "fare_amount"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 5. Single prediction – wrong type → 422
# ---------------------------------------------------------------------------
def test_predict_invalid_type():
    bad = {**VALID_TRIP, "trip_distance": "not_a_number"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 6. Single prediction – out-of-range value → 422
# ---------------------------------------------------------------------------
def test_predict_out_of_range_hour():
    bad = {**VALID_TRIP, "pickup_hour": 25}   # max is 23
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 7. Batch prediction – happy path
# ---------------------------------------------------------------------------
def test_batch_prediction():
    payload = {"records": [VALID_TRIP] * 5}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 5
    assert len(data["predictions"]) == 5
    assert "processing_time_ms" in data


# ---------------------------------------------------------------------------
# 8. Batch prediction – exceeds 100-record limit → 422
# ---------------------------------------------------------------------------
def test_batch_exceeds_limit():
    payload = {"records": [VALID_TRIP] * 101}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 9. Model info endpoint
# ---------------------------------------------------------------------------
def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "metrics" in data
    assert "model_name" in data


# ---------------------------------------------------------------------------
# 10. Edge case – zero-distance trip (gt=0 should reject it) → 422
# ---------------------------------------------------------------------------
def test_zero_distance_rejected():
    bad = {**VALID_TRIP, "trip_distance": 0.0}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 11. Edge case – extreme but valid fare
# ---------------------------------------------------------------------------
def test_extreme_fare_valid():
    high_fare = {**VALID_TRIP, "fare_amount": 999.0, "fare_per_mile": 100.0, "fare_per_minute": 50.0}
    response = client.post("/predict", json=high_fare)
    assert response.status_code == 200
    assert response.json()["tip_amount"] >= 0
