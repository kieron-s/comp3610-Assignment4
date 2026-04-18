# COMP 3610 – Assignment 4: MLOps & Model Deployment

Deploys a **Linear Regression** model (trained on NYC Yellow Taxi data) as a containerised REST API, with MLflow experiment tracking.

## Prerequisites

- Python
- Docker Desktop
- Git

## Project Structure

```
assignment4/
├── assignment4.ipynb     
├── app.py                # FastAPI application
├── test_app.py           # pytest test suite
├── Dockerfile            # Container recipe
├── docker-compose.yml    # Orchestrates api + mlflow services
├── requirements.txt      # Python dependencies
├── README.md
├── .gitignore
├── .dockerignore
└── models/               
    ├── linear_regression.pkl
    └── scaler.pkl
```

## Quick Start

### 1 – Install dependencies (local dev)

```bash
pip install -r requirements.txt
```

### 2 – Train and save models

Run all cells in `assignment4.ipynb` (Part 1 saves `models/linear_regression.pkl` and `models/scaler.pkl`).

### 3 – Run API locally

```bash
uvicorn app:app --reload --port 8000
```

Visit http://localhost:8000/docs for the interactive Swagger UI.

### 4 – Run tests

```bash
pytest test_app.py -v
```

### 5 – Run with Docker Compose

**API image size:** 1.26 GB (286 MB content size)

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Prediction API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

### 6 – Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_hour": 14, "pickup_day_of_week": 2, "is_weekend": 0,
    "trip_duration_minutes": 12.5, "trip_speed_mph": 18.4,
    "log_trip_distance": 1.35, "trip_distance": 3.2,
    "passenger_count": 1, "fare_amount": 13.5,
    "fare_per_mile": 4.22, "fare_per_minute": 1.08,
    "extra": 0.5, "mta_tax": 0.5, "tolls_amount": 0.0,
    "improvement_surcharge": 0.3,
    "pickup_borough_enc": 3, "dropoff_borough_enc": 2, "RatecodeID": 1
  }'
```

### 7 – Shut down

```bash
docker compose down
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/linear_regression.pkl` | Path to saved model |
| `SCALER_PATH` | `models/scaler.pkl` | Path to saved scaler |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URI |
