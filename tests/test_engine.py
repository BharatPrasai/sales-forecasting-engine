import pytest
import numpy as np
import pandas as pd
import os
import joblib
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

# 1. Supply Chain & Metrics Unit Tests
def calculate_wape(y_true, y_pred):
    return (np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)) * 100.0

def test_wape_calculation_exact():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])  # Errors: 10 + 10 + 0 = 20 / Total 600
    expected_wape = (20.0 / 600.0) * 100.0
    assert np.isclose(calculate_wape(y_true, y_pred), expected_wape)

def test_model_artifact_exists():
    model_path = "data/rf_forecast_model.joblib"
    assert os.path.exists(model_path), f"Artifact {model_path} missing!"
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Loaded artifact does not implement predict()."

# 2. FastAPI Endpoint Validation
def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "cutoff_date" in data
    assert data["historical_records"] > 0

def test_api_forecast_endpoint_success():
    payload = {
        "horizon_days": 7,
        "lead_time_days": 5,
        "service_level_z": 1.645,
        "planned_promotions": []
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["horizon_days"] == 7
    assert len(data["daily_breakdown"]) == 7
    assert data["directives"]["total_projected_demand"] > 0
    assert data["directives"]["reorder_point"] > 0
    assert data["directives"]["safety_stock"] > 0

def test_api_forecast_validation_bounds():
    # Horizon days must be between 1 and 90
    invalid_payload = {
        "horizon_days": 150,
        "lead_time_days": 7
    }
    response = client.post("/forecast", json=invalid_payload)
    assert response.status_code == 422