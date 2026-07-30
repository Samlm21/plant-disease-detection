"""
Integration tests for the FastAPI application endpoints.
"""
import os
import pytest
import json
import tempfile
from PIL import Image
from io import BytesIO
from unittest.mock import patch, MagicMock

# We patch the predictor initialization before importing the app
# to avoid requiring trained model weights during testing
import sys


@pytest.fixture
def client():
    """Create FastAPI test client with mocked predictor."""
    mock_pred = MagicMock()
    mock_pred.class_names = ["Tomato___healthy", "Tomato___Late_blight"]
    mock_pred.model_name = "resnet18"
    mock_pred.device = "cpu"
    mock_pred.validate_image_quality.return_value = (True, "")
    mock_pred.predict.return_value = {
        "success": True,
        "class_raw": "Tomato___healthy",
        "disease": "Tomato (Healthy)",
        "confidence": 0.97,
        "recommendation": "Plant looks healthy!",
        "model_used": "resnet18"
    }

    with patch("app.main.get_predictor", return_value=mock_pred):
        from fastapi.testclient import TestClient
        from app.main import app
        yield TestClient(app)


class TestHealthEndpoint:
    """Tests for the GET / health check endpoint."""

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_app_name(self, client):
        data = client.get("/").json()
        assert "Plant Disease Detection" in data["app"]

    def test_root_status_online(self, client):
        data = client.get("/").json()
        assert data["status"] == "online"


class TestPredictEndpoint:
    """Tests for the POST /predict inference endpoint."""

    def _make_image_bytes(self, width=224, height=224, color=(100, 180, 80)):
        """Helper: generates a JPEG image as bytes."""
        img = Image.new("RGB", (width, height), color=color)
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf

    def test_predict_valid_image(self, client):
        img_bytes = self._make_image_bytes()
        response = client.post(
            "/predict",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "disease" in data
        assert "confidence" in data
        assert "recommendation" in data

    def test_predict_rejects_non_image(self, client):
        """Uploading a text file should return 400."""
        response = client.post(
            "/predict",
            files={"file": ("document.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    def test_predict_confidence_range(self, client):
        """Confidence should be a float between 0 and 1."""
        img_bytes = self._make_image_bytes()
        data = client.post(
            "/predict",
            files={"file": ("leaf.jpg", img_bytes, "image/jpeg")}
        ).json()
        assert 0.0 <= data["confidence"] <= 1.0
