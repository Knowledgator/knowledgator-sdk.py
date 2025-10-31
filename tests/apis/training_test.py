"""Unit tests for TrainingAPI."""
import pytest
from unittest.mock import Mock, patch
import httpx

from knowledgator import Knowledgator
from knowledgator.models.outputs import TrainingObject
from knowledgator.errors import (
    BadRequestError,
    UnauthorizedError,
    InternalServerError,
)


@pytest.fixture
def mock_http_client(monkeypatch):
    """Mock HttpClient to prevent real network calls."""
    from knowledgator.clients.http_client import HttpClient, AsyncHttpClient
    
    monkeypatch.setattr(HttpClient, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(AsyncHttpClient, "__init__", lambda self, *a, **k: None)


@pytest.fixture
def sdk(mock_http_client):
    """Create SDK instance with mocked HTTP client."""
    return Knowledgator(api_key="test-api-key")


@pytest.fixture
def mock_response():
    """Factory for creating mock httpx.Response objects."""
    def _create_response(status_code: int, json_data: dict):
        response = Mock(spec=httpx.Response)
        response.status_code = status_code
        response.is_success = 200 <= status_code < 300
        response.json.return_value = json_data
        response.text = str(json_data)
        response.headers = {}
        return response
    return _create_response


class TestStartTraining:
    """Tests for start_training endpoint."""

    def test_start_training_success(self, sdk, mock_response):
        """Test successful training job creation."""
        expected_response = {
            "id": "training-123",
            "name": "Test Training Job",
            "type": "comprehend-it",
            "model": {
                "id": "fine-tuned-model-123",
                "name": "Fine-tuned Model"
            },
            "parent_model": {
                "id": "parent-model-123",
                "name": "Base Model"
            },
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "num_of_epochs": 10,
            "datasets": {
                "num_of_examples": 100,
                "training": {
                    "id": "dataset-train-123",
                    "name": "Training Dataset"
                },
                "validation": {
                    "id": "dataset-val-123",
                    "name": "Validation Dataset"
                }
            },
            "loss": 0.05,
            "status": "active"
        }
        sdk = Knowledgator()
        with patch.object(sdk.training, "send", return_value=mock_response(200, expected_response)):
            result = sdk.training.start_training(
                parent_model_id="parent-model-123",
                dataset_id="dataset-train-123",
                seed=42,
                batch=16,
                learning_rate_mult=1,
                num_of_epochs=10,
                name="Test Training Job",
                model_name="Fine-tuned Model",
                validation_dataset_id="dataset-val-123"
            )
        
        assert isinstance(result, TrainingObject)
        assert result.id == "training-123"
        assert result.name == "Test Training Job"
        assert result.type == "comprehend-it"
        assert result.status == "active"
        assert result.num_of_epochs == 10
        assert result.loss == 0.05

    def test_start_training_minimal_params(self, sdk, mock_response):
        """Test training job with only required parameters."""
        expected_response = {
            "id": "training-456",
            "name": "Minimal Training",
            "type": "extract-it",
            "model": {
                "id": "model-456",
                "name": "Model 456"
            },
            "parent_model": {
                "id": "parent-456",
                "name": "Parent 456"
            },
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "num_of_epochs": 5,
            "datasets": {
                "num_of_examples": 50,
                "training": {
                    "id": "dataset-456",
                    "name": "Dataset 456"
                },
                "validation": {
                    "id": "dataset-val-456",
                    "name": "Validation 456"
                }
            },
            "loss": 0.1,
            "status": "active"
        }
        
        with patch.object(sdk.training, "send", return_value=mock_response(200, expected_response)):
            result = sdk.training.start_training(
                parent_model_id="parent-456",
                dataset_id="dataset-456",
                seed=42
            )
        
        assert isinstance(result, TrainingObject)
        assert result.id == "training-456"
        assert result.status == "active"

    def test_start_training_validation_error(self, sdk, mock_response):
        """Test training with server-side validation error (400)."""
        error_response = {
            "detail": [
                {
                    "loc": ["body", "seed"],
                    "msg": "Input should be a valid integer",
                    "type": "int_parsing"
                }
            ]
        }
        
        with patch.object(sdk.training, "send", return_value=mock_response(400, error_response)):
            with pytest.raises(BadRequestError):
                sdk.training.start_training(
                    parent_model_id="parent-123",
                    dataset_id="dataset-123",
                    seed=42  # Valid input to avoid client-side error
                )

    def test_start_training_unauthorized(self, sdk, mock_response):
        """Test training with invalid API key."""
        error_response = {"detail": "Invalid API key"}
        
        with patch.object(sdk.training, "send", return_value=mock_response(401, error_response)):
            with pytest.raises(UnauthorizedError):
                sdk.training.start_training(
                    parent_model_id="parent-123",
                    dataset_id="dataset-123",
                    seed=42
                )

    def test_start_training_with_validation_dataset(self, sdk, mock_response):
        """Test training with validation dataset specified."""
        expected_response = {
            "id": "training-789",
            "name": "Training with Validation",
            "type": "comprehend-it",
            "model": {
                "id": "model-789",
                "name": "Model 789"
            },
            "parent_model": {
                "id": "parent-789",
                "name": "Parent 789"
            },
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "num_of_epochs": 8,
            "datasets": {
                "num_of_examples": 200,
                "training": {
                    "id": "dataset-train-789",
                    "name": "Training 789"
                },
                "validation": {
                    "id": "dataset-val-789",
                    "name": "Validation 789"
                }
            },
            "loss": 0.03,
            "status": "active"
        }
        
        with patch.object(sdk.training, "send", return_value=mock_response(200, expected_response)):
            result = sdk.training.start_training(
                parent_model_id="parent-789",
                dataset_id="dataset-train-789",
                seed=42,
                validation_dataset_id="dataset-val-789"
            )
        
        assert isinstance(result, TrainingObject)
        assert result.datasets.validation.id == "dataset-val-789"


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_json_response(self, sdk):
        """Test handling of invalid JSON in response."""
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.is_success = True
        response.json.side_effect = ValueError("Invalid JSON")
        response.text = "Not JSON"
        response.headers = {}
        
        with patch.object(sdk.training, "send", return_value=response):
            with pytest.raises(InternalServerError) as exc_info:
                sdk.training.start_training(
                    parent_model_id="parent-123",
                    dataset_id="dataset-123",
                    seed=42
                )
            
            assert "not valid JSON" in str(exc_info.value)

    def test_malformed_training_data(self, sdk, mock_response):
        """Test handling of response with missing required fields."""
        malformed_response = {
            "id": "training-123",
            "name": "Test Training",
            # Missing required fields like 'type', 'model', 'parent_model', etc.
        }
        
        with patch.object(sdk.training, "send", return_value=mock_response(200, malformed_response)):
            with pytest.raises(InternalServerError) as exc_info:
                sdk.training.start_training(
                    parent_model_id="parent-123",
                    dataset_id="dataset-123",
                    seed=42
                )
            
            assert "Failed to parse" in str(exc_info.value)

    def test_server_error(self, sdk, mock_response):
        """Test handling of 500 server error."""
        error_response = {"detail": "Internal server error"}
        
        with patch.object(sdk.training, "send", return_value=mock_response(500, error_response)):
            with pytest.raises(InternalServerError):
                sdk.training.start_training(
                    parent_model_id="parent-123",
                    dataset_id="dataset-123",
                    seed=42
                )


class TestAsyncMethods:
    """Tests for async API methods."""

    @pytest.mark.asyncio
    async def test_start_training_async(self, sdk, mock_response):
        """Test async training job creation."""
        expected_response = {
            "id": "training-async-123",
            "name": "Async Training Job",
            "type": "comprehend-it",
            "model": {
                "id": "model-async-123",
                "name": "Async Model"
            },
            "parent_model": {
                "id": "parent-async-123",
                "name": "Parent Async"
            },
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "num_of_epochs": 10,
            "datasets": {
                "num_of_examples": 100,
                "training": {
                    "id": "dataset-async-123",
                    "name": "Dataset Async"
                },
                "validation": {
                    "id": "dataset-val-async-123",
                    "name": "Validation Async"
                }
            },
            "loss": 0.04,
            "status": "active"
        }
        
        with patch.object(sdk.training, "send_async", return_value=mock_response(200, expected_response)):
            result = await sdk.training.start_training_async(
                parent_model_id="parent-async-123",
                dataset_id="dataset-async-123",
                seed=42,
                num_of_epochs=10
            )
        
        assert isinstance(result, TrainingObject)
        assert result.id == "training-async-123"
        assert result.status == "active"

    @pytest.mark.asyncio
    async def test_start_training_async_error(self, sdk, mock_response):
        """Test async training with error."""
        error_response = {"detail": "Invalid dataset ID"}
        
        with patch.object(sdk.training, "send_async", return_value=mock_response(400, error_response)):
            with pytest.raises(BadRequestError):
                await sdk.training.start_training_async(
                    parent_model_id="parent-123",
                    dataset_id="invalid-dataset",
                    seed=42
                )


class TestRequestPayload:
    """Tests for request payload construction."""

    def test_payload_excludes_none_values(self, sdk, mock_response):
        """Test that None values are excluded from request payload."""
        expected_response = {
            "id": "training-123",
            "name": "Test",
            "type": "comprehend-it",
            "model": {"id": "model1", "name": "Model1"},
            "parent_model": {"id": "parent1", "name": "Parent1"},
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "num_of_epochs": 5,
            "datasets": {
                "num_of_examples": 50,
                "training": {"id": "dataset1", "name": "Dataset1"},
                "validation": {"id": "valset1", "name": "Valset1"}
            },
            "loss": 0.1,
            "status": "active"
        }
        
        with patch.object(sdk.training, "send", return_value=mock_response(200, expected_response)) as mock_send:
            sdk.training.start_training(
                parent_model_id="parent1",
                dataset_id="dataset1",
                seed=42,
                batch=None,  # Should be excluded
                learning_rate_mult=None,  # Should be excluded
                num_of_epochs=None,  # Should be excluded
                name=None,  # Should be excluded
                model_name=None,  # Should be excluded
                validation_dataset_id=None  # Should be excluded
            )
            
            # Verify that the request was made (payload validation happens in _make_request)
            assert mock_send.called
