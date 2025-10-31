"""Unit tests for ModelsAPI."""
import pytest
from unittest.mock import Mock, patch
import httpx

from knowledgator import Knowledgator
from knowledgator.models.outputs import (
    ModelObject,
    ModelsListResponse,
    ModelTypesListResponse,
)
from knowledgator.errors import (
    BadRequestError,
    UnauthorizedError,
    NotFoundError,
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


class TestCreateModel:
    """Tests for create_model endpoint."""

    def test_create_model_success(self, sdk, mock_response):
        """Test successful model creation."""
        expected_response = {
            "id": "model-123",
            "name": "Test Model",
            "type": "comprehend-it",
            "description": "A test model",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "status": "active",
        }
        
        with patch.object(sdk.models, "send", return_value=mock_response(200, expected_response)):
            result = sdk.models.create_model(
                name="Test Model",
                type="comprehend-it",
                description="A test model"
            )
        
        assert isinstance(result, ModelObject)
        assert result.id == "model-123"
        assert result.name == "Test Model"
        assert result.type == "comprehend-it"
        assert result.status == "active"

    def test_create_model_validation_error(self, sdk, mock_response):
        """Test model creation with server-side validation error (400)."""
        error_response = {
            "detail": [
                {
                    "loc": ["body", "name"],
                    "msg": "String should have at least 3 characters",
                    "type": "string_too_short"
                }
            ]
        }
        
        with patch.object(sdk.models, "send", return_value=mock_response(400, error_response)):
            with pytest.raises(BadRequestError):
                sdk.models.create_model(
                    name="Valid Name",  # Avoid client-side validation error
                    type="comprehend-it"
                )

    def test_create_model_unauthorized(self, sdk, mock_response):
        """Test model creation with invalid API key."""
        error_response = {"detail": "Invalid API key"}
        
        with patch.object(sdk.models, "send", return_value=mock_response(401, error_response)):
            with pytest.raises(UnauthorizedError):
                sdk.models.create_model(
                    name="Test Model",
                    type="comprehend-it"
                )


class TestListModels:
    """Tests for list_models endpoint."""

    def test_list_models_success(self, sdk, mock_response):
        """Test successful models list retrieval."""
        expected_response = {
            "data": [
                {
                    "id": "model-1",
                    "name": "Model 1",
                    "type": "comprehend-it",
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z",
                    "status": "active",
                },
                {
                    "id": "model-2",
                    "name": "Model 2",
                    "type": "extract-it",
                    "description": "Test model 2",
                    "created_at": "2025-01-02T00:00:00Z",
                    "updated_at": "2025-01-02T00:00:00Z",
                    "status": "active",
                },
            ],
            "total": 2,
        }
        
        with patch.object(sdk.models, "send", return_value=mock_response(200, expected_response)):
            result = sdk.models.list_models()
        
        assert isinstance(result, ModelsListResponse)
        assert result.total == 2
        assert len(result.data) == 2
        assert result.data[0].id == "model-1"
        assert result.data[1].type == "extract-it"

    def test_list_models_empty(self, sdk, mock_response):
        """Test listing models when none exist."""
        expected_response = {
            "data": [],
            "total": 0,
        }
        
        with patch.object(sdk.models, "send", return_value=mock_response(200, expected_response)):
            result = sdk.models.list_models()
        
        assert isinstance(result, ModelsListResponse)
        assert result.total == 0
        assert len(result.data) == 0


class TestListModelTypes:
    """Tests for list_model_types endpoint."""

    def test_list_model_types_success(self, sdk, mock_response):
        """Test successful model types list retrieval."""
        expected_response = {
            "data": ["comprehend-it", "extract-it", "instruct-it"],
            "total": 3,
        }
        
        with patch.object(sdk.models, "send", return_value=mock_response(200, expected_response)):
            result = sdk.models.list_model_types()
        
        assert isinstance(result, ModelTypesListResponse)
        assert result.total == 3
        assert len(result.data) == 3
        assert "comprehend-it" in result.data
        assert "extract-it" in result.data
        assert "instruct-it" in result.data

    def test_list_model_types_empty(self, sdk, mock_response):
        """Test listing model types when none exist."""
        expected_response = {
            "data": [],
            "total": 0,
        }
        
        with patch.object(sdk.models, "send", return_value=mock_response(200, expected_response)):
            result = sdk.models.list_model_types()
        
        assert isinstance(result, ModelTypesListResponse)
        assert result.total == 0
        assert len(result.data) == 0


class TestGetModel:
    """Tests for get_model endpoint."""

    def test_get_model_success(self, sdk, mock_response):
        """Test successful model retrieval by ID."""
        expected_response = {
            "id": "model-123",
            "name": "Test Model",
            "type": "comprehend-it",
            "description": "A test model",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "status": "active",
        }
        
        with patch.object(sdk.models, "send", return_value=mock_response(200, expected_response)):
            result = sdk.models.get_model("model-123")
        
        assert isinstance(result, ModelObject)
        assert result.id == "model-123"
        assert result.name == "Test Model"
        assert result.type == "comprehend-it"

    def test_get_model_not_found(self, sdk, mock_response):
        """Test retrieving non-existent model."""
        error_response = {"detail": "Model not found"}
        
        with patch.object(sdk.models, "send", return_value=mock_response(404, error_response)):
            with pytest.raises(NotFoundError):
                sdk.models.get_model("non-existent-id")

    def test_get_model_invalid_id(self, sdk, mock_response):
        """Test retrieving model with invalid ID format."""
        error_response = {"detail": "Invalid model ID format"}
        
        with patch.object(sdk.models, "send", return_value=mock_response(400, error_response)):
            with pytest.raises(BadRequestError):
                sdk.models.get_model("invalid@id")


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
        
        with patch.object(sdk.models, "send", return_value=response):
            with pytest.raises(InternalServerError) as exc_info:
                sdk.models.list_models()
            
            assert "not valid JSON" in str(exc_info.value)

    def test_malformed_model_data(self, sdk, mock_response):
        """Test handling of response with missing required fields."""
        malformed_response = {
            "id": "model-123",
            "name": "Test Model",
            # Missing required 'type', 'created_at', 'updated_at', 'status'
        }
        
        with patch.object(sdk.models, "send", return_value=mock_response(200, malformed_response)):
            with pytest.raises(InternalServerError) as exc_info:
                sdk.models.get_model("model-123")
            
            assert "Failed to parse" in str(exc_info.value)

    def test_server_error(self, sdk, mock_response):
        """Test handling of 500 server error."""
        error_response = {"detail": "Internal server error"}
        
        with patch.object(sdk.models, "send", return_value=mock_response(500, error_response)):
            with pytest.raises(InternalServerError):
                sdk.models.list_models()


class TestAsyncMethods:
    """Tests for async API methods."""

    @pytest.mark.asyncio
    async def test_create_model_async(self, sdk, mock_response):
        """Test async model creation."""
        expected_response = {
            "id": "model-123",
            "name": "Test Model",
            "type": "comprehend-it",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "status": "active",
        }
        
        with patch.object(sdk.models, "send_async", return_value=mock_response(200, expected_response)):
            result = await sdk.models.create_model_async(
                name="Test Model",
                type="comprehend-it"
            )
        
        assert isinstance(result, ModelObject)
        assert result.id == "model-123"

    @pytest.mark.asyncio
    async def test_list_models_async(self, sdk, mock_response):
        """Test async models list retrieval."""
        expected_response = {
            "data": [
                {
                    "id": "model-1",
                    "name": "Model 1",
                    "type": "comprehend-it",
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z",
                    "status": "active",
                }
            ],
            "total": 1,
        }
        
        with patch.object(sdk.models, "send_async", return_value=mock_response(200, expected_response)):
            result = await sdk.models.list_models_async()
        
        assert isinstance(result, ModelsListResponse)
        assert result.total == 1

    @pytest.mark.asyncio
    async def test_list_model_types_async(self, sdk, mock_response):
        """Test async model types list retrieval."""
        expected_response = {
            "data": ["comprehend-it", "extract-it"],
            "total": 2,
        }
        
        with patch.object(sdk.models, "send_async", return_value=mock_response(200, expected_response)):
            result = await sdk.models.list_model_types_async()
        
        assert isinstance(result, ModelTypesListResponse)
        assert result.total == 2

    @pytest.mark.asyncio
    async def test_get_model_async(self, sdk, mock_response):
        """Test async model retrieval by ID."""
        expected_response = {
            "id": "model-123",
            "name": "Test Model",
            "type": "comprehend-it",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "status": "active",
        }
        
        with patch.object(sdk.models, "send_async", return_value=mock_response(200, expected_response)):
            result = await sdk.models.get_model_async("model-123")
        
        assert isinstance(result, ModelObject)
        assert result.id == "model-123"
