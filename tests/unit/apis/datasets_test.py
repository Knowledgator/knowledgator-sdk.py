"""
Mock API Request Tests for Datasets Endpoint

This module demonstrates the Datasets API with mock HTTP requests,
verifying the complete API contract including:
- CRUD operations (list, create, get, update, delete)
- File uploads
- Query parameters and filtering
- Error scenarios
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from knowledgator.configs import SDKConfigs, ServersConfigs
from knowledgator.exceptions import BadRequest, ServerError
from knowledgator.apis.datasets import DatasetsAPI
from knowledgator.datasets.common import DatasetStatus
from knowledgator.exceptions.base import ClientError


class TestDatasetsAPIMockRequests:
    """Test Datasets API with mock HTTP requests matching the API contract."""

    @pytest.fixture
    def api(self):
        """Create DatasetsAPI instance with mock configuration."""
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return DatasetsAPI(configs)

    def test_list_datasets_minimal(self, api):
        """Test listing datasets with default parameters."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "datasets": [
                {
                    "id": "ds-123",
                    "name": "my-ner-dataset",
                    "type": "ner",
                    "format": "jsonl",
                    "status": "ready",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            ],
            "total": 1,
            "offset": 0,
            "limit": 100,
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request
        result = api.list()

        # Verify request
        assert api.configs.client.request.called

        # Verify response
        assert len(result.datasets) == 1
        assert result.datasets[0].id == "ds-123"
        assert result.datasets[0].name == "my-ner-dataset"
        assert result.total == 1

    def test_list_datasets_with_filters(self, api):
        """Test listing datasets with all filter parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "datasets": [],
            "total": 0,
            "offset": 10,
            "limit": 50,
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request with filters
        result = api.list(
            offset=10,
            limit=50,
            name="test-dataset",
            type="text-classification",
            format="csv",
            status=DatasetStatus.READY,
        )

        # Verify request was made
        assert api.configs.client.request.called

        # Verify response
        assert result.total == 0
        assert result.offset == 10
        assert result.limit == 50

    def test_create_dataset_without_file(self, api):
        """Test creating a dataset without initial file upload."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "ds-456",
            "name": "new-dataset",
            "type": "ner",
            "format": "jsonl",
            "status": "pending",
            "created_at": "2024-01-02T00:00:00Z",
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request
        result = api.create(name="new-dataset", type="ner", format="jsonl", description="Test dataset")

        # Verify request (multipart form-data)
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "data" in call_kwargs
        assert call_kwargs["data"]["name"] == "new-dataset"
        assert call_kwargs["data"]["type"] == "ner"
        assert call_kwargs["data"]["format"] == "jsonl"
        assert call_kwargs["data"]["description"] == "Test dataset"
        assert "files" not in call_kwargs  # No file uploaded

        # Verify response
        assert result.id == "ds-456"
        assert result.name == "new-dataset"
        assert result.status == DatasetStatus.PENDING

    def test_create_dataset_with_file(self, api):
        """Test creating a dataset with initial file upload."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "ds-789",
            "name": "dataset-with-file",
            "type": "text-classification",
            "format": "csv",
            "status": "processing",
            "created_at": "2024-01-03T00:00:00Z",
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request with file
        file_content = b"text,label\nHello,positive\nBad,negative"
        result = api.create(name="dataset-with-file", type="text-classification", format="csv", file=file_content)

        # Verify request includes file
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "data" in call_kwargs
        assert "files" in call_kwargs
        assert "file" in call_kwargs["files"]

        # Verify response
        assert result.id == "ds-789"
        assert result.status == DatasetStatus.PROCESSING

    def test_get_dataset(self, api):
        """Test getting dataset details by ID."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "ds-123",
            "name": "my-dataset",
            "type": "ner",
            "format": "jsonl",
            "status": "ready",
            "description": "My NER dataset",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "file_count": 3,
            "total_size": 1024000,
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request
        result = api.get("ds-123")

        # Verify request was made
        assert api.configs.client.request.called

        # Verify response
        assert result.id == "ds-123"
        assert result.name == "my-dataset"
        assert result.file_count == 3
        assert result.total_size == 1024000

    def test_update_dataset_name(self, api):
        """Test updating dataset name."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request
        api.update("ds-123", name="updated-name")

        # Verify request
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "json" in call_kwargs
        assert call_kwargs["json"]["name"] == "updated-name"
        assert "description" not in call_kwargs["json"]

    def test_update_dataset_description(self, api):
        """Test updating dataset description."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request
        api.update("ds-123", description="New description")

        # Verify request
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert call_kwargs["json"]["description"] == "New description"
        assert "name" not in call_kwargs["json"]

    def test_update_dataset_both_fields(self, api):
        """Test updating both name and description."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request
        api.update("ds-123", name="new-name", description="New description")

        # Verify request
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert call_kwargs["json"]["name"] == "new-name"
        assert call_kwargs["json"]["description"] == "New description"

    def test_update_dataset_no_fields_error(self, api):
        """Test that updating without any fields raises BadRequest."""
        with pytest.raises(BadRequest) as exc_info:
            api.update("ds-123")

        assert "At least one field" in str(exc_info.value)

    def test_delete_dataset(self, api):
        """Test deleting a dataset."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request
        api.delete("ds-123")

        # Verify request was made
        assert api.configs.client.request.called

    def test_upload_file_to_dataset(self, api):
        """Test uploading a file to an existing dataset."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "ds-123",
            "name": "my-dataset",
            "type": "ner",
            "format": "jsonl",
            "status": "processing",
            "created_at": "2024-01-01T00:00:00Z",
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request
        file_content = b'{"text": "Sample", "entities": []}\n'
        result = api.upload("ds-123", file=file_content)

        # Verify request
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "files" in call_kwargs
        assert "file" in call_kwargs["files"]

        # Verify response
        assert result.id == "ds-123"
        assert result.status == DatasetStatus.PROCESSING

    def test_list_formats(self, api):
        """Test listing available dataset formats."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "formats": [
                {"name": "jsonl", "description": "JSON Lines format"},
                {"name": "csv", "description": "Comma-separated values"},
                {"name": "parquet", "description": "Apache Parquet format"},
            ],
            "total": 3,
            "offset": 0,
            "limit": 100,
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request
        result = api.list_formats()

        # Verify request was made
        assert api.configs.client.request.called

        # Verify response
        assert len(result.formats) == 3
        assert result.formats[0].name == "jsonl"
        assert result.total == 3

    def test_server_error_500(self, api):
        """Test handling of 500 Internal Server Error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.return_value = {"error": "Server error occurred"}
        mock_response.headers = {"x-request-id": "req-abc-123"}
        api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError):
            api.list()

    def test_dataset_not_found_400(self, api):
        """Test handling of dataset not found error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Dataset not found"
        mock_response.json.return_value = {"error": "Dataset doesn't exist"}
        mock_response.headers = {}
        mock_response.reason_phrase = "Not Found"
        api.configs.client.request.return_value = mock_response

        with pytest.raises(ClientError):
            api.get("nonexistent-id")


class TestDatasetsAPIAsyncMockRequests:
    """Test async version of Datasets API with mock requests."""

    @pytest.fixture
    def api(self):
        """Create DatasetsAPI instance with mock configuration."""
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return DatasetsAPI(configs)

    @pytest.mark.asyncio
    async def test_async_list_datasets(self, api):
        """Test async list datasets."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "datasets": [
                {
                    "id": "ds-async-1",
                    "name": "async-dataset",
                    "type": "ner",
                    "format": "jsonl",
                    "status": "ready",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                }
            ],
            "total": 1,
            "offset": 0,
            "limit": 100,
        }
        mock_response.headers = {}

        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await api.list_async()

        assert len(result.datasets) == 1
        assert result.datasets[0].id == "ds-async-1"

    @pytest.mark.asyncio
    async def test_async_create_dataset(self, api):
        """Test async create dataset."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "ds-async-2",
            "name": "async-new",
            "type": "text-classification",
            "format": "csv",
            "status": "pending",
            "created_at": "2024-01-02T00:00:00Z",
        }
        mock_response.headers = {}

        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await api.create_async(name="async-new", type="text-classification", format="csv")

        assert result.id == "ds-async-2"
        assert result.name == "async-new"

    @pytest.mark.asyncio
    async def test_async_get_dataset(self, api):
        """Test async get dataset."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "ds-async-3",
            "name": "async-dataset",
            "type": "ner",
            "format": "jsonl",
            "status": "ready",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }
        mock_response.headers = {}

        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await api.get_async("ds-async-3")

        assert result.id == "ds-async-3"
        assert result.name == "async-dataset"

    @pytest.mark.asyncio
    async def test_async_update_dataset(self, api):
        """Test async update dataset."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.headers = {}

        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        await api.update_async("ds-async-4", name="updated-async")

        # Verify the async client was called
        assert api.configs.async_client.request.called

    @pytest.mark.asyncio
    async def test_async_delete_dataset(self, api):
        """Test async delete dataset."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.headers = {}

        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        await api.delete_async("ds-async-5")

        assert api.configs.async_client.request.called

    @pytest.mark.asyncio
    async def test_async_upload_file(self, api):
        """Test async upload file."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "ds-async-6",
            "name": "async-dataset",
            "type": "ner",
            "format": "jsonl",
            "status": "processing",
            "created_at": "2024-01-01T00:00:00Z",
        }
        mock_response.headers = {}

        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        file_content = b'{"text": "test"}\n'
        result = await api.upload_async("ds-async-6", file=file_content)

        assert result.id == "ds-async-6"
        assert result.status == DatasetStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_async_list_formats(self, api):
        """Test async list formats."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "formats": [{"name": "jsonl", "description": "JSON Lines"}],
            "total": 1,
            "offset": 0,
            "limit": 100,
        }
        mock_response.headers = {}

        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await api.list_formats_async()

        assert len(result.formats) == 1
        assert result.formats[0].name == "jsonl"
