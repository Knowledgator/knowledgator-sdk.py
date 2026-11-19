"""
Mock API Request Tests for Raw Data Endpoints

This module demonstrates the Datasets API (raw data part) with mock HTTP requests,
verifying the complete API contract including:
- CRUD operations (list_data, create_data, get_data, update_data, delete_data)
- File uploads for data
- Query parameters and filtering
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from knowledgator.configs import SDKConfigs, ServersConfigs
from knowledgator.apis.datasets import DatasetsAPI


class TestDataAPIMockRequests:
    """Test Datasets API (raw data methods) with mock HTTP requests."""

    @pytest.fixture
    def api(self):
        """Create DatasetsAPI instance with mock configuration."""
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.platform.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return DatasetsAPI(configs)

    def test_list_data_minimal(self, api):
        """Test listing raw data with default parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "data-123",
                    "name": "my-raw-data",
                    "type": "ner",
                    "format": "jsonl",
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
        result = api.list_data()

        # Verify request
        assert api.configs.client.request.called
        assert (
            str(api.configs.client.request.call_args.kwargs["url"])
            == "https://api.platform.knowledgator.com/data?offset=0&limit=100"
        )

        # Verify response
        assert len(result.data) == 1
        assert result.data[0].id == "data-123"
        assert result.data[0].name == "my-raw-data"
        assert result.total == 1

    def test_list_data_server_payload_with_short_description_and_files(self, api):
        """Test parsing actual API server response for GET /data/."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total": 1,
            "data": [
                {
                    "id": "data-server-1",
                    "name": "server-data-item",
                    "type": "comprehend-it",
                    "format": "json",
                    "short_description": "Server short description text",
                    "created_by_oauth_id": "oauth-user-999",
                    "created_at": "2026-09-01T00:00:00Z",
                    "updated_at": "2026-09-01T00:00:00Z",
                    "files": [
                        {
                            "id": "file-abc",
                            "created_by_oauth_id": "oauth-user-999",
                            "created_at": "2026-09-01T00:00:00Z",
                            "updated_at": "2026-09-01T00:00:00Z",
                            "size": 2048,
                        }
                    ],
                }
            ],
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.list_data()

        assert len(result.data) == 1
        item = result.data[0]
        assert item.id == "data-server-1"
        assert item.description == "Server short description text"
        assert item.created_by_oauth_id == "oauth-user-999"
        assert item.files is not None and len(item.files) == 1
        assert item.files[0].id == "file-abc"
        assert item.files[0].size == 2048
        assert result.offset == 0
        assert result.limit == 100

    def test_create_data(self, api):
        """Test creating raw data."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        # ObjectCreation schema only guarantees 'id'
        mock_response.json.return_value = {
            "id": "data-456",
            "name": "new-data",
            "type": "ner",
            "format": "jsonl",
            "status": "created",
            "created_at": "2024-01-01T00:00:00Z",
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request
        result = api.create_data(name="new-data", type="ner", format="jsonl", short_description="Test data")

        # Verify request (multipart form-data)
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert str(call_kwargs["url"]) == "https://api.platform.knowledgator.com/data"
        assert "data" in call_kwargs
        assert call_kwargs["data"]["name"] == "new-data"
        assert call_kwargs["data"]["type"] == "ner"

        # Verify response
        assert result.id == "data-456"

    def test_get_data(self, api):
        """Test getting raw data details by ID."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "data-123",
            "name": "my-data",
            "type": "ner",
            "format": "jsonl",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.get_data("data-123")

        assert api.configs.client.request.called
        assert "data/data-123" in str(api.configs.client.request.call_args.kwargs["url"])
        assert result.id == "data-123"

    def test_update_data(self, api):
        """Test updating raw data."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        api.update_data("data-123", name="updated-name")

        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "data/data-123" in str(call_kwargs["url"])
        assert call_kwargs["json"]["name"] == "updated-name"

    def test_delete_data(self, api):
        """Test deleting raw data."""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        api.delete_data("data-123")

        assert api.configs.client.request.called
        assert "data/data-123" in str(api.configs.client.request.call_args.kwargs["url"])

    def test_upload_data(self, api):
        """Test uploading file to raw data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "data-123",
            "name": "data-123",
            "type": "ner",
            "format": "jsonl",
            "status": "uploaded",
            "created_at": "2024-01-01T00:00:00Z",
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        file_content = b"test content"
        result = api.upload_data("data-123", file=file_content)

        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "data/data-123/files/upload" in str(call_kwargs["url"])
        assert "files" in call_kwargs
        assert "file" in call_kwargs["files"]

        assert result.id == "data-123"


class TestDataAPIAsyncMockRequests:
    """Test async version of Datasets API (raw data methods)."""

    @pytest.fixture
    def api(self):
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.platform.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return DatasetsAPI(configs)

    @pytest.mark.asyncio
    async def test_async_list_data(self, api):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "data-async-1",
                    "name": "async",
                    "type": "ner",
                    "format": "jsonl",
                    "created_at": "",
                    "updated_at": "",
                }
            ],
            "total": 1,
            "offset": 0,
            "limit": 100,
        }
        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await api.list_data_async()
        assert len(result.data) == 1
        assert result.data[0].id == "data-async-1"

    @pytest.mark.asyncio
    async def test_async_create_data(self, api):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "data-async-2",
            "name": "async-new",
            "type": "ner",
            "format": "jsonl",
            "status": "created",
            "created_at": "2024-01-01T00:00:00Z",
        }
        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await api.create_data_async(name="async-new", type="ner", format="jsonl")
        assert result.id == "data-async-2"
