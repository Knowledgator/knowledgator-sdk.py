import os
import pytest
from unittest.mock import MagicMock, patch

from knowledgator.sdk import Knowledgator, KnowledgatorRapidAPI
from knowledgator.configs import SDKConfigs, ServersConfigs, RetryPolicyConfigs
from knowledgator.apis import EndpointsConfig, ComprehendEndpointsConfig, ExtractEndpointsConfig

class MockHttpClient:
    def __init__(self, *args, **kwargs):
        pass

class MockAsyncHttpClient:
    def __init__(self, *args, **kwargs):
        pass

@pytest.fixture
def mock_clients(monkeypatch):
    monkeypatch.setattr("knowledgator.clients.http_client.HttpClient", MockHttpClient)
    monkeypatch.setattr("knowledgator.clients.http_client.AsyncHttpClient", MockAsyncHttpClient)

def test_knowledgator_init_defaults(mock_clients):
    """Test Knowledgator initializes with default configs"""
    # Set environment variable for API key
    os.environ["KG_API_KEY"] = "env_api_key"
    
    kg = Knowledgator()
    
    assert kg.configs.server == "main"
    assert kg.configs.server_config.servers["main"] == "api.knowledgator.com"
    assert kg.configs.client is not None
    assert kg.configs.async_client is not None
    assert kg.configs.client_supplied is False
    assert kg.configs.async_client_supplied is False
    assert kg.configs.retry_config.max_tries == 3
    assert kg.comprehend.configs == kg.configs
    assert kg.extract.configs == kg.configs

def test_knowledgator_init_custom_params(mock_clients):
    """Test Knowledgator initializes with custom parameters"""
    custom_retry = RetryPolicyConfigs(max_tries=5)
    mock_client = MockHttpClient()
    mock_async_client = MockAsyncHttpClient()
    
    kg = Knowledgator(
        api_key="custom_key",
        client=mock_client,
        async_client=mock_async_client,
        server_url="https://custom.example.com",
        max_tries=7,
        retry_config=custom_retry,
        timeout_ms=15000,
        comprehend_endpoints=ComprehendEndpointsConfig(main="custom-comprehend"),
        extract_endpoints=ExtractEndpointsConfig(main="custom-extract")
    )
    
    assert kg.configs.server_config.servers["main"] == "https://custom.example.com"
    assert kg.configs.client == mock_client
    assert kg.configs.async_client == mock_async_client
    assert kg.configs.client_supplied is True
    assert kg.configs.async_client_supplied is True
    assert kg.configs.retry_config.max_tries == 5
    assert kg.comprehend.ENDPOINTS_CONFIG.main == "custom-comprehend"
    assert kg.extract.ENDPOINTS_CONFIG.main == "custom-extract"

def test_knowledgator_rapidapi_init_defaults(mock_clients):
    """Test KnowledgatorRapidAPI initializes correctly"""
    kg = KnowledgatorRapidAPI(api_key="rapid_key")
    
    # Verify server configurations
    servers = kg.configs.server_config.servers
    headers = kg.configs.server_config.base_headers
    assert servers["comprehend-it-rapidapi"] == "https://comprehend-it.p.rapidapi.com"
    assert servers["extract-it-rapidapi"] == "https://extract-it.p.rapidapi.com"
    assert headers["comprehend-it-rapidapi"] == {
        "x-rapidapi-key": "rapid_key",
        "x-rapidapi-host": "comprehend-it.p.rapidapi.com",
        "Content-Type": "application/json",
    }
    # Verify API-specific configs
    assert kg.comprehend.configs.server == "comprehend-it-rapidapi"
    assert kg.extract.configs.server == "extract-it-rapidapi"

def test_knowledgator_rapidapi_init_custom_hosts(mock_clients):
    """Test KnowledgatorRapidAPI with custom hosts"""
    kg = KnowledgatorRapidAPI(
        api_key="custom_rapid_key",
        comprehend_host="custom-comprehend.host",
        extract_host="custom-extract.host"
    )
    
    servers = kg.configs.server_config.servers
    headers = kg.configs.server_config.base_headers
    assert servers["comprehend-it-rapidapi"] == "https://custom-comprehend.host"
    assert servers["extract-it-rapidapi"] == "https://custom-extract.host"
    assert headers["comprehend-it-rapidapi"]["x-rapidapi-host"] == "custom-comprehend.host"
    assert headers["extract-it-rapidapi"]["x-rapidapi-host"] == "custom-extract.host"

def test_knowledgator_with_prebuilt_config(mock_clients):
    """Test using pre-built SDKConfigs instance"""
    custom_config = SDKConfigs(
        server="custom_server",
        retry_config=RetryPolicyConfigs(max_tries=10)
    )
    
    kg = Knowledgator(configs=custom_config, api_key="key")
    
    assert kg.configs == custom_config
    assert kg.configs.retry_config.max_tries == 10
    assert kg.configs.server == "custom_server"

def test_knowledgator_headers_without_key(mock_clients):
    """Test header generation without API key"""
    kg = Knowledgator(api_key=None)
    base_headers = kg.configs.server_config.base_headers
    assert "KnowledgatorAPI-Key" not in base_headers

def test_knowledgator_context_management(mock_clients):
    """Test context manager properly closes clients"""
    with Knowledgator(api_key="key") as kg:
        kg.configs.client = MagicMock()
        kg.configs.async_client = MagicMock()
    
    kg.configs.client.close.assert_called_once()
    kg.configs.async_client.aclose.assert_not_called()  # Sync context doesn't close async

@pytest.mark.asyncio
async def test_knowledgator_async_context(mock_clients):
    """Test async context manager"""
    async with Knowledgator(api_key="key") as kg:
        kg.configs.async_client = MagicMock()
    
    kg.configs.async_client.aclose.assert_called_once()