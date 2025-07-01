import asyncio
import httpx
import pytest

from knowledgator.clients.http_client import HttpClient, AsyncHttpClient   # ← change if needed
from knowledgator.configs import SDKConfigs, ServersConfigs, RetryPolicyConfigs
from knowledgator.apis import (BaseAPI, ComprehendAPI, ComprehendEndpointsConfig,
                               ExtractAPI, ExtractEndpointsConfig)
from knowledgator.utils.logger import Logger

def test_base_api_init_with_defaults(monkeypatch):
    """Test BaseAPI initializes with default clients when none provided"""
    mock_config = SDKConfigs()
    mock_config.server_config = ServersConfigs(servers={"main": "https://api.example.com"})
    
    # Mock client creation to avoid real network calls
    monkeypatch.setattr(HttpClient, "__init__", lambda *_, **__: None)
    monkeypatch.setattr(AsyncHttpClient, "__init__", lambda *_, **__: None)
    
    api = BaseAPI(configs=mock_config)
    
    assert api.configs == mock_config
    assert isinstance(api.configs.client, HttpClient)
    assert isinstance(api.configs.async_client, AsyncHttpClient)
    assert api.configs.client_supplied is False
    assert api.configs.async_client_supplied is False


def test_base_api_init_with_custom_clients():
    """Test BaseAPI uses pre-configured clients when provided"""
    mock_config = SDKConfigs(
        client=HttpClient(base_url="https://custom.example.com"),
        async_client=AsyncHttpClient(base_url="https://custom.example.com"),
    )
    
    api = BaseAPI(configs=mock_config)
    
    assert api.configs.client == mock_config.client
    assert api.configs.async_client == mock_config.async_client
    assert api.configs.client_supplied is True
    assert api.configs.async_client_supplied is True


def test_base_api_path_override():
    """Test BaseAPI respects path override during initialization"""
    mock_config = SDKConfigs()
    api = BaseAPI(configs=mock_config, path="/custom/path")
    
    assert api.PATH == "/custom/path"


def test_comprehend_api_init():
    """Test ComprehendAPI initializes with correct default configuration"""
    mock_config = SDKConfigs()
    api = ComprehendAPI(configs=mock_config)
    
    assert api.PATH == '/comprehend_it/'
    assert api.METHOD == 'POST'
    assert api.CONTENT_TYPE == 'application/json'
    assert isinstance(api.ENDPOINTS_CONFIG, ComprehendEndpointsConfig)
    assert api.ENDPOINTS_CONFIG.main == "classify"


def test_comprehend_api_endpoint_override():
    """Test ComprehendAPI respects custom endpoints config"""
    mock_config = SDKConfigs()
    custom_endpoints = ComprehendEndpointsConfig(main="custom-classify")
    
    api = ComprehendAPI(
        configs=mock_config,
        endpoints_config=custom_endpoints
    )
    
    assert api.ENDPOINTS_CONFIG == custom_endpoints
    assert api.ENDPOINTS_CONFIG.main == "custom-classify"


def test_comprehend_api_inherits_base_properties():
    """Test ComprehendAPI properly inherits from BaseAPI"""
    mock_config = SDKConfigs(
        client=HttpClient(base_url="https://comprehend.example.com"),
        async_client=AsyncHttpClient(base_url="https://comprehend.example.com")
    )
    
    api = ComprehendAPI(configs=mock_config)
    
    # Verify inherited properties
    assert api.configs == mock_config
    assert api.client() == mock_config.client
    assert api.async_client() == mock_config.async_client


def test_comprehend_api_path_override():
    """Test ComprehendAPI respects path override during initialization"""
    mock_config = SDKConfigs()
    custom_path = "/custom/comprehend/path"
    
    api = ComprehendAPI(
        configs=mock_config,
        path=custom_path
    )
    
    assert api.PATH == custom_path


@pytest.mark.asyncio
async def test_comprehend_api_async_client_initialization():
    """Test async client is properly initialized for ComprehendAPI"""
    mock_config = SDKConfigs(
        async_client=AsyncHttpClient(base_url="https://async-comprehend.example.com")
    )
    
    api = ComprehendAPI(configs=mock_config)
    async_client = api.async_client()
    
    assert async_client == mock_config.async_client
    assert async_client.base_url == "https://async-comprehend.example.com"


def test_comprehend_api_single_endpoint_config():
    """Test ComprehendAPI maintains proper endpoint configuration"""
    mock_config = SDKConfigs()
    api = ComprehendAPI(configs=mock_config)
    
    # Verify endpoint path construction
    assert "{}/{}".format(api.PATH, api.ENDPOINTS_CONFIG.main) == "/comprehend_it/classify"


def test_comprehend_api_with_custom_retry_policy():
    """Test ComprehendAPI initialization with custom retry policy"""
    custom_retry = RetryPolicyConfigs(max_tries=3, base_delay=0.2)
    mock_config = SDKConfigs(retry_config=custom_retry)
    
    api = ComprehendAPI(configs=mock_config)
    
    assert api.configs.retry_config == custom_retry
    assert api.configs.retry_config.max_tries == 3
    assert api.configs.retry_config.base_delay == 0.2


def test_comprehend_api_logging_enabled():
    """Test ComprehendAPI properly handles logging configuration"""
    # Create a mock logger
    mock_logger = Logger()
    
    mock_config = SDKConfigs(debug_logger=mock_logger)
    api = ComprehendAPI(configs=mock_config)
    
    assert api._log == mock_logger

def test_extract_api_init():
    """Test ExtractAPI initializes with correct default configuration"""
    mock_config = SDKConfigs()
    api = ExtractAPI(configs=mock_config)
    
    assert api.PATH == '/extract_it/'
    assert api.METHOD == 'POST'
    assert api.CONTENT_TYPE == 'application/json'
    assert isinstance(api.ENDPOINTS_CONFIG, ExtractEndpointsConfig)
    assert api.ENDPOINTS_CONFIG.main == "general"
    assert api.ENDPOINTS_CONFIG.relations == "relation-extraction"
    assert api.ENDPOINTS_CONFIG.summary == "summarization"


def test_extract_api_endpoint_override():
    """Test ExtractAPI respects custom endpoints config"""
    mock_config = SDKConfigs()
    custom_endpoints = ExtractEndpointsConfig(
        main="custom-main",
        relations="custom-relations",
        summary="custom-summary"
    )
    
    api = ExtractAPI(
        configs=mock_config,
        endpoints_config=custom_endpoints
    )
    
    assert api.ENDPOINTS_CONFIG == custom_endpoints
    assert api.ENDPOINTS_CONFIG.main == "custom-main"
    assert api.ENDPOINTS_CONFIG.relations == "custom-relations"
    assert api.ENDPOINTS_CONFIG.summary == "custom-summary"


def test_extract_api_inherits_base_properties():
    """Test ExtractAPI properly inherits from BaseAPI"""
    mock_config = SDKConfigs(
        client=HttpClient(base_url="https://test.example.com"),
        async_client=AsyncHttpClient(base_url="https://test.example.com")
    )
    
    api = ExtractAPI(configs=mock_config)
    
    # Verify inherited properties
    assert api.configs == mock_config
    assert api.client() == mock_config.client
    assert api.async_client() == mock_config.async_client


def test_extract_api_path_override():
    """Test ExtractAPI respects path override during initialization"""
    mock_config = SDKConfigs()
    custom_path = "/custom/extract/path"
    
    api = ExtractAPI(
        configs=mock_config,
        path=custom_path
    )
    
    assert api.PATH == custom_path


@pytest.mark.asyncio
async def test_extract_api_async_client_initialization():
    """Test async client is properly initialized for ExtractAPI"""
    mock_config = SDKConfigs(
        async_client=AsyncHttpClient(base_url="https://async-extract.example.com")
    )
    
    api = ExtractAPI(configs=mock_config)
    async_client = api.async_client()
    
    assert async_client == mock_config.async_client
    assert async_client.base_url == "https://async-extract.example.com"


def test_extract_api_multiple_endpoint_configs():
    """Test ExtractAPI maintains separate endpoint configs for each operation"""
    mock_config = SDKConfigs()
    api = ExtractAPI(configs=mock_config)
    
    # Verify all endpoint types are properly configured
    assert "{}/{}".format(api.PATH, api.ENDPOINTS_CONFIG.main) == "/extract_it/general"
    assert "{}/{}".format(api.PATH, api.ENDPOINTS_CONFIG.relations) == "/extract_it/relation-extraction"
    assert "{}/{}".format(api.PATH, api.ENDPOINTS_CONFIG.summary) == "/extract_it/summarization"


def test_extract_api_with_custom_retry_policy():
    """Test ExtractAPI initialization with custom retry policy"""
    custom_retry = RetryPolicyConfigs(max_tries=5, base_delay=0.1)
    mock_config = SDKConfigs(retry_config=custom_retry)
    
    api = ExtractAPI(configs=mock_config)
    
    assert api.configs.retry_config == custom_retry
    assert api.configs.retry_config.max_tries == 5
    assert api.configs.retry_config.base_delay == 0.1