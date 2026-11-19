import os
from unittest.mock import MagicMock

import pytest

from knowledgator.sdk import Knowledgator, KnowledgatorRapidAPI
from knowledgator.apis import (
    ExtractEndpointsConfig,
    InstructEndpointsConfig,
    ComprehendEndpointsConfig,
)
from knowledgator.configs import SDKConfigs, ServersConfigs, RetryPolicyConfigs


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
    assert kg.configs.server_config.servers["main"] == "api.platform.knowledgator.com"
    assert kg.configs.client is not None
    assert kg.configs.async_client is not None
    assert kg.configs.client_supplied is False
    assert kg.configs.async_client_supplied is False
    assert kg.configs.retry_config.max_tries == 3
    assert kg.comprehend.configs == kg.configs
    assert kg.extract.configs == kg.configs
    assert kg.instruct.configs == kg.configs


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
        extract_endpoints=ExtractEndpointsConfig(main="custom-extract"),
        instruct_endpoints=InstructEndpointsConfig(main="custom-instruct"),
    )

    assert kg.configs.server_config.servers["main"] == "https://custom.example.com"
    assert kg.configs.client == mock_client
    assert kg.configs.async_client == mock_async_client
    assert kg.configs.client_supplied is True
    assert kg.configs.async_client_supplied is True
    assert kg.configs.retry_config.max_tries == 5
    assert kg.comprehend.ENDPOINTS_CONFIG.main == "custom-comprehend"
    assert kg.extract.ENDPOINTS_CONFIG.main == "custom-extract"
    assert kg.instruct.ENDPOINTS_CONFIG.main == "custom-instruct"


def test_knowledgator_rapidapi_init_defaults(mock_clients):
    """Test KnowledgatorRapidAPI initializes correctly"""
    kg = KnowledgatorRapidAPI(api_key="rapid_key")

    # Verify server configurations
    servers = kg.configs.server_config.servers
    headers = kg.configs.server_config.base_headers
    assert servers["comprehend-it-rapidapi"] == "https://comprehend-it.p.rapidapi.com"
    assert servers["extract-it-rapidapi"] == "https://extract-it.p.rapidapi.com"
    assert servers["instruct-it-rapidapi"] == "https://instruct-it.p.rapidapi.com"
    assert headers["comprehend-it-rapidapi"] == {
        "x-rapidapi-key": "rapid_key",
        "x-rapidapi-host": "comprehend-it.p.rapidapi.com",
        "Content-Type": "application/json",
    }
    # Verify API-specific configs
    assert kg.comprehend.configs.server == "comprehend-it-rapidapi"
    assert kg.extract.configs.server == "extract-it-rapidapi"
    assert kg.instruct.configs.server == "instruct-it-rapidapi"


def test_knowledgator_rapidapi_init_custom_hosts(mock_clients):
    """Test KnowledgatorRapidAPI with custom hosts"""
    kg = KnowledgatorRapidAPI(
        api_key="custom_rapid_key",
        comprehend_host="custom-comprehend.host",
        extract_host="custom-extract.host",
        instruct_host="custom-instruct.host",
    )

    servers = kg.configs.server_config.servers
    headers = kg.configs.server_config.base_headers
    assert servers["comprehend-it-rapidapi"] == "https://custom-comprehend.host"
    assert servers["extract-it-rapidapi"] == "https://custom-extract.host"
    assert servers["instruct-it-rapidapi"] == "https://custom-instruct.host"
    assert headers["comprehend-it-rapidapi"]["x-rapidapi-host"] == "custom-comprehend.host"
    assert headers["extract-it-rapidapi"]["x-rapidapi-host"] == "custom-extract.host"
    assert headers["instruct-it-rapidapi"]["x-rapidapi-host"] == "custom-instruct.host"


def test_knowledgator_with_prebuilt_config(mock_clients):
    """Test using pre-built SDKConfigs instance"""
    custom_config = SDKConfigs(server="custom_server", retry_config=RetryPolicyConfigs(max_tries=10))

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


# Headers usage tests


def test_knowledgator_main_server_headers(mock_clients):
    """Test that main server uses KnowledgatorAPI-Key header"""
    api_key = "test_main_api_key"
    kg = Knowledgator(api_key=api_key)

    # Verify the main server headers contain KnowledgatorAPI-Key
    main_headers = kg.configs.server_config.base_headers.get("main", {})
    assert "KnowledgatorAPI-Key" in main_headers
    assert main_headers["KnowledgatorAPI-Key"] == api_key

    # Verify RapidAPI headers are NOT used for main server
    assert "x-rapidapi-key" not in main_headers
    assert "x-rapidapi-host" not in main_headers


def test_knowledgator_rapidapi_headers(mock_clients):
    """Test that RapidAPI servers use x-rapidapi-key and x-rapidapi-host headers"""
    api_key = "test_rapid_api_key"
    kg = KnowledgatorRapidAPI(api_key=api_key)

    # Test comprehend-it headers
    comprehend_headers = kg.configs.server_config.base_headers.get("comprehend-it-rapidapi", {})
    assert "x-rapidapi-key" in comprehend_headers
    assert comprehend_headers["x-rapidapi-key"] == api_key
    assert "x-rapidapi-host" in comprehend_headers
    assert comprehend_headers["x-rapidapi-host"] == "comprehend-it.p.rapidapi.com"
    assert "Content-Type" in comprehend_headers

    # Verify main server header is NOT used
    assert "KnowledgatorAPI-Key" not in comprehend_headers

    # Test extract-it headers
    extract_headers = kg.configs.server_config.base_headers.get("extract-it-rapidapi", {})
    assert "x-rapidapi-key" in extract_headers
    assert extract_headers["x-rapidapi-key"] == api_key
    assert "x-rapidapi-host" in extract_headers
    assert extract_headers["x-rapidapi-host"] == "extract-it.p.rapidapi.com"

    # Test instruct-it headers
    instruct_headers = kg.configs.server_config.base_headers.get("instruct-it-rapidapi", {})
    assert "x-rapidapi-key" in instruct_headers
    assert instruct_headers["x-rapidapi-key"] == api_key
    assert "x-rapidapi-host" in instruct_headers
    assert instruct_headers["x-rapidapi-host"] == "instruct-it.p.rapidapi.com"


def test_servers_config_headers_method(mock_clients):
    """Test that ServersConfigs.headers() method returns correct headers for each server"""
    # Test main server
    kg_main = Knowledgator(api_key="main_key")
    server_config = kg_main.configs.server_config

    # Test headers() method returns correct headers for main server
    main_headers = server_config.headers("main")
    assert "KnowledgatorAPI-Key" in main_headers
    assert main_headers["KnowledgatorAPI-Key"] == "main_key"

    # Test headers() method with None (should return base_server headers)
    default_headers = server_config.headers(None)
    assert default_headers == main_headers

    # Test RapidAPI servers
    kg_rapid = KnowledgatorRapidAPI(api_key="rapid_key")
    rapid_config = kg_rapid.configs.server_config

    # Test headers() method returns correct headers for each RapidAPI server
    comprehend_headers = rapid_config.headers("comprehend-it-rapidapi")
    assert "x-rapidapi-key" in comprehend_headers
    assert comprehend_headers["x-rapidapi-key"] == "rapid_key"
    assert comprehend_headers["x-rapidapi-host"] == "comprehend-it.p.rapidapi.com"

    extract_headers = rapid_config.headers("extract-it-rapidapi")
    assert "x-rapidapi-key" in extract_headers
    assert extract_headers["x-rapidapi-host"] == "extract-it.p.rapidapi.com"

    instruct_headers = rapid_config.headers("instruct-it-rapidapi")
    assert "x-rapidapi-key" in instruct_headers
    assert instruct_headers["x-rapidapi-host"] == "instruct-it.p.rapidapi.com"


def test_api_specific_configs_use_correct_headers(mock_clients):
    """Test that each API (comprehend, extract, instruct) uses correct server-specific headers"""
    kg = KnowledgatorRapidAPI(api_key="rapid_key")

    # Verify each API has the correct server configured
    assert kg.comprehend.configs.server == "comprehend-it-rapidapi"
    assert kg.extract.configs.server == "extract-it-rapidapi"
    assert kg.instruct.configs.server == "instruct-it-rapidapi"

    # Verify each API gets the correct headers via get_server_details()
    comprehend_url, comprehend_headers = kg.comprehend.configs.get_server_details()
    assert "x-rapidapi-key" in comprehend_headers
    assert comprehend_headers["x-rapidapi-host"] == "comprehend-it.p.rapidapi.com"

    extract_url, extract_headers = kg.extract.configs.get_server_details()
    assert "x-rapidapi-key" in extract_headers
    assert extract_headers["x-rapidapi-host"] == "extract-it.p.rapidapi.com"

    instruct_url, instruct_headers = kg.instruct.configs.get_server_details()
    assert "x-rapidapi-key" in instruct_headers
    assert instruct_headers["x-rapidapi-host"] == "instruct-it.p.rapidapi.com"


def test_custom_server_headers(mock_clients):
    """Test that custom server headers can be added and override defaults"""
    custom_headers = {"Custom-Header": "custom_value"}
    kg = Knowledgator(api_key="test_key", server_base_headers=custom_headers)

    main_headers = kg.configs.server_config.base_headers.get("main", {})
    assert "KnowledgatorAPI-Key" in main_headers
    assert "Custom-Header" in main_headers
    assert main_headers["Custom-Header"] == "custom_value"


def test_manual_servers_config_without_headers(mock_clients):
    """Test that manually creating ServersConfigs without base_headers has no auth headers"""
    from knowledgator.apis.extract import ExtractAPI

    # This is what happens in the example - no headers are set!
    configs = SDKConfigs()
    configs.server_config = ServersConfigs(servers={"main": "https://api.platform.knowledgator.com"})
    api = ExtractAPI(configs)

    # Verify that there are NO authentication headers
    _, headers = api.configs.get_server_details()
    assert headers == {}  # Empty! No authentication!
    assert "KnowledgatorAPI-Key" not in headers
    assert "x-rapidapi-key" not in headers


def test_manual_servers_config_with_headers(mock_clients):
    """Test that manually creating ServersConfigs WITH base_headers works correctly"""
    from knowledgator.apis.extract import ExtractAPI

    # Correct way to manually set up with headers
    configs = SDKConfigs()
    configs.server_config = ServersConfigs(
        servers={"main": "https://api.platform.knowledgator.com"},
        base_headers={"main": {"KnowledgatorAPI-Key": "manual-api-key"}},
    )
    api = ExtractAPI(configs)

    # Verify that authentication headers are present
    _, headers = api.configs.get_server_details()
    assert "KnowledgatorAPI-Key" in headers
    assert headers["KnowledgatorAPI-Key"] == "manual-api-key"


def test_documentation_example_headers(mock_clients):
    """Test that the exact documentation example uses correct headers"""
    # This is the exact example from docs/api/extract-it/index.md
    from knowledgator import Knowledgator

    kg = Knowledgator(api_key="YOUR_API_KEY")

    # Verify that kg.extract uses the correct headers
    _, headers = kg.extract.configs.get_server_details()

    # Should use KnowledgatorAPI-Key for main server
    assert "KnowledgatorAPI-Key" in headers
    assert headers["KnowledgatorAPI-Key"] == "YOUR_API_KEY"

    # Should NOT use RapidAPI headers
    assert "x-rapidapi-key" not in headers
    assert "x-rapidapi-host" not in headers

    # Verify the extract API is configured for main server
    assert kg.extract.configs.server == "main"
