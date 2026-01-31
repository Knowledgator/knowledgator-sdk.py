import pytest

from knowledgator.apis import (
    BaseAPI,
    ExtractAPI,
    InstructAPI,
    ComprehendAPI,
    ExtractEndpointsConfig,
    InstructEndpointsConfig,
    ComprehendEndpointsConfig,
)
from knowledgator.configs import SDKConfigs, ServersConfigs, RetryPolicyConfigs
from knowledgator.utils.logger import Logger
from knowledgator.clients.http_client import HttpClient, AsyncHttpClient


@pytest.fixture(autouse=True)
def disable_http(monkeypatch):
    monkeypatch.setattr(HttpClient, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(AsyncHttpClient, "__init__", lambda self, *args, **kwargs: None)


def test_base_api_default_clients():
    configs = SDKConfigs()
    configs.server_config = ServersConfigs(servers={"main": "https://api.test"})
    api = BaseAPI(configs)
    assert api.configs.client
    assert api.configs.async_client
    assert not getattr(api.configs, "client_supplied", False)
    assert not getattr(api.configs, "async_client_supplied", False)


def test_base_api_custom_clients():
    http = HttpClient(base_url="https://custom.test")
    async_http = AsyncHttpClient(base_url="https://custom.test")
    configs = SDKConfigs(client=http, async_client=async_http, client_supplied=True, async_client_supplied=True)
    api = BaseAPI(configs)
    assert api.configs.client is http
    assert api.configs.async_client is async_http
    assert api.configs.client_supplied
    assert api.configs.async_client_supplied


def test_base_api_path_override():
    configs = SDKConfigs()
    api = BaseAPI(configs, path="/override")
    assert api.PATH == "/override"


def test_comprehend_defaults():
    configs = SDKConfigs()
    api = ComprehendAPI(configs)
    assert api.PATH == "/comprehend/"
    assert api.METHOD == "POST"
    assert api.CONTENT_TYPE == "application/json"
    assert isinstance(api.ENDPOINTS_CONFIG, ComprehendEndpointsConfig)
    assert api.ENDPOINTS_CONFIG.main == "classify"


def test_comprehend_endpoint_override():
    configs = SDKConfigs()
    custom = ComprehendEndpointsConfig(main="custom-classify")
    api = ComprehendAPI(configs, endpoints_config=custom)
    assert api.ENDPOINTS_CONFIG is custom
    assert api.ENDPOINTS_CONFIG.main == "custom-classify"


def test_comprehend_inherits_clients():
    http = HttpClient(base_url="x")
    async_http = AsyncHttpClient(base_url="x")
    configs = SDKConfigs(client=http, async_client=async_http)
    api = ComprehendAPI(configs)
    assert api.client() is http
    assert api.async_client() is async_http


def test_comprehend_path_override():
    configs = SDKConfigs()
    api = ComprehendAPI(configs, path="/cpath")
    assert api.PATH == "/cpath"


@pytest.mark.asyncio
async def test_comprehend_async_client():
    async_http = AsyncHttpClient(base_url="async-c")
    configs = SDKConfigs(async_client=async_http)
    api = ComprehendAPI(configs)
    assert api.async_client() is async_http


def test_comprehend_retry_and_logging():
    retry = RetryPolicyConfigs(max_tries=2, base_delay=0.5)
    logger = Logger()
    configs = SDKConfigs(retry_config=retry, debug_logger=logger)
    api = ComprehendAPI(configs)
    assert api.configs.retry_config is retry
    assert api._log is logger


def test_comprehend_endpoint_paths():
    configs = SDKConfigs()
    api = ComprehendAPI(configs)
    assert api.PATH + api.ENDPOINTS_CONFIG.main == "/comprehend/classify"


def test_extract_defaults():
    configs = SDKConfigs()
    api = ExtractAPI(configs)
    assert api.PATH == "/extract/"
    assert api.METHOD == "POST"
    assert api.CONTENT_TYPE == "application/json"
    assert isinstance(api.ENDPOINTS_CONFIG, ExtractEndpointsConfig)
    assert api.ENDPOINTS_CONFIG.main == ""
    assert api.ENDPOINTS_CONFIG.relations == "relations"
    assert api.ENDPOINTS_CONFIG.summary == "summary"


def test_extract_endpoint_override():
    configs = SDKConfigs()
    custom = ExtractEndpointsConfig(main="m", relations="r", summary="s")
    api = ExtractAPI(configs, endpoints_config=custom)
    assert api.ENDPOINTS_CONFIG is custom
    assert api.ENDPOINTS_CONFIG.main == "m"
    assert api.ENDPOINTS_CONFIG.relations == "r"
    assert api.ENDPOINTS_CONFIG.summary == "s"


def test_extract_inherits_clients():
    http = HttpClient(base_url="x")
    async_http = AsyncHttpClient(base_url="x")
    configs = SDKConfigs(client=http, async_client=async_http)
    api = ExtractAPI(configs)
    assert api.client() is http
    assert api.async_client() is async_http


def test_extract_path_override():
    configs = SDKConfigs()
    api = ExtractAPI(configs, path="/xpath")
    assert api.PATH == "/xpath"


@pytest.mark.asyncio
async def test_extract_async_client():
    async_http = AsyncHttpClient(base_url="async-e")
    configs = SDKConfigs(async_client=async_http)
    api = ExtractAPI(configs)
    assert api.async_client() is async_http


def test_extract_endpoint_paths():
    configs = SDKConfigs()
    api = ExtractAPI(configs)
    assert api.PATH + api.ENDPOINTS_CONFIG.main == "/extract/"
    assert api.PATH + api.ENDPOINTS_CONFIG.relations == "/extract/relations"
    assert api.PATH + api.ENDPOINTS_CONFIG.summary == "/extract/summary"


def test_extract_retry_policy():
    retry = RetryPolicyConfigs(max_tries=4, base_delay=0.2)
    configs = SDKConfigs(retry_config=retry)
    api = ExtractAPI(configs)
    assert api.configs.retry_config is retry
    assert api.configs.retry_config.max_tries == 4


# InstructAPI Tests


def test_instruct_defaults():
    configs = SDKConfigs()
    api = InstructAPI(configs)
    assert api.PATH == "/instruct/"
    assert api.METHOD == "POST"
    assert api.CONTENT_TYPE == "multipart/form-data"
    assert isinstance(api.ENDPOINTS_CONFIG, InstructEndpointsConfig)
    assert api.ENDPOINTS_CONFIG.main == ""


def test_instruct_endpoint_override():
    configs = SDKConfigs()
    custom = InstructEndpointsConfig(main="custom-instruct")
    api = InstructAPI(configs, endpoints_config=custom)
    assert api.ENDPOINTS_CONFIG is custom
    assert api.ENDPOINTS_CONFIG.main == "custom-instruct"


def test_instruct_inherits_clients():
    http = HttpClient(base_url="x")
    async_http = AsyncHttpClient(base_url="x")
    configs = SDKConfigs(client=http, async_client=async_http)
    api = InstructAPI(configs)
    assert api.client() is http
    assert api.async_client() is async_http


def test_instruct_path_override():
    configs = SDKConfigs()
    api = InstructAPI(configs, path="/custom-instruct-path")
    assert api.PATH == "/custom-instruct-path"


@pytest.mark.asyncio
async def test_instruct_async_client():
    async_http = AsyncHttpClient(base_url="async-i")
    configs = SDKConfigs(async_client=async_http)
    api = InstructAPI(configs)
    assert api.async_client() is async_http


def test_instruct_retry_and_logging():
    retry = RetryPolicyConfigs(max_tries=2, base_delay=0.5)
    logger = Logger()
    configs = SDKConfigs(retry_config=retry, debug_logger=logger)
    api = InstructAPI(configs)
    assert api.configs.retry_config is retry
    assert api._log is logger


def test_instruct_endpoint_paths():
    configs = SDKConfigs()
    api = InstructAPI(configs)
    assert api.PATH == "/instruct/"
    # With empty main endpoint, path should be just the base path
    expected_path = f"{api.PATH}/{api.ENDPOINTS_CONFIG.main}".rstrip("/")
    assert expected_path == "/instruct"


def test_instruct_retry_policy():
    retry = RetryPolicyConfigs(max_tries=6, base_delay=0.3)
    configs = SDKConfigs(retry_config=retry)
    api = InstructAPI(configs)
    assert api.configs.retry_config is retry
    assert api.configs.retry_config.max_tries == 6
