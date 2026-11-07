from unittest.mock import MagicMock

import pytest

from knowledgator.configs import SDKConfigs, ServersConfigs
from knowledgator.apis.extract import ExtractAPI
from knowledgator.apis.instruct import InstructAPI
from knowledgator.apis.comprehend import ComprehendAPI


class TestVerification:
    @pytest.fixture
    def mock_configs(self):
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return configs

    def test_comprehend_path_and_adapter_id(self, mock_configs):
        api = ComprehendAPI(mock_configs)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": []}
        api.configs.client.request.return_value = mock_response

        api.it(text="test", labels=["l1"], adapter_id="my-custom-model")

        call_kwargs = api.configs.client.request.call_args.kwargs
        url = call_kwargs["url"]
        headers = call_kwargs["headers"]

        assert "/inference/comprehend/" in str(url)
        assert headers.get("KnowledgatorAPI-Model-ID") == "my-custom-model"

    def test_extract_path_and_adapter_id(self, mock_configs):
        api = ExtractAPI(mock_configs)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": []}
        api.configs.client.request.return_value = mock_response

        api.it(text="test", labels=["l1"], adapter_id="my-extract-model")

        call_kwargs = api.configs.client.request.call_args.kwargs
        url = call_kwargs["url"]
        headers = call_kwargs["headers"]

        assert "/inference/extract/" in str(url)
        assert headers.get("KnowledgatorAPI-Model-ID") == "my-extract-model"

    def test_instruct_path_and_adapter_id(self, mock_configs):
        api = InstructAPI(mock_configs)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": {"result": "ok"}}
        api.configs.client.request.return_value = mock_response

        api.it(text="test", adapter_id="my-instruct-model")

        call_kwargs = api.configs.client.request.call_args.kwargs
        url = call_kwargs["url"]
        headers = call_kwargs["headers"]

        assert "/inference/instruct" in str(url)
        assert headers.get("KnowledgatorAPI-Model-ID") == "my-instruct-model"

    @pytest.mark.asyncio
    async def test_comprehend_async_path_and_adapter_id(self, mock_configs):
        from unittest.mock import AsyncMock

        api = ComprehendAPI(mock_configs)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": []}
        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        await api.it_async(text="test", labels=["l1"], adapter_id="async-model")

        call_kwargs = api.configs.async_client.request.call_args.kwargs
        url = call_kwargs["url"]
        headers = call_kwargs["headers"]

        assert "/inference/comprehend/" in str(url)
        assert headers.get("knowledgatorapi-model-id") == "async-model"
