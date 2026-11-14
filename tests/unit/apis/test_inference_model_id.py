"""
Unit tests verifying model_id routing headers for Comprehend, Extract, and Instruct APIs.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from knowledgator.configs import SDKConfigs, ServersConfigs
from knowledgator.apis.extract import ExtractAPI
from knowledgator.apis.instruct import InstructAPI
from knowledgator.apis.comprehend import ComprehendAPI
from knowledgator.utils.dictionaries_op import get_model_id_header


def test_get_model_id_header_utility():
    # When model_id is None, returns empty dict
    assert get_model_id_header(None) == {}

    # When model_id is given, returns KnowledgatorAPI-Model-ID
    headers = get_model_id_header("custom-model-v1")
    assert headers == {
        "KnowledgatorAPI-Model-ID": "custom-model-v1",
    }


class TestInferenceModelIdRouting:
    @pytest.fixture
    def sdk_configs(self):
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return configs

    def test_comprehend_it_sync_attaches_adapter_id_headers(self, sdk_configs):
        api = ComprehendAPI(sdk_configs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "outputs": {
                "topic": {"type": "classify", "value": "tech", "scores": {"tech": 0.98}}
            }
        }
        mock_resp.headers = {}
        sdk_configs.client.request.return_value = mock_resp

        res = api.it(
            text="Quantum computing breakthrough",
            tasks={"topic": {"type": "classify", "labels": ["tech", "finance"]}},
            adapter_id="finetuned-comprehend-99",
        )
        assert res is not None

        call_kwargs = sdk_configs.client.request.call_args.kwargs
        headers = call_kwargs.get("headers", {})
        assert headers.get("KnowledgatorAPI-Model-ID") == "finetuned-comprehend-99"

    @pytest.mark.asyncio
    async def test_comprehend_it_async_attaches_adapter_id_headers(self, sdk_configs):
        api = ComprehendAPI(sdk_configs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "outputs": {
                "topic": {"type": "classify", "value": "tech", "scores": {"tech": 0.98}}
            }
        }
        mock_resp.headers = {}
        sdk_configs.async_client.request = AsyncMock(return_value=mock_resp)

        res = await api.it_async(
            text="Quantum computing breakthrough",
            tasks={"topic": {"type": "classify", "labels": ["tech", "finance"]}},
            adapter_id="finetuned-comprehend-99",
        )
        assert res is not None

        call_kwargs = sdk_configs.async_client.request.call_args.kwargs
        headers = call_kwargs.get("headers", {})
        assert headers.get("KnowledgatorAPI-Model-ID") == "finetuned-comprehend-99"

    def test_extract_it_sync_attaches_adapter_id_headers(self, sdk_configs):
        api = ExtractAPI(sdk_configs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "outputs": {
                "entities": {
                    "type": "span",
                    "value": [{"label": "ORG", "span": "Apple", "start": 0, "end": 5, "score": 0.99}],
                }
            }
        }
        mock_resp.headers = {}
        sdk_configs.client.request.return_value = mock_resp

        res = api.it(
            text="Apple released a new chip.",
            tasks={"entities": {"type": "span", "labels": ["ORG", "PRODUCT"]}},
            adapter_id="gliclass-adapter",
        )
        assert res is not None

        call_kwargs = sdk_configs.client.request.call_args.kwargs
        headers = call_kwargs.get("headers", {})
        assert headers.get("KnowledgatorAPI-Model-ID") == "gliclass-adapter"

    @pytest.mark.asyncio
    async def test_extract_it_async_attaches_adapter_id_headers(self, sdk_configs):
        api = ExtractAPI(sdk_configs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "outputs": {
                "entities": {
                    "type": "span",
                    "value": [{"label": "ORG", "span": "Apple", "start": 0, "end": 5, "score": 0.99}],
                }
            }
        }
        mock_resp.headers = {}
        sdk_configs.async_client.request = AsyncMock(return_value=mock_resp)

        res = await api.it_async(
            text="Apple released a new chip.",
            tasks={"entities": {"type": "span", "labels": ["ORG", "PRODUCT"]}},
            adapter_id="gliclass-adapter",
        )
        assert res is not None

        call_kwargs = sdk_configs.async_client.request.call_args.kwargs
        headers = call_kwargs.get("headers", {})
        assert headers.get("KnowledgatorAPI-Model-ID") == "gliclass-adapter"

    def test_instruct_it_sync_attaches_adapter_id_headers(self, sdk_configs):
        api = InstructAPI(sdk_configs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"outputs": "Structured extraction result"}
        mock_resp.headers = {}
        sdk_configs.client.request.return_value = mock_resp

        res = api.it(
            text="Extract entities from this text",
            template='{"entities": []}',
            adapter_id="custom-instruct-42",
        )
        assert res is not None

        call_kwargs = sdk_configs.client.request.call_args.kwargs
        headers = call_kwargs.get("headers", {})
        assert headers.get("KnowledgatorAPI-Model-ID") == "custom-instruct-42"

    @pytest.mark.asyncio
    async def test_instruct_it_async_attaches_adapter_id_headers(self, sdk_configs):
        api = InstructAPI(sdk_configs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"outputs": "Structured extraction result"}
        mock_resp.headers = {}
        sdk_configs.async_client.request = AsyncMock(return_value=mock_resp)

        res = await api.it_async(
            text="Extract entities from this text",
            template='{"entities": []}',
            adapter_id="custom-instruct-42",
        )
        assert res is not None

        call_kwargs = sdk_configs.async_client.request.call_args.kwargs
        headers = call_kwargs.get("headers", {})
        assert headers.get("KnowledgatorAPI-Model-ID") == "custom-instruct-42"


