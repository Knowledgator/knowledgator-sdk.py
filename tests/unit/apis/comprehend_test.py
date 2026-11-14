from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import ValidationError

from knowledgator.models import (
    ComprehendRequest,
    ComprehendResults,
    ClassifyTaskOutput,
)
from knowledgator.configs import SDKConfigs, ServersConfigs
from knowledgator.exceptions import (
    ServerError,
    TransportTimeout,
    ConnectionFailure,
)
from knowledgator.apis.comprehend import ComprehendAPI


@pytest.fixture
def mock_configs():
    """Create mock SDK configs for testing."""
    configs = SDKConfigs()
    configs.server_config = ServersConfigs(servers={"main": "https://api.test.com"})
    configs.client = MagicMock()
    configs.async_client = MagicMock()
    configs.user_agent = "test-agent/1.0"
    return configs


@pytest.fixture
def comprehend_api(mock_configs):
    """Create ComprehendAPI instance with mocked configs."""
    return ComprehendAPI(mock_configs)


class TestComprehendRequest:
    """Tests for new Phase 1 ComprehendRequest model."""

    def test_minimal_request(self):
        req = ComprehendRequest(
            input="This is a great product!",
            tasks={"topic": {"type": "classify", "labels": ["positive", "negative", "neutral"]}},
        )
        assert req.input.root == "This is a great product!"
        assert "topic" in req.tasks

    def test_request_with_options(self):
        req = ComprehendRequest(
            input="Sample text",
            tasks={"topic": {"type": "classify", "labels": ["label1", "label2"], "threshold": 0.7}},
            options={"adapter_id": "custom-adapter"},
        )
        assert req.tasks["topic"].threshold == 0.7
        assert req.options.adapter_id == "custom-adapter"

    def test_missing_required_fields(self):
        """Test that missing tasks raises validation error."""
        with pytest.raises(ValidationError):
            ComprehendRequest(input="Test")


class TestComprehendResults:
    """Tests for ComprehendResults model."""

    def test_results_creation(self):
        outputs = {
            "topic": ClassifyTaskOutput(value="positive", scores={"positive": 0.92, "neutral": 0.65}),
        }
        comprehend_results = ComprehendResults(
            outputs=outputs,
            input_text="This is great!",
            threshold=0.5,
            status="ok",
        )
        assert "topic" in comprehend_results.outputs
        assert comprehend_results.outputs["topic"].value == "positive"
        assert comprehend_results.input_text == "This is great!"
        assert comprehend_results.threshold == 0.5
        assert comprehend_results.status == "ok"

    def test_results_access_via_getattr_and_getitem(self):
        comprehend_results = ComprehendResults(
            {
                "outputs": {
                    "topic": {"type": "classify", "value": "tech", "scores": {"tech": 0.9}},
                }
            }
        )
        assert comprehend_results.topic.value == "tech"
        assert comprehend_results["topic"].value == "tech"
        assert "topic" in comprehend_results


class TestComprehendAPISync:
    """Tests for synchronous ComprehendAPI.it() method."""

    def test_labels_argument_raises_type_error(self, comprehend_api):
        """Test that passing 'labels' raises TypeError informing the user it was removed."""
        with pytest.raises(TypeError) as exc:
            comprehend_api.it(text="Test", labels=["a", "b"])
        assert "The 'labels' argument has been removed" in str(exc.value)

    def test_basic_classification(self, comprehend_api):
        """Test basic classification without optional parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": {
                "topic": {
                    "type": "classify",
                    "value": "positive",
                    "scores": {"positive": 0.95, "neutral": 0.52},
                }
            }
        }
        mock_response.headers = {}
        comprehend_api.configs.client.request.return_value = mock_response

        result = comprehend_api.it(
            text="This is a great product!",
            tasks={"topic": {"type": "classify", "labels": ["positive", "negative", "neutral"]}},
        )

        assert isinstance(result, ComprehendResults)
        assert result.topic.value == "positive"
        assert result.topic.scores["positive"] == 0.95
        assert result.input_text == "This is a great product!"

    def test_classification_with_threshold(self, comprehend_api):
        """Test classification with custom threshold."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": {
                "topic": {
                    "type": "classify",
                    "value": "technology",
                    "scores": {"technology": 0.88},
                    "threshold": 0.7,
                }
            }
        }
        mock_response.headers = {}
        comprehend_api.configs.client.request.return_value = mock_response

        result = comprehend_api.it(
            text="AI is transforming the world",
            tasks={"topic": {"type": "classify", "labels": ["technology", "politics", "sports"], "threshold": 0.7}},
        )

        assert result.topic.value == "technology"
        assert result.threshold == 0.7

    def test_classification_with_examples(self, comprehend_api):
        """Test classification with few-shot examples."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": {
                "topic": {
                    "type": "classify",
                    "value": "complaint",
                    "scores": {"complaint": 0.91},
                }
            }
        }
        mock_response.headers = {}
        comprehend_api.configs.client.request.return_value = mock_response

        result = comprehend_api.it(
            text="Very disappointed with the quality",
            tasks={"topic": {"type": "classify", "labels": ["praise", "complaint", "question"]}},
        )

        assert result.topic.value == "complaint"

    def test_multi_label_classification(self, comprehend_api):
        """Test multi-label classification returning multiple labels."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": {
                "topic": {
                    "type": "classify",
                    "value": ["technology", "business", "innovation"],
                    "scores": {"technology": 0.92, "business": 0.85, "innovation": 0.78},
                }
            }
        }
        mock_response.headers = {}
        comprehend_api.configs.client.request.return_value = mock_response

        result = comprehend_api.it(
            text="Apple announces new AI-powered features for business users",
            tasks={
                "topic": {
                    "type": "classify",
                    "labels": ["technology", "business", "sports", "politics", "innovation"],
                    "threshold": 0.5,
                }
            },
        )

        assert result.topic.value == ["technology", "business", "innovation"]

    def test_no_labels_above_threshold(self, comprehend_api):
        """Test when no labels meet the threshold."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": {
                "topic": {
                    "type": "classify",
                    "value": [],
                    "scores": {"label1": 0.1, "label2": 0.2, "label3": 0.3},
                    "threshold": 0.9,
                }
            }
        }
        mock_response.headers = {}
        comprehend_api.configs.client.request.return_value = mock_response

        result = comprehend_api.it(
            text="Ambiguous text",
            tasks={"topic": {"type": "classify", "labels": ["label1", "label2", "label3"], "threshold": 0.9}},
        )

        assert len(result.topic.value) == 0

    def test_timeout_exception(self, comprehend_api):
        """Test that timeout raises TransportTimeout."""
        comprehend_api.configs.client.request.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(TransportTimeout):
            comprehend_api.it(text="Test text", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})

    def test_connection_error(self, comprehend_api):
        """Test that connection errors raise ConnectionFailure."""
        comprehend_api.configs.client.request.side_effect = httpx.RequestError("connection failed")

        with pytest.raises(ConnectionFailure):
            comprehend_api.it(text="Test text", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})

    def test_server_error_response(self, comprehend_api):
        """Test handling of server error responses."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.return_value = {"error": "Server error"}
        mock_response.headers = {"x-request-id": "req-123"}
        comprehend_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError):
            comprehend_api.it(text="Test text", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})

    def test_missing_outputs_in_response(self, comprehend_api):
        """Test that missing 'outputs' in response raises ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "something"}  # No 'outputs' key
        mock_response.headers = {"x-request-id": "req-456"}
        comprehend_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            comprehend_api.it(text="Test text", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})
        assert "missing 'outputs'" in str(exc_info.value)

    def test_invalid_outputs_format(self, comprehend_api):
        """Test that invalid outputs format (like a list) raises ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": "not a dict"}
        mock_response.headers = {"x-request-id": "req-789"}
        comprehend_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError):
            comprehend_api.it(text="Test text", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})

    def test_malformed_output_item(self, comprehend_api):
        """Test that legacy list format in outputs raises ServerError (no silent fallback)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": [{"label": "test"}]}
        mock_response.headers = {"x-request-id": "req-999"}
        comprehend_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError):
            comprehend_api.it(text="Test text", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})

    def test_non_json_response(self, comprehend_api):
        """Test that non-JSON response raises ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("not json")
        mock_response.text = "Plain text response"
        mock_response.headers = {}
        comprehend_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            comprehend_api.it(text="Test text", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})
        assert "not valid JSON" in str(exc_info.value)


class TestComprehendAPIAsync:
    """Tests for asynchronous ComprehendAPI.it_async() method."""

    @pytest.mark.asyncio
    async def test_basic_classification_async(self, comprehend_api):
        """Test basic async classification."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": {
                "topic": {
                    "type": "classify",
                    "value": "positive",
                    "scores": {"positive": 0.89},
                }
            }
        }
        mock_response.headers = {}

        comprehend_api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await comprehend_api.it_async(
            text="Great product!",
            tasks={"topic": {"type": "classify", "labels": ["positive", "negative", "neutral"]}},
        )

        assert result.topic.value == "positive"
        assert result.topic.scores["positive"] == 0.89

    @pytest.mark.asyncio
    async def test_async_with_threshold(self, comprehend_api):
        """Test async classification with threshold."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": {
                "topic": {
                    "type": "classify",
                    "value": "urgent",
                    "scores": {"urgent": 0.95},
                    "threshold": 0.8,
                }
            }
        }
        mock_response.headers = {}

        comprehend_api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await comprehend_api.it_async(
            text="URGENT: System down!",
            tasks={"topic": {"type": "classify", "labels": ["urgent", "normal", "low-priority"], "threshold": 0.8}},
        )

        assert result.topic.value == "urgent"
        assert result.threshold == 0.8

    @pytest.mark.asyncio
    async def test_async_timeout(self, comprehend_api):
        """Test async timeout handling."""
        comprehend_api.configs.async_client.request = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with pytest.raises(TransportTimeout):
            await comprehend_api.it_async(text="Test", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})

    @pytest.mark.asyncio
    async def test_async_connection_error(self, comprehend_api):
        """Test async connection error handling."""
        comprehend_api.configs.async_client.request = AsyncMock(side_effect=httpx.RequestError("connection failed"))

        with pytest.raises(ConnectionFailure):
            await comprehend_api.it_async(text="Test", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})


class TestThresholdBehavior:
    """Tests for different threshold values."""

    @pytest.fixture
    def api_with_mock(self, comprehend_api):
        def setup_response(threshold, value):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "outputs": {
                    "topic": {
                        "type": "classify",
                        "value": value,
                        "threshold": threshold,
                    }
                }
            }
            mock_response.headers = {}
            comprehend_api.configs.client.request.return_value = mock_response
            return comprehend_api

        return setup_response

    def test_default_threshold(self, api_with_mock):
        """Test that default threshold is 0.5."""
        api = api_with_mock(0.5, "high")
        result = api.it(text="Test", tasks={"topic": {"type": "classify", "labels": ["high", "medium", "low"]}})
        assert result.topic.threshold == 0.5

    def test_high_threshold(self, api_with_mock):
        """Test with high threshold (0.9)."""
        api = api_with_mock(0.9, "confident")
        result = api.it(
            text="Test",
            tasks={"topic": {"type": "classify", "labels": ["confident", "uncertain"], "threshold": 0.9}},
        )
        assert result.threshold == 0.9
        assert result.topic.value == "confident"

    def test_low_threshold(self, api_with_mock):
        """Test with low threshold (0.1)."""
        api = api_with_mock(0.1, ["label1", "label2", "label3"])
        result = api.it(
            text="Test",
            tasks={"topic": {"type": "classify", "labels": ["label1", "label2", "label3"], "threshold": 0.1}},
        )
        assert result.threshold == 0.1
        assert len(result.topic.value) == 3
