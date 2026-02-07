from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import ValidationError

from knowledgator.configs import SDKConfigs, ServersConfigs
from knowledgator.exceptions import (
    ServerError,
    TransportTimeout,
    ConnectionFailure,
)
from knowledgator.models.inputs import ComprehendRequest, ClassificationExample
from knowledgator.models.outputs import ComprehendResults, ComprehendResponse
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
    """Tests for ComprehendRequest model."""

    def test_minimal_request(self):
        req = ComprehendRequest(text="This is a great product!", labels=["positive", "negative", "neutral"])
        assert req.text == "This is a great product!"
        assert req.labels == ["positive", "negative", "neutral"]
        assert req.threshold == 0.5  # default
        assert req.examples is None

    def test_request_with_threshold(self):
        req = ComprehendRequest(text="Sample text", labels=["label1", "label2"], threshold=0.7)
        assert req.threshold == 0.7

    def test_request_with_examples(self):
        examples = [
            ClassificationExample(text="Great product", label="positive"),
            ClassificationExample(text="Terrible service", label="negative"),
        ]
        req = ComprehendRequest(text="Test text", labels=["positive", "negative"], examples=examples)
        assert len(req.examples) == 2
        assert req.examples[0].text == "Great product"
        assert req.examples[0].label == "positive"

    def test_threshold_validation_min(self):
        """Test that threshold < 0 raises validation error."""
        with pytest.raises(ValidationError):
            ComprehendRequest(text="Test", labels=["a", "b"], threshold=-0.1)

    def test_threshold_validation_max(self):
        """Test that threshold > 1 raises validation error."""
        with pytest.raises(ValidationError):
            ComprehendRequest(text="Test", labels=["a", "b"], threshold=1.1)


class TestComprehendResponse:
    """Tests for ComprehendResponse model."""

    def test_response_creation(self):
        resp = ComprehendResponse(label="positive", score=0.95)
        assert resp.label == "positive"
        assert resp.score == 0.95

    def test_response_from_dict(self):
        data = {"label": "negative", "score": 0.12}
        resp = ComprehendResponse.model_validate(data)
        assert resp.label == "negative"
        assert resp.score == 0.12


class TestComprehendResults:
    """Tests for ComprehendResults model."""

    def test_results_creation(self):
        results = [ComprehendResponse(label="positive", score=0.92), ComprehendResponse(label="neutral", score=0.65)]
        comprehend_results = ComprehendResults(
            results=results,
            input_text="This is great!",
            labels=["positive", "negative", "neutral"],
            threshold=0.5,
            status="ok",
        )
        assert len(comprehend_results.results) == 2
        assert comprehend_results.input_text == "This is great!"
        assert comprehend_results.threshold == 0.5
        assert comprehend_results.status == "ok"

    def test_results_filtering_by_threshold(self):
        """Test that results above threshold are included."""
        results = [
            ComprehendResponse(label="positive", score=0.92),
            ComprehendResponse(label="neutral", score=0.45),
            ComprehendResponse(label="negative", score=0.15),
        ]
        comprehend_results = ComprehendResults(
            results=results, input_text="Test", labels=["positive", "negative", "neutral"], threshold=0.5, status="ok"
        )
        # Note: filtering is done by the API, not the model
        assert len(comprehend_results.results) == 3


class TestComprehendAPISync:
    """Tests for synchronous ComprehendAPI.it() method."""

    def test_basic_classification(self, comprehend_api):
        """Test basic classification without optional parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"label": "positive", "score": 0.95}, {"label": "neutral", "score": 0.52}]
        }
        mock_response.headers = {}
        comprehend_api.configs.client.request.return_value = mock_response

        result = comprehend_api.it(text="This is a great product!", labels=["positive", "negative", "neutral"])

        assert isinstance(result, ComprehendResults)
        assert len(result.results) == 2
        assert result.results[0].label == "positive"
        assert result.results[0].score == 0.95
        assert result.input_text == "This is a great product!"

    def test_classification_with_threshold(self, comprehend_api):
        """Test classification with custom threshold."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": [{"label": "technology", "score": 0.88}]}
        mock_response.headers = {}
        comprehend_api.configs.client.request.return_value = mock_response

        result = comprehend_api.it(
            text="AI is transforming the world", labels=["technology", "politics", "sports"], threshold=0.7
        )

        assert len(result.results) == 1
        assert result.results[0].label == "technology"
        assert result.threshold == 0.7

    def test_classification_with_examples(self, comprehend_api):
        """Test classification with few-shot examples."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": [{"label": "complaint", "score": 0.91}]}
        mock_response.headers = {}
        comprehend_api.configs.client.request.return_value = mock_response

        examples = [
            {"text": "Great service!", "label": "praise"},
            {"text": "Terrible experience", "label": "complaint"},
        ]

        result = comprehend_api.it(
            text="Very disappointed with the quality", labels=["praise", "complaint", "question"], examples=examples
        )

        assert result.results[0].label == "complaint"

    def test_multi_label_classification(self, comprehend_api):
        """Test multi-label classification returning multiple labels."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {"label": "technology", "score": 0.92},
                {"label": "business", "score": 0.85},
                {"label": "innovation", "score": 0.78},
            ]
        }
        mock_response.headers = {}
        comprehend_api.configs.client.request.return_value = mock_response

        result = comprehend_api.it(
            text="Apple announces new AI-powered features for business users",
            labels=["technology", "business", "sports", "politics", "innovation"],
            threshold=0.5,
        )

        assert len(result.results) == 3
        assert all(r.score >= 0.5 for r in result.results)

    def test_no_labels_above_threshold(self, comprehend_api):
        """Test when no labels meet the threshold."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": []}
        mock_response.headers = {}
        comprehend_api.configs.client.request.return_value = mock_response

        result = comprehend_api.it(text="Ambiguous text", labels=["label1", "label2", "label3"], threshold=0.9)

        assert len(result.results) == 0

    def test_timeout_exception(self, comprehend_api):
        """Test that timeout raises TransportTimeout."""
        comprehend_api.configs.client.request.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(TransportTimeout):
            comprehend_api.it(text="Test text", labels=["label1", "label2"])

    def test_connection_error(self, comprehend_api):
        """Test that connection errors raise ConnectionFailure."""
        comprehend_api.configs.client.request.side_effect = httpx.RequestError("connection failed")

        with pytest.raises(ConnectionFailure):
            comprehend_api.it(text="Test text", labels=["label1", "label2"])

    def test_server_error_response(self, comprehend_api):
        """Test handling of server error responses."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.return_value = {"error": "Server error"}
        mock_response.headers = {"x-request-id": "req-123"}
        comprehend_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError):
            comprehend_api.it(text="Test text", labels=["label1", "label2"])

    def test_missing_outputs_in_response(self, comprehend_api):
        """Test that missing 'outputs' in response raises ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "something"}  # No 'outputs' key
        mock_response.headers = {"x-request-id": "req-456"}
        comprehend_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            comprehend_api.it(text="Test text", labels=["label1", "label2"])
        assert "missing 'outputs'" in str(exc_info.value)

    def test_invalid_outputs_format(self, comprehend_api):
        """Test that invalid outputs format raises ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": "not a list"  # Should be a list
        }
        mock_response.headers = {"x-request-id": "req-789"}
        comprehend_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            comprehend_api.it(text="Test text", labels=["label1", "label2"])
        assert "missing 'outputs'" in str(exc_info.value)

    def test_malformed_output_item(self, comprehend_api):
        """Test that malformed items in outputs raise ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {"label": "test"},  # Missing 'score'
            ]
        }
        mock_response.headers = {"x-request-id": "req-999"}
        comprehend_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            comprehend_api.it(text="Test text", labels=["label1", "label2"])
        assert "Unexpected record format" in str(exc_info.value)

    def test_non_json_response(self, comprehend_api):
        """Test that non-JSON response raises ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("not json")
        mock_response.text = "Plain text response"
        mock_response.headers = {}
        comprehend_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            comprehend_api.it(text="Test text", labels=["label1", "label2"])
        assert "not valid JSON" in str(exc_info.value)


class TestComprehendAPIAsync:
    """Tests for asynchronous ComprehendAPI.it_async() method."""

    @pytest.mark.asyncio
    async def test_basic_classification_async(self, comprehend_api):
        """Test basic async classification."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": [{"label": "positive", "score": 0.89}]}
        mock_response.headers = {}

        comprehend_api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await comprehend_api.it_async(text="Great product!", labels=["positive", "negative", "neutral"])

        assert result.results[0].label == "positive"
        assert result.results[0].score == 0.89

    @pytest.mark.asyncio
    async def test_async_with_threshold(self, comprehend_api):
        """Test async classification with threshold."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": [{"label": "urgent", "score": 0.95}]}
        mock_response.headers = {}

        comprehend_api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await comprehend_api.it_async(
            text="URGENT: System down!", labels=["urgent", "normal", "low-priority"], threshold=0.8
        )

        assert result.results[0].label == "urgent"
        assert result.threshold == 0.8

    @pytest.mark.asyncio
    async def test_async_timeout(self, comprehend_api):
        """Test async timeout handling."""
        comprehend_api.configs.async_client.request = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with pytest.raises(TransportTimeout):
            await comprehend_api.it_async(text="Test", labels=["label1", "label2"])

    @pytest.mark.asyncio
    async def test_async_connection_error(self, comprehend_api):
        """Test async connection error handling."""
        comprehend_api.configs.async_client.request = AsyncMock(side_effect=httpx.RequestError("connection failed"))

        with pytest.raises(ConnectionFailure):
            await comprehend_api.it_async(text="Test", labels=["label1", "label2"])


class TestClassificationExample:
    """Tests for ClassificationExample dataclass."""

    def test_example_creation(self):
        """Test creating a classification example."""
        example = ClassificationExample(text="This is amazing!", label="positive")
        assert example.text == "This is amazing!"
        assert example.label == "positive"

    def test_example_from_dict(self):
        """Test creating example from dictionary."""
        data = {"text": "Bad service", "label": "negative"}
        example = ClassificationExample.model_validate(data)
        assert example.text == "Bad service"
        assert example.label == "negative"


class TestThresholdBehavior:
    """Tests for different threshold values."""

    @pytest.fixture
    def api_with_mock(self, comprehend_api):
        def setup_response(outputs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"outputs": outputs}
            mock_response.headers = {}
            comprehend_api.configs.client.request.return_value = mock_response
            return comprehend_api

        return setup_response

    def test_default_threshold(self, api_with_mock):
        """Test that default threshold is 0.5."""
        api = api_with_mock([{"label": "high", "score": 0.8}, {"label": "medium", "score": 0.6}])
        result = api.it(text="Test", labels=["high", "medium", "low"])
        assert result.threshold == 0.5

    def test_high_threshold(self, api_with_mock):
        """Test with high threshold (0.9)."""
        api = api_with_mock([{"label": "confident", "score": 0.95}])
        result = api.it(text="Test", labels=["confident", "uncertain"], threshold=0.9)
        assert result.threshold == 0.9
        assert len(result.results) == 1

    def test_low_threshold(self, api_with_mock):
        """Test with low threshold (0.1)."""
        api = api_with_mock(
            [{"label": "label1", "score": 0.3}, {"label": "label2", "score": 0.2}, {"label": "label3", "score": 0.15}]
        )
        result = api.it(text="Test", labels=["label1", "label2", "label3"], threshold=0.1)
        assert result.threshold == 0.1
        assert len(result.results) == 3
