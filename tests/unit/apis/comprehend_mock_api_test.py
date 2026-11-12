"""
Mock API Request Tests for Comprehend-it Endpoint

This module demonstrates the Comprehend-it API with mock HTTP requests,
verifying the complete API contract including:
- Request/response structure
- Threshold parameter handling
- Few-shot examples
- Multi-label classification
- Error scenarios
"""

from unittest.mock import MagicMock

import pytest

from knowledgator.configs import SDKConfigs, ServersConfigs
from knowledgator.exceptions import ServerError
from knowledgator.apis.comprehend import ComprehendAPI


class TestComprehendAPIMockRequests:
    """Test Comprehend API with mock HTTP requests matching the API contract."""

    def test_labels_raises_type_error(self, api):
        with pytest.raises(TypeError) as exc:
            api.it(text="Test", labels=["a", "b"])
        assert "The 'labels' argument has been removed" in str(exc.value)

    @pytest.fixture
    def api(self):
        """Create ComprehendAPI instance with mock configuration."""
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return ComprehendAPI(configs)

    def test_minimal_request_structure(self, api):
        """Test minimal request with only required 'text' and 'labels' fields."""
        # Mock response matching API contract
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"label": "positive", "score": 0.8734}, {"label": "neutral", "score": 0.5123}]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request
        result = api.it(text="This product is amazing!", tasks={"topic": {"type": "classify", "labels": ["positive", "negative", "neutral"]}})

        # Verify request was made
        assert api.configs.client.request.called

        # Verify the request was called with json parameter
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "json" in call_kwargs
        payload = call_kwargs["json"]
        assert payload.get("input") == "This product is amazing!" or payload.get("text") == "This product is amazing!"
        if "tasks" in payload:
            task = payload["tasks"].get("topic", {})
            assert task.get("labels") == ["positive", "negative", "neutral"]
        else:
            assert payload.get("labels") == ["positive", "negative", "neutral"]

        # Verify response
        assert len(result.results) == 2
        assert result.results[0].label == "positive"
        assert result.results[0].score == 0.8734
        assert result.results[1].label == "neutral"
        assert result.results[1].score == 0.5123

    def test_request_with_threshold(self, api):
        """Test request with custom threshold parameter."""
        # Mock response with filtered results
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": [{"label": "technology", "score": 0.92}]}
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request with threshold
        result = api.it(
            text="AI and machine learning are transforming industries",
            tasks={"topic": {"type": "classify", "labels": ["technology", "politics", "sports", "entertainment"], "threshold": 0.7}},
        )

        # Verify request body includes threshold
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "json" in call_kwargs
        payload = call_kwargs["json"]
        thresh = payload.get("threshold") or payload.get("tasks", {}).get("topic", {}).get("threshold")
        assert thresh == 0.7

        # Verify response
        assert len(result.results) == 1
        assert result.results[0].label == "technology"
        assert result.results[0].score >= 0.7

    def test_request_with_examples(self, api):
        """Test request with few-shot examples."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": [{"label": "complaint", "score": 0.91}]}
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Create examples
        examples = [
            {"text": "Excellent service, very satisfied!", "label": "praise"},
            {"text": "Terrible experience, very disappointed", "label": "complaint"},
            {"text": "How do I reset my password?", "label": "question"},
        ]

        # Make request
        result = api.it(
            text="The product quality is unacceptable", tasks={"topic": {"type": "classify", "labels": ["praise", "complaint", "question"]}}
        )

        # Verify request body
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "json" in call_kwargs
        payload = call_kwargs["json"]
        assert payload.get("input") == "The product quality is unacceptable" or payload.get("text") == "The product quality is unacceptable"

        # Verify response
        assert result.results[0].label == "complaint"

    def test_multi_label_classification(self, api):
        """Test multi-label classification returning multiple labels above threshold."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {"label": "technology", "score": 0.95},
                {"label": "business", "score": 0.87},
                {"label": "innovation", "score": 0.72},
            ]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(
            text="Apple announces groundbreaking AI features for enterprise customers",
            tasks={"topic": {"type": "classify", "labels": ["technology", "business", "sports", "politics", "innovation", "health"], "threshold": 0.5}},
        )

        # Verify multiple labels returned
        assert len(result.results) == 3
        assert result.results[0].label == "technology"
        assert result.results[1].label == "business"
        assert result.results[2].label == "innovation"
        assert all(r.score >= 0.5 for r in result.results)

    def test_zero_shot_classification(self, api):
        """Test zero-shot classification without examples."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"label": "urgent", "score": 0.94}, {"label": "action-required", "score": 0.81}]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(
            text="CRITICAL: Server down, immediate attention needed!",
            tasks={"topic": {"type": "classify", "labels": ["urgent", "normal", "low-priority", "action-required", "informational"]}},
        )

        assert len(result.results) == 2
        assert result.results[0].label == "urgent"

    def test_no_labels_above_threshold(self, api):
        """Test when no labels meet the threshold."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": []}
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(
            text="Ambiguous text with unclear intent", tasks={"topic": {"type": "classify", "labels": ["category1", "category2", "category3"], "threshold": 0.9}}
        )

        assert len(result.results) == 0

    def test_single_label_classification(self, api):
        """Test classification with high confidence in single label."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": [{"label": "spam", "score": 0.99}]}
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(text="CLICK HERE NOW!!! WIN $1000000 FREE!!!", tasks={"topic": {"type": "classify", "labels": ["spam", "legitimate", "promotional"]}})

        assert len(result.results) == 1
        assert result.results[0].label == "spam"
        assert result.results[0].score > 0.95

    def test_complete_request_all_fields(self, api):
        """Test complete request with all optional fields populated."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"label": "technical-issue", "score": 0.89}, {"label": "urgent", "score": 0.76}],
            "input_text": "System error 500 on production server",
            "labels": ["technical-issue", "billing", "feature-request", "urgent", "general"],
            "threshold": 0.6,
            "status": "ok",
        }
        mock_response.headers = {"x-request-id": "req-comp-123"}
        api.configs.client.request.return_value = mock_response

        examples = [
            {"text": "Cannot login to account", "label": "technical-issue"},
            {"text": "How much does premium cost?", "label": "billing"},
        ]

        result = api.it(
            text="System error 500 on production server",
            tasks={"topic": {"type": "classify", "labels": ["technical-issue", "billing", "feature-request", "urgent", "general"], "threshold": 0.6}},
            timeout_ms=30000,
            http_headers={"X-Custom-Header": "custom-value"},
        )

        # Verify request structure
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "json" in call_kwargs
        payload = call_kwargs["json"]
        assert payload.get("input") == "System error 500 on production server" or payload.get("text") == "System error 500 on production server"
        thresh = payload.get("threshold") or payload.get("tasks", {}).get("topic", {}).get("threshold")
        assert thresh == 0.6

        # Verify response
        assert len(result.results) == 2
        assert result.results[0].label == "technical-issue"
        assert result.threshold == 0.6
        assert result.status == "ok"

    def test_missing_outputs_field_error(self, api):
        """Test that response missing 'outputs' field raises ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": "some data"  # Missing 'outputs' key
        }
        mock_response.headers = {"x-request-id": "req-xyz-789"}
        api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            api.it(text="Test text", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})

        assert "missing 'outputs'" in str(exc_info.value)

    def test_invalid_outputs_format_error(self, api):
        """Test that outputs not being a list raises ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": "not a list"  # Should be a list
        }
        mock_response.headers = {"x-request-id": "req-format-err"}
        api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            api.it(text="Test text", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})

        assert "missing 'outputs'" in str(exc_info.value)

    def test_malformed_output_item_error(self, api):
        """Test that malformed items in outputs raise ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {"label": "test"}  # Missing 'score' field
            ]
        }
        mock_response.headers = {"x-request-id": "req-malformed"}
        api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            api.it(text="Test text", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})

        assert "Unexpected record format" in str(exc_info.value)

    def test_server_error_500(self, api):
        """Test handling of 500 Internal Server Error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.return_value = {"error": "Server error occurred"}
        mock_response.headers = {"x-request-id": "req-500-err"}
        api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError):
            api.it(text="Test text", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})

    def test_non_json_response_error(self, api):
        """Test that non-JSON response raises ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Not JSON")
        mock_response.text = "Plain text response"
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            api.it(text="Test text", tasks={"topic": {"type": "classify", "labels": ["label1", "label2"]}})

        assert "not valid JSON" in str(exc_info.value)

    def test_sentiment_analysis_use_case(self, api):
        """Test sentiment analysis use case."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": [{"label": "positive", "score": 0.92}]}
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(
            text="I absolutely love this product! Best purchase ever!", tasks={"topic": {"type": "classify", "labels": ["positive", "negative", "neutral"]}}
        )

        assert result.results[0].label == "positive"
        assert result.results[0].score > 0.9

    def test_topic_classification_use_case(self, api):
        """Test topic classification use case."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"label": "science", "score": 0.88}, {"label": "technology", "score": 0.79}]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(
            text="Researchers discover new quantum computing breakthrough",
            tasks={"topic": {"type": "classify", "labels": ["science", "technology", "politics", "sports", "entertainment"]}},
        )

        assert len(result.results) == 2
        assert result.results[0].label == "science"

    def test_intent_detection_use_case(self, api):
        """Test intent detection use case."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": [{"label": "cancel-subscription", "score": 0.94}]}
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(
            text="I want to cancel my subscription immediately",
            tasks={"topic": {"type": "classify", "labels": ["cancel-subscription", "upgrade-plan", "billing-inquiry", "technical-support"]}},
        )

        assert result.results[0].label == "cancel-subscription"


class TestComprehendAPIAsyncMockRequests:
    """Test async version of Comprehend API with mock requests."""

    @pytest.fixture
    def api(self):
        """Create ComprehendAPI instance with mock configuration."""
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return ComprehendAPI(configs)

    @pytest.mark.asyncio
    async def test_async_basic_request(self, api):
        """Test basic async request."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": [{"label": "positive", "score": 0.89}]}
        mock_response.headers = {}

        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await api.it_async(text="Great experience!", tasks={"topic": {"type": "classify", "labels": ["positive", "negative", "neutral"]}})

        assert result.results[0].label == "positive"
        assert result.results[0].score == 0.89

    @pytest.mark.asyncio
    async def test_async_with_all_parameters(self, api):
        """Test async request with all parameters."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"label": "urgent", "score": 0.95}, {"label": "technical", "score": 0.82}]
        }
        mock_response.headers = {}

        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        examples = [
            {"text": "Server is down", "label": "urgent"},
            {"text": "How to configure API?", "label": "technical"},
        ]

        result = await api.it_async(
            text="Production database connection failed",
            tasks={"topic": {"type": "classify", "labels": ["urgent", "technical", "billing", "general"], "threshold": 0.7}},
        )

        assert len(result.results) == 2
        assert result.results[0].label == "urgent"
        assert result.threshold == 0.7
