"""Unit tests for ComprehendGatewayAPI."""
import pytest
from unittest.mock import Mock, patch
import httpx

from knowledgator import Knowledgator
from knowledgator.models.outputs import ComprehendResults, ComprehendResponse
from knowledgator.errors import (
    BadRequestError,
    UnauthorizedError,
    InternalServerError,
    ServerError,
    TransportTimeout,
    ConnectionFailure,
)


@pytest.fixture
def mock_http_client(monkeypatch):
    """Mock HttpClient to prevent real network calls."""
    from knowledgator.clients.http_client import HttpClient, AsyncHttpClient
    
    monkeypatch.setattr(HttpClient, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(AsyncHttpClient, "__init__", lambda self, *a, **k: None)


@pytest.fixture
def sdk(mock_http_client):
    """Create SDK instance with mocked HTTP client."""
    return Knowledgator(api_key="test-api-key")


@pytest.fixture
def mock_response():
    """Factory for creating mock httpx.Response objects."""
    def _create_response(status_code: int, json_data: dict):
        response = Mock(spec=httpx.Response)
        response.status_code = status_code
        response.is_success = 200 <= status_code < 300
        response.json.return_value = json_data
        response.text = str(json_data)
        response.headers = {}
        return response
    return _create_response


class TestResolve:
    """Tests for resolve endpoint."""

    def test_resolve_success_with_results(self, sdk, mock_response):
        """Test successful comprehend resolution with 'results' field."""
        expected_response = {
            "results": [
                {"label": "positive", "score": 0.95},
                {"label": "negative", "score": 0.05}
            ],
            "text": "This is great!",
            "labels": ["positive", "negative"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.gateway, "send", return_value=mock_response(200, expected_response)):
            result = sdk.gateway.resolve(
                text="This is great!",
                labels=["positive", "negative"],
                threshold=0.5
            )
        
        assert isinstance(result, ComprehendResults)
        assert len(result.results) == 2
        assert result.results[0].label == "positive"
        assert result.results[0].score == 0.95
        assert result.input_text == "This is great!"
        assert result.labels == ["positive", "negative"]
        assert result.threshold == 0.5
        assert result.status == "ok"

    def test_resolve_success_with_outputs(self, sdk, mock_response):
        """Test successful comprehend resolution with 'outputs' field (alternative)."""
        expected_response = {
            "outputs": [
                {"label": "spam", "score": 0.1},
                {"label": "not_spam", "score": 0.9}
            ],
            "text": "Hello world",
            "labels": ["spam", "not_spam"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.gateway, "send", return_value=mock_response(200, expected_response)):
            result = sdk.gateway.resolve(
                text="Hello world",
                labels=["spam", "not_spam"]
            )
        
        assert isinstance(result, ComprehendResults)
        assert len(result.results) == 2
        assert result.results[1].label == "not_spam"
        assert result.results[1].score == 0.9

    def test_resolve_with_examples(self, sdk, mock_response):
        """Test resolve with few-shot examples."""
        expected_response = {
            "results": [
                {"label": "tech", "score": 0.85}
            ],
            "text": "New smartphone released",
            "labels": ["tech", "sports"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        examples = [
            {"text": "iPhone 15 announced", "label": "tech"},
            {"text": "Lakers win championship", "label": "sports"}
        ]
        
        with patch.object(sdk.gateway, "send", return_value=mock_response(200, expected_response)):
            result = sdk.gateway.resolve(
                text="New smartphone released",
                labels=["tech", "sports"],
                examples=examples
            )
        
        assert isinstance(result, ComprehendResults)
        assert result.results[0].label == "tech"

    def test_resolve_with_custom_service_id(self, sdk, mock_response):
        """Test resolve with custom service_id."""
        expected_response = {
            "results": [{"label": "question", "score": 0.9}],
            "text": "What is AI?",
            "labels": ["question", "statement"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.gateway, "send", return_value=mock_response(200, expected_response)):
            result = sdk.gateway.resolve(
                text="What is AI?",
                labels=["question", "statement"],
                service_id="instruct-it"
            )
        
        assert isinstance(result, ComprehendResults)
        assert result.results[0].label == "question"

    def test_resolve_missing_results_field(self, sdk, mock_response):
        """Test resolve with response missing both 'results' and 'outputs'."""
        malformed_response = {
            "text": "Test",
            "labels": ["a", "b"],
            # Missing 'results' or 'outputs'
        }
        
        with patch.object(sdk.gateway, "send", return_value=mock_response(200, malformed_response)):
            with pytest.raises(ServerError) as exc_info:
                sdk.gateway.resolve(
                    text="Test",
                    labels=["a", "b"]
                )
            
            assert "missing 'results' or 'outputs'" in str(exc_info.value)

    def test_resolve_invalid_result_format(self, sdk, mock_response):
        """Test resolve with invalid result object format."""
        malformed_response = {
            "results": [
                {"label": "test"}  # Missing 'score'
            ],
            "text": "Test",
            "labels": ["test"]
        }
        
        with patch.object(sdk.gateway, "send", return_value=mock_response(200, malformed_response)):
            with pytest.raises(ServerError) as exc_info:
                sdk.gateway.resolve(
                    text="Test",
                    labels=["test"]
                )
            
            assert "Unexpected record format" in str(exc_info.value)

    def test_resolve_unauthorized(self, sdk, mock_response):
        """Test resolve with invalid API key."""
        error_response = {"detail": "Invalid API key"}
        
        with patch.object(sdk.gateway, "send", return_value=mock_response(401, error_response)):
            with pytest.raises(UnauthorizedError):
                sdk.gateway.resolve(
                    text="Test",
                    labels=["a", "b"]
                )


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_json_response(self, sdk):
        """Test handling of invalid JSON in response."""
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.is_success = True
        response.json.side_effect = ValueError("Invalid JSON")
        response.text = "Not JSON"
        response.headers = {}
        
        with patch.object(sdk.gateway, "send", return_value=response):
            with pytest.raises(InternalServerError) as exc_info:
                sdk.gateway.resolve(
                    text="Test",
                    labels=["a", "b"]
                )
            
            assert "not valid JSON" in str(exc_info.value)

    def test_timeout_error(self, sdk):
        """Test handling of timeout."""
        with patch.object(sdk.gateway, "send", side_effect=httpx.TimeoutException("Timeout")):
            with pytest.raises(TransportTimeout) as exc_info:
                sdk.gateway.resolve(
                    text="Test",
                    labels=["a", "b"]
                )
            
            assert "timed-out" in str(exc_info.value)

    def test_connection_error(self, sdk):
        """Test handling of connection failure."""
        with patch.object(sdk.gateway, "send", side_effect=httpx.RequestError("Connection failed")):
            with pytest.raises(ConnectionFailure) as exc_info:
                sdk.gateway.resolve(
                    text="Test",
                    labels=["a", "b"]
                )
            
            assert "Transport error" in str(exc_info.value)

    def test_fallback_values_used(self, sdk, mock_response):
        """Test that fallback values are used when fields missing in response."""
        minimal_response = {
            "results": [
                {"label": "test", "score": 0.9}
            ]
            # Missing text, labels, threshold, status
        }
        
        input_text = "Original text"
        input_labels = ["test", "other"]
        input_threshold = 0.7
        
        with patch.object(sdk.gateway, "send", return_value=mock_response(200, minimal_response)):
            result = sdk.gateway.resolve(
                text=input_text,
                labels=input_labels,
                threshold=input_threshold
            )
        
        # Should use fallback values from input
        assert result.input_text == input_text
        assert result.labels == input_labels
        assert result.threshold == input_threshold
        assert result.status == "ok"  # Default


class TestAsyncMethods:
    """Tests for async API methods."""

    @pytest.mark.asyncio
    async def test_resolve_async(self, sdk, mock_response):
        """Test async comprehend resolution."""
        expected_response = {
            "results": [
                {"label": "positive", "score": 0.9},
                {"label": "negative", "score": 0.1}
            ],
            "text": "Async test",
            "labels": ["positive", "negative"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.gateway, "send_async", return_value=mock_response(200, expected_response)):
            result = await sdk.gateway.resolve_async(
                text="Async test",
                labels=["positive", "negative"]
            )
        
        assert isinstance(result, ComprehendResults)
        assert len(result.results) == 2
        assert result.results[0].label == "positive"

    @pytest.mark.asyncio
    async def test_resolve_async_error(self, sdk, mock_response):
        """Test async resolve with error."""
        error_response = {"detail": "Bad request"}
        
        with patch.object(sdk.gateway, "send_async", return_value=mock_response(400, error_response)):
            with pytest.raises(BadRequestError):
                await sdk.gateway.resolve_async(
                    text="Test",
                    labels=["a", "b"]
                )


class TestGatewayVerbWrappers:
    """Ensure convenience wrappers for common HTTP verbs call through to the
    generic gateway implementation and return parsed results (sync + async)."""

    def test_gateway_get_wrapper(self, sdk, mock_response):
        expected_response = {
            "results": [{"label": "positive", "score": 0.95}],
            "text": "GET test",
            "labels": ["positive", "negative"],
            "threshold": 0.5,
            "status": "ok",
        }

        with patch.object(sdk.gateway, "send", return_value=mock_response(200, expected_response)):
            result = sdk.gateway.gateway_get(
                text="GET test",
                labels=["positive", "negative"],
                service_id="comprehend-it",
            )

        assert isinstance(result, ComprehendResults)
        assert result.results[0].label == "positive"

    def test_gateway_post_wrapper(self, sdk, mock_response):
        expected_response = {
            "results": [{"label": "ok", "score": 0.9}],
            "text": "POST test",
            "labels": ["ok"],
        }

        with patch.object(sdk.gateway, "send", return_value=mock_response(200, expected_response)):
            result = sdk.gateway.gateway_post(
                text="POST test",
                labels=["ok"],
                service_id="comprehend-it",
            )

        assert isinstance(result, ComprehendResults)
        assert result.input_text == "POST test"

    def test_gateway_put_patch_delete_wrappers(self, sdk, mock_response):
        expected_response = {
            "results": [{"label": "update", "score": 0.77}],
            "text": "UPDATE test",
            "labels": ["update"],
        }

        with patch.object(sdk.gateway, "send", return_value=mock_response(200, expected_response)):
            put_result = sdk.gateway.gateway_put(text="UPDATE test", labels=["update"])
            patch_result = sdk.gateway.gateway_patch(text="UPDATE test", labels=["update"])
            delete_result = sdk.gateway.gateway_delete(text="UPDATE test", labels=["update"])

        assert isinstance(put_result, ComprehendResults)
        assert isinstance(patch_result, ComprehendResults)
        assert isinstance(delete_result, ComprehendResults)

    @pytest.mark.asyncio
    async def test_gateway_async_wrappers(self, sdk, mock_response):
        expected_response = {
            "results": [{"label": "async", "score": 0.66}],
            "text": "Async test",
            "labels": ["async"],
        }

        with patch.object(sdk.gateway, "send_async", return_value=mock_response(200, expected_response)):
            get_res = await sdk.gateway.gateway_get_async(text="Async test", labels=["async"])
            post_res = await sdk.gateway.gateway_post_async(text="Async test", labels=["async"])
            put_res = await sdk.gateway.gateway_put_async(text="Async test", labels=["async"])
            patch_res = await sdk.gateway.gateway_patch_async(text="Async test", labels=["async"])
            delete_res = await sdk.gateway.gateway_delete_async(text="Async test", labels=["async"])

        assert isinstance(get_res, ComprehendResults)
        assert isinstance(post_res, ComprehendResults)
        assert isinstance(put_res, ComprehendResults)
        assert isinstance(patch_res, ComprehendResults)
        assert isinstance(delete_res, ComprehendResults)