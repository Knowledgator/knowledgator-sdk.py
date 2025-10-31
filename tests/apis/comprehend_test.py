"""Unit tests for ComprehendAPI."""
import pytest
from unittest.mock import Mock, patch
import httpx

from knowledgator import Knowledgator
from knowledgator.models.outputs import ComprehendResults, ComprehendResponse
from knowledgator.errors.legacy import (
    ServerError,
    ClientError,
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
        response.headers = {"x-request-id": "test-request-123"}
        return response
    return _create_response


class TestComprehendIt:
    """Tests for comprehend.it() endpoint."""

    def test_comprehend_success(self, sdk, mock_response):
        sdk = Knowledgator()
        """Test successful text classification."""
        expected_response = {
            "outputs": [
                {
                    "label": "Technology",
                    "score": 0.95
                },
                {
                    "label": "Science",
                    "score": 0.85
                }
            ],
            "input_text": "Python is a programming language",
            "labels": ["Technology", "Science", "Sports"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.comprehend, "send", return_value=mock_response(200, expected_response)):
            result = sdk.comprehend.it(
                text="Python is a programming language",
                labels=["Technology", "Science", "Sports"],
                threshold=0.5
            )
        
        assert isinstance(result, ComprehendResults)
        assert len(result.results) == 2
        assert result.results[0].label == "Technology"
        assert result.results[0].score == 0.95
        assert result.input_text == "Python is a programming language"
        assert result.threshold == 0.5
        assert result.status == "ok"

    def test_comprehend_with_examples(self, sdk, mock_response):
        """Test classification with examples."""
        expected_response = {
            "outputs": [
                {
                    "label": "Positive",
                    "score": 0.92
                }
            ],
            "input_text": "This is great!",
            "labels": ["Positive", "Negative"],
            "threshold": 0.7,
            "status": "ok"
        }
        
        examples = [
            {"text": "I love this", "label": "Positive"},
            {"text": "I hate this", "label": "Negative"}
        ]
        
        with patch.object(sdk.comprehend, "send", return_value=mock_response(200, expected_response)):
            result = sdk.comprehend.it(
                text="This is great!",
                labels=["Positive", "Negative"],
                threshold=0.7,
                examples=examples
            )
        
        assert isinstance(result, ComprehendResults)
        assert len(result.results) == 1
        assert result.results[0].label == "Positive"

    def test_comprehend_missing_outputs(self, sdk, mock_response):
        """Test response with missing 'outputs' field."""
        malformed_response = {
            "input_text": "Test",
            "labels": ["A", "B"],
            "status": "ok"
        }
        
        with patch.object(sdk.comprehend, "send", return_value=mock_response(200, malformed_response)):
            with pytest.raises(ServerError) as exc_info:
                sdk.comprehend.it(
                    text="Test",
                    labels=["A", "B"]
                )
            
            assert "missing 'outputs'" in str(exc_info.value)

    def test_comprehend_invalid_outputs_format(self, sdk, mock_response):
        """Test response with invalid outputs structure."""
        malformed_response = {
            "outputs": [
                {
                    "label": "Test",
                    # Missing required 'score' field
                }
            ],
            "input_text": "Test",
            "labels": ["Test"],
            "status": "ok"
        }
        
        with patch.object(sdk.comprehend, "send", return_value=mock_response(200, malformed_response)):
            with pytest.raises(ServerError) as exc_info:
                sdk.comprehend.it(
                    text="Test",
                    labels=["Test"]
                )
            
            assert "Unexpected record format" in str(exc_info.value)

    def test_comprehend_uses_fallback_values(self, sdk, mock_response):
        """Test that fallback values are used when fields are missing."""
        response_without_optional_fields = {
            "outputs": [
                {
                    "label": "Tech",
                    "score": 0.8
                }
            ]
            # Missing input_text, labels, threshold, status
        }
        
        with patch.object(sdk.comprehend, "send", return_value=mock_response(200, response_without_optional_fields)):
            result = sdk.comprehend.it(
                text="Fallback text",
                labels=["Tech", "Sports"],
                threshold=0.6
            )
        
        assert result.input_text == "Fallback text"
        assert result.labels == ["Tech", "Sports"]
        assert result.threshold == 0.6
        assert result.status == "ok"  # Default status

    def test_comprehend_default_threshold(self, sdk, mock_response):
        """Test default threshold when not provided."""
        expected_response = {
            "outputs": [
                {"label": "A", "score": 0.9}
            ]
        }
        
        with patch.object(sdk.comprehend, "send", return_value=mock_response(200, expected_response)):
            result = sdk.comprehend.it(
                text="Test",
                labels=["A", "B"]
                # No threshold provided
            )
        
        assert result.threshold == 0.5  # DEFAULT_THRESHOLD

    def test_comprehend_unauthorized(self, sdk, mock_response):
        """Test handling of 401 unauthorized error."""
        error_response = {"detail": "Invalid API key"}
        
        with patch.object(sdk.comprehend, "send", return_value=mock_response(401, error_response)):
            with pytest.raises(ClientError):
                sdk.comprehend.it(
                    text="Test",
                    labels=["A", "B"]
                )

    def test_comprehend_bad_request(self, sdk, mock_response):
        """Test handling of 400 bad request error."""
        error_response = {
            "detail": [
                {
                    "loc": ["body", "labels"],
                    "msg": "field required",
                    "type": "value_error.missing"
                }
            ]
        }
        
        with patch.object(sdk.comprehend, "send", return_value=mock_response(400, error_response)):
            with pytest.raises(ClientError):
                sdk.comprehend.it(
                    text="Test",
                    labels=["A", "B"]
                )


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_timeout_error(self, sdk):
        sdk = Knowledgator()
        """Test handling of request timeout."""
        with patch.object(sdk.comprehend, "send", side_effect=httpx.TimeoutException("Timeout")):
            with pytest.raises(TransportTimeout) as exc_info:
                sdk.comprehend.it(
                    text="Test",
                    labels=["A", "B"]
                )
            
            assert "timed-out" in str(exc_info.value)

    def test_connection_error(self, sdk):
        """Test handling of connection failure."""
        with patch.object(sdk.comprehend, "send", side_effect=httpx.ConnectError("Connection failed")):
            with pytest.raises(ConnectionFailure) as exc_info:
                sdk.comprehend.it(
                    text="Test",
                    labels=["A", "B"]
                )
            
            assert "Transport error" in str(exc_info.value)

    def test_invalid_json_response(self, sdk):
        """Test handling of invalid JSON in response."""
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.is_success = True
        response.json.side_effect = ValueError("Invalid JSON")
        response.text = "Not JSON"
        response.headers = {"x-request-id": "test-123"}
        
        with patch.object(sdk.comprehend, "send", return_value=response):
            with pytest.raises(ServerError) as exc_info:
                sdk.comprehend.it(
                    text="Test",
                    labels=["A", "B"]
                )
            
            assert "not valid JSON" in str(exc_info.value)

    def test_server_error(self, sdk, mock_response):
        """Test handling of 500 server error."""
        error_response = {"detail": "Internal server error"}
        
        with patch.object(sdk.comprehend, "send", return_value=mock_response(500, error_response)):
            with pytest.raises(ServerError):
                sdk.comprehend.it(
                    text="Test",
                    labels=["A", "B"]
                )


class TestAsyncMethods:
    """Tests for async API methods."""

    @pytest.mark.asyncio
    async def test_comprehend_async_success(self, sdk, mock_response):
        sdk = Knowledgator()
        """Test async text classification."""
        expected_response = {
            "outputs": [
                {
                    "label": "Sports",
                    "score": 0.88
                }
            ],
            "input_text": "Football is exciting",
            "labels": ["Sports", "Technology"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.comprehend, "send_async", return_value=mock_response(200, expected_response)):
            result = await sdk.comprehend.it_async(
                text="Football is exciting",
                labels=["Sports", "Technology"],
                threshold=0.5
            )
        
        assert isinstance(result, ComprehendResults)
        assert result.results[0].label == "Sports"
        assert result.results[0].score == 0.88

    @pytest.mark.asyncio
    async def test_comprehend_async_error(self, sdk, mock_response):
        """Test async method with error."""
        error_response = {"detail": "Invalid request"}
        
        with patch.object(sdk.comprehend, "send_async", return_value=mock_response(400, error_response)):
            with pytest.raises(ClientError):
                await sdk.comprehend.it_async(
                    text="Test",
                    labels=["A", "B"]
                )

    @pytest.mark.asyncio
    async def test_comprehend_async_timeout(self, sdk):
        """Test async timeout handling."""
        with patch.object(sdk.comprehend, "send_async", side_effect=httpx.TimeoutException("Timeout")):
            with pytest.raises(TransportTimeout):
                await sdk.comprehend.it_async(
                    text="Test",
                    labels=["A", "B"]
                )


class TestCustomConfiguration:
    """Tests for custom configuration options."""
    
    def test_custom_server_url(self, sdk, mock_response):
        sdk = Knowledgator()
        """Test using custom server URL."""
        expected_response = {
            "outputs": [{"label": "A", "score": 0.9}],
            "input_text": "Test",
            "labels": ["A", "B"],
            "threshold": 0.5,
            "status": "ok"
        }
        with patch.object(sdk.comprehend, "send", return_value=mock_response(200, expected_response)):
            result = sdk.comprehend.it(
                text="Test",
                labels=["A", "B"],
                server_url="https://custom.api.com"
            )
        
        assert isinstance(result, ComprehendResults)

    def test_custom_timeout(self, sdk, mock_response):
        """Test custom timeout configuration."""
        expected_response = {
            "outputs": [{"label": "A", "score": 0.9}],
            "input_text": "Test",
            "labels": ["A", "B"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.comprehend, "send", return_value=mock_response(200, expected_response)):
            result = sdk.comprehend.it(
                text="Test",
                labels=["A", "B"],
                timeout_ms=30000
            )
        
        assert isinstance(result, ComprehendResults)

    def test_custom_headers(self, sdk, mock_response):
        """Test custom HTTP headers."""
        expected_response = {
            "outputs": [{"label": "A", "score": 0.9}],
            "input_text": "Test",
            "labels": ["A", "B"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.comprehend, "send", return_value=mock_response(200, expected_response)):
            result = sdk.comprehend.it(
                text="Test",
                labels=["A", "B"],
                http_headers={"X-Custom-Header": "value"}
            )
        
        assert isinstance(result, ComprehendResults)

    def test_comprehend_calls_training_endpoint(self, sdk, mock_response):
        """Ensure comprehend.it sends requests to the /trainings/ endpoint."""
        expected_response = {"outputs": [{"label": "A", "score": 0.9}], "input_text": "Test"}

        def _send_and_check(request):
            # request.url may be a httpx.URL or string — check path contains trainings
            url = getattr(request, "url", None)
            path = str(url)
            assert "/trainings/" in path
            return mock_response(200, expected_response)

        with patch.object(sdk.comprehend, "send", side_effect=_send_and_check):
            result = sdk.comprehend.it(text="Test", labels=["A"]) 
        assert isinstance(result, ComprehendResults)

    @pytest.mark.asyncio
    async def test_comprehend_async_calls_training_endpoint(self, sdk, mock_response):
        """Ensure comprehend.it_async sends requests to the /trainings/ endpoint."""
        expected_response = {"outputs": [{"label": "A", "score": 0.9}], "input_text": "Test"}

        async def _send_async_and_check(request):
            url = getattr(request, "url", None)
            path = str(url)
            assert "/trainings/" in path
            return mock_response(200, expected_response)

        with patch.object(sdk.comprehend, "send_async", side_effect=_send_async_and_check):
            result = await sdk.comprehend.it_async(text="Test", labels=["A"]) 
        assert isinstance(result, ComprehendResults)
