"""Unit tests for ExtractAPI."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import httpx

from knowledgator import Knowledgator
from knowledgator.models.outputs import (
    ExtractBaseResults,
    ExtractRelationsResults,
    ExtractSummaryResults,
    EntityResponse,
    RelationResponse,
)
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


class TestExtractIt:
    """Tests for extract.it() endpoint (entity extraction)."""

    def test_extract_entities_success(self, sdk, mock_response):
        """Test successful entity extraction."""
        expected_response = {
            "outputs": [
                {
                    "span": "Python",
                    "entity": "Programming Language",
                    "score": 0.95,
                    "start": 0,
                    "end": 6
                },
                {
                    "span": "Google",
                    "entity": "Organization",
                    "score": 0.92,
                    "start": 20,
                    "end": 26
                }
            ],
            "input_text": "Python is developed by Google",
            "labels": ["Programming Language", "Organization"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, expected_response)):
            result = sdk.extract.it(
                text="Python is developed by Google",
                labels=["Programming Language", "Organization"],
                threshold=0.5
            )
        
        assert isinstance(result, ExtractBaseResults)
        assert len(result.results) == 2
        assert result.results[0].span == "Python"
        assert result.results[0].entity == "Programming Language"
        assert result.results[0].score == 0.95
        assert result.input_text == "Python is developed by Google"
        assert result.threshold == 0.5

    def test_extract_with_prompt(self, sdk, mock_response):
        """Test entity extraction with custom prompt."""
        expected_response = {
            "outputs": [
                {
                    "span": "Alice",
                    "entity": "Person",
                    "score": 0.88,
                    "start": 0,
                    "end": 5
                }
            ],
            "input_text": "Alice went to the store",
            "labels": ["Person", "Location"],
            "threshold": 0.7,
            "status": "ok"
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, expected_response)):
            result = sdk.extract.it(
                text="Alice went to the store",
                labels=["Person", "Location"],
                threshold=0.7,
                prompt="Extract named entities"
            )
        
        assert isinstance(result, ExtractBaseResults)
        assert result.results[0].span == "Alice"

    def test_extract_missing_outputs(self, sdk, mock_response):
        """Test response with missing 'outputs' field."""
        malformed_response = {
            "input_text": "Test",
            "labels": ["A", "B"],
            "status": "ok"
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, malformed_response)):
            with pytest.raises(ServerError) as exc_info:
                sdk.extract.it(
                    text="Test",
                    labels=["A", "B"]
                )
            
            assert "missing 'outputs'" in str(exc_info.value)

    def test_extract_invalid_entity_format(self, sdk, mock_response):
        """Test response with invalid entity structure."""
        malformed_response = {
            "outputs": [
                {
                    "span": "Test",
                    # Missing required fields
                }
            ],
            "input_text": "Test",
            "labels": ["Label"],
            "status": "ok"
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, malformed_response)):
            with pytest.raises(ServerError) as exc_info:
                sdk.extract.it(
                    text="Test",
                    labels=["Label"]
                )
            
            assert "Unexpected entity record format" in str(exc_info.value)

    def test_extract_uses_fallback_values(self, sdk, mock_response):
        """Test that fallback values are used when fields are missing."""
        response_without_optional_fields = {
            "outputs": [
                {
                    "span": "Test",
                    "entity": "Label",
                    "score": 0.8,
                    "start": 0,
                    "end": 4
                }
            ]
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, response_without_optional_fields)):
            result = sdk.extract.it(
                text="Fallback text",
                labels=["Label"],
                threshold=0.6
            )
        
        assert result.input_text == "Fallback text"
        assert result.labels == ["Label"]
        assert result.threshold == 0.6

    def test_extract_unauthorized(self, sdk, mock_response):
        """Test handling of 401 unauthorized error."""
        error_response = {"detail": "Invalid API key"}
        
        with patch.object(sdk.extract, "send", return_value=mock_response(401, error_response)):
            with pytest.raises(ClientError):
                sdk.extract.it(
                    text="Test",
                    labels=["A", "B"]
                )


class TestExtractRelations:
    """Tests for extract.relations() endpoint."""

    def test_extract_relations_success(self, sdk, mock_response):
        """Test successful relation extraction."""
        expected_response = {
            "outputs": [
                {
                    "source": {
                        "span": "Alice",
                        "entity": "Person",
                        "score": 0.9,
                        "start": 0,
                        "end": 5
                    },
                    "relation": "works_for",
                    "target": {
                        "span": "Google",
                        "entity": "Organization",
                        "score": 0.88,
                        "start": 17,
                        "end": 23
                    },
                    "score": 0.91
                }
            ],
            "input_text": "Alice works for Google",
            "entities": ["Person", "Organization"],
            "relations": ["works_for", "manages"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, expected_response)):
            result = sdk.extract.relations(
                text="Alice works for Google",
                entities=["Person", "Organization"],
                relations=["works_for", "manages"],
                threshold=0.5
            )
        
        assert isinstance(result, ExtractRelationsResults)
        assert len(result.results) == 1
        assert result.results[0].source.span == "Alice"
        assert result.results[0].relation == "works_for"
        assert result.results[0].target.span == "Google"
        assert result.results[0].score == 0.91

    def test_extract_relations_with_rules(self, sdk, mock_response):
        """Test relation extraction with custom rules."""
        expected_response = {
            "outputs": [
                {
                    "source": {
                        "span": "John",
                        "entity": "Person",
                        "score": 0.92,
                        "start": 0,
                        "end": 4
                    },
                    "relation": "founded",
                    "target": {
                        "span": "Company X",
                        "entity": "Organization",
                        "score": 0.87,
                        "start": 13,
                        "end": 22
                    },
                    "score": 0.89
                }
            ],
            "input_text": "John founded Company X",
            "entities": ["Person", "Organization"],
            "relations": ["founded"],
            "threshold": 0.7,
            "status": "ok"
        }
        
        rules = [
            {"type": "founded", "constraints": {"subject": "Person", "object": "Organization"}}
        ]
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, expected_response)):
            result = sdk.extract.relations(
                text="John founded Company X",
                entities=["Person", "Organization"],
                relations=["founded"],
                threshold=0.7,
                rules=rules
            )
        
        assert isinstance(result, ExtractRelationsResults)
        assert result.results[0].relation == "founded"

    def test_extract_relations_missing_outputs(self, sdk, mock_response):
        """Test response with missing 'outputs' field."""
        malformed_response = {
            "input_text": "Test",
            "entities": ["A"],
            "relations": ["B"],
            "status": "ok"
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, malformed_response)):
            with pytest.raises(ServerError) as exc_info:
                sdk.extract.relations(
                    text="Test",
                    entities=["A"],
                    relations=["B"]
                )
            
            assert "missing 'outputs'" in str(exc_info.value)

    def test_extract_relations_invalid_format(self, sdk, mock_response):
        """Test response with invalid relation structure."""
        malformed_response = {
            "outputs": [
                {
                    "relation": "Test",
                    # Missing required fields
                }
            ],
            "input_text": "Test",
            "entities": ["A"],
            "relations": ["B"],
            "status": "ok"
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, malformed_response)):
            with pytest.raises(ServerError) as exc_info:
                sdk.extract.relations(
                    text="Test",
                    entities=["A"],
                    relations=["B"]
                )
            
            assert "Unexpected relation record format" in str(exc_info.value)


class TestExtractSummary:
    """Tests for extract.summary() endpoint."""

    def test_extract_summary_success(self, sdk, mock_response):
        """Test successful text summarization."""
        expected_response = {
            "summary": "Python is a popular programming language.",
            "input_text": "Python is a high-level, interpreted programming language...",
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, expected_response)):
            result = sdk.extract.summary(
                text="Python is a high-level, interpreted programming language...",
                threshold=0.5
            )
        
        assert isinstance(result, ExtractSummaryResults)
        assert result.summary == "Python is a popular programming language."
        assert result.input_text == "Python is a high-level, interpreted programming language..."
        assert result.threshold == 0.5

    def test_extract_summary_with_prompt(self, sdk, mock_response):
        """Test summarization with custom prompt."""
        expected_response = {
            "summary": "Brief overview of the topic.",
            "input_text": "Long text...",
            "threshold": 0.7,
            "status": "ok"
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, expected_response)):
            result = sdk.extract.summary(
                text="Long text...",
                threshold=0.7,
                prompt="Provide a brief summary"
            )
        
        assert isinstance(result, ExtractSummaryResults)
        assert result.summary == "Brief overview of the topic."

    def test_extract_summary_missing_field(self, sdk, mock_response):
        """Test response with missing 'summary' field."""
        malformed_response = {
            "input_text": "Test",
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, malformed_response)):
            with pytest.raises(ServerError) as exc_info:
                sdk.extract.summary(text="Test")
            
            assert "missing 'summary'" in str(exc_info.value)

    def test_extract_summary_uses_fallback(self, sdk, mock_response):
        """Test that fallback values are used."""
        response_without_optional_fields = {
            "summary": "Summary text"
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, response_without_optional_fields)):
            result = sdk.extract.summary(
                text="Fallback text",
                threshold=0.6
            )
        
        assert result.summary == "Summary text"
        assert result.input_text == "Fallback text"
        assert result.threshold == 0.6


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_timeout_error(self, sdk):
        """Test handling of request timeout."""
        with patch.object(sdk.extract, "send", side_effect=httpx.TimeoutException("Timeout")):
            with pytest.raises(TransportTimeout) as exc_info:
                sdk.extract.it(
                    text="Test",
                    labels=["A", "B"]
                )
            
            assert "timed-out" in str(exc_info.value)

    def test_connection_error(self, sdk):
        """Test handling of connection failure."""
        with patch.object(sdk.extract, "send", side_effect=httpx.ConnectError("Connection failed")):
            with pytest.raises(ConnectionFailure) as exc_info:
                sdk.extract.it(
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
        
        with patch.object(sdk.extract, "send", return_value=response):
            with pytest.raises(ServerError) as exc_info:
                sdk.extract.it(
                    text="Test",
                    labels=["A", "B"]
                )
            
            assert "Malformed or unexpected response" in str(exc_info.value)

    def test_server_error(self, sdk, mock_response):
        """Test handling of 500 server error."""
        error_response = {"detail": "Internal server error"}
        
        with patch.object(sdk.extract, "send", return_value=mock_response(500, error_response)):
            with pytest.raises(ServerError):
                sdk.extract.it(
                    text="Test",
                    labels=["A", "B"]
                )


class TestAsyncMethods:
    """Tests for async API methods."""

    @pytest.mark.asyncio
    async def test_extract_it_async(self, sdk):
        """Test async entity extraction."""
        expected_response = {
            "outputs": [
                {
                    "span": "Python",
                    "entity": "Language",
                    "score": 0.9,
                    "start": 0,
                    "end": 6
                }
            ],
            "input_text": "Python programming",
            "labels": ["Language"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        mock_resp = Mock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.json.return_value = expected_response
        mock_resp.text = str(expected_response)
        mock_resp.headers = {"x-request-id": "test-123"}
        
        async_mock = AsyncMock(return_value=mock_resp)
        
        with patch.object(sdk.extract, "send", async_mock):
            result = await sdk.extract.it_async(
                text="Python programming",
                labels=["Language"],
                threshold=0.5
            )
        
        assert isinstance(result, ExtractBaseResults)
        assert result.results[0].span == "Python"

    @pytest.mark.asyncio
    async def test_extract_relations_async(self, sdk):
        """Test async relation extraction."""
        expected_response = {
            "outputs": [
                {
                    "source": {
                        "span": "A",
                        "entity": "Entity",
                        "score": 0.9,
                        "start": 0,
                        "end": 1
                    },
                    "relation": "relates_to",
                    "target": {
                        "span": "B",
                        "entity": "Entity",
                        "score": 0.88,
                        "start": 13,
                        "end": 14
                    },
                    "score": 0.85
                }
            ],
            "input_text": "A relates to B",
            "entities": ["Entity"],
            "relations": ["relates_to"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        mock_resp = Mock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.json.return_value = expected_response
        mock_resp.text = str(expected_response)
        mock_resp.headers = {"x-request-id": "test-123"}
        
        async_mock = AsyncMock(return_value=mock_resp)
        
        with patch.object(sdk.extract, "send", async_mock):
            result = await sdk.extract.relations_async(
                text="A relates to B",
                entities=["Entity"],
                relations=["relates_to"],
                threshold=0.5
            )
        
        assert isinstance(result, ExtractRelationsResults)
        assert result.results[0].source.span == "A"

    @pytest.mark.asyncio
    async def test_extract_summary_async(self, sdk):
        """Test async text summarization."""
        expected_response = {
            "summary": "Summary of text",
            "input_text": "Long text",
            "threshold": 0.5,
            "status": "ok"
        }
        
        mock_resp = Mock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.is_success = True
        mock_resp.json.return_value = expected_response
        mock_resp.text = str(expected_response)
        mock_resp.headers = {"x-request-id": "test-123"}
        
        async_mock = AsyncMock(return_value=mock_resp)
        
        with patch.object(sdk.extract, "send", async_mock):
            result = await sdk.extract.summary_async(
                text="Long text",
                threshold=0.5
            )
        
        assert isinstance(result, ExtractSummaryResults)
        assert result.summary == "Summary of text"

    @pytest.mark.asyncio
    async def test_extract_async_error(self, sdk):
        """Test async method with error."""
        error_response = {"detail": "Invalid request"}
        mock_resp = Mock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.is_success = False
        mock_resp.json.return_value = error_response
        mock_resp.text = str(error_response)
        mock_resp.headers = {"x-request-id": "test-123"}
        
        # Create an async mock that returns the response
        async_mock = AsyncMock(return_value=mock_resp)
        
        with patch.object(sdk.extract, "send", async_mock):
            with pytest.raises(ClientError):
                await sdk.extract.it_async(
                    text="Test",
                    labels=["A", "B"]
                )

    @pytest.mark.asyncio
    async def test_extract_async_timeout(self, sdk):
        """Test async timeout handling."""
        with patch.object(sdk.extract, "send", side_effect=httpx.TimeoutException("Timeout")):
            with pytest.raises(TransportTimeout):
                await sdk.extract.it_async(
                    text="Test",
                    labels=["A", "B"]
                )


class TestCustomConfiguration:
    """Tests for custom configuration options."""

    def test_custom_server_url(self, sdk, mock_response):
        """Test using custom server URL."""
        expected_response = {
            "outputs": [],
            "input_text": "Test",
            "labels": ["A"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, expected_response)):
            result = sdk.extract.it(
                text="Test",
                labels=["A"],
                server_url="https://custom.api.com"
            )
        
        assert isinstance(result, ExtractBaseResults)

    def test_custom_timeout(self, sdk, mock_response):
        """Test custom timeout configuration."""
        expected_response = {
            "outputs": [],
            "input_text": "Test",
            "labels": ["A"],
            "threshold": 0.5,
            "status": "ok"
        }
        
        with patch.object(sdk.extract, "send", return_value=mock_response(200, expected_response)):
            result = sdk.extract.it(
                text="Test",
                labels=["A"],
                timeout_ms=30000
            )
        
        assert isinstance(result, ExtractBaseResults)


def test_extract_calls_training_endpoint(sdk, mock_response):
    """Ensure extract.it sends requests to the /trainings/ endpoint."""
    expected_response = {"outputs": [{"span": "X", "entity": "Y", "score": 0.9, "start": 0, "end": 1}], "input_text": "Test"}

    def _send_and_check(request):
        url = getattr(request, "url", None)
        path = str(url)
        assert "/trainings/" in path
        return mock_response(200, expected_response)

    with patch.object(sdk.extract, "send", side_effect=_send_and_check):
        result = sdk.extract.it(text="Test", labels=["Y"]) 
    assert isinstance(result, ExtractBaseResults)


@pytest.mark.asyncio
async def test_extract_async_calls_training_endpoint(sdk, mock_response):
    """Ensure extract.it_async sends requests to the /trainings/ endpoint."""
    expected_response = {"outputs": [{"span": "X", "entity": "Y", "score": 0.9, "start": 0, "end": 1}], "input_text": "Test"}

    async def _send_async_and_check(request):
        url = getattr(request, "url", None)
        path = str(url)
        assert "/trainings/" in path
        return mock_response(200, expected_response)

    with patch.object(sdk.extract, "send", side_effect=_send_async_and_check):
        result = await sdk.extract.it_async(text="Test", labels=["Y"]) 
    assert isinstance(result, ExtractBaseResults)
