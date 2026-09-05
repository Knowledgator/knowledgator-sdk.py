from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from knowledgator.configs import SDKConfigs, ServersConfigs
from knowledgator.exceptions import (
    ServerError,
    TransportTimeout,
    ConnectionFailure,
)
from knowledgator.apis.extract import ExtractAPI
from knowledgator.models.inputs import ExtractBaseRequest, ExtractSummaryRequest, ExtractRelationsRequest
from knowledgator.models.outputs import (
    EntityResponse,
    RelationResponse,
    ExtractBaseResults,
    ExtractSummaryResults,
    ExtractRelationsResults,
)


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
def extract_api(mock_configs):
    """Create ExtractAPI instance with mocked configs."""
    return ExtractAPI(mock_configs)


class TestExtractBaseRequest:
    """Tests for ExtractBaseRequest model."""

    def test_minimal_request(self):
        req = ExtractBaseRequest(
            text="Apple Inc. announced new products in California.", labels=["person", "organization", "location"]
        )
        assert req.text == "Apple Inc. announced new products in California."
        assert req.labels == ["person", "organization", "location"]
        assert req.threshold == 0.5  # default
        assert req.prompt is None

    def test_request_with_threshold(self):
        req = ExtractBaseRequest(text="Sample text", labels=["entity1", "entity2"], threshold=0.7)
        assert req.threshold == 0.7

    def test_request_with_prompt(self):
        req = ExtractBaseRequest(
            text="Extract entities from this text", labels=["person", "location"], prompt="Focus on named entities"
        )
        assert req.prompt == "Focus on named entities"

    def test_threshold_validation_min(self):
        """Test that threshold < 0 raises validation error."""
        with pytest.raises(ValidationError):
            ExtractBaseRequest(text="Test", labels=["a", "b"], threshold=-0.1)

    def test_threshold_validation_max(self):
        """Test that threshold > 1 raises validation error."""
        with pytest.raises(ValidationError):
            ExtractBaseRequest(text="Test", labels=["a", "b"], threshold=1.1)


class TestExtractRelationsRequest:
    """Tests for ExtractRelationsRequest model."""

    def test_minimal_request(self):
        req = ExtractRelationsRequest(
            text="Steve Jobs founded Apple Inc.",
            entities=["person", "organization"],
            relations=["founded_by", "works_for"],
        )
        assert req.text == "Steve Jobs founded Apple Inc."
        assert req.entities == ["person", "organization"]
        assert req.relations == ["founded_by", "works_for"]
        assert req.threshold == 0.5
        assert req.rules is None
        assert req.prompt is None

    def test_request_with_rules(self):
        rules = [{"relation": "founded_by", "pairs_filter": [["person", "organization"]], "distance": 200}]
        req = ExtractRelationsRequest(
            text="Test text", entities=["person", "organization"], relations=["founded_by"], rules=rules
        )
        assert len(req.rules) == 1
        assert req.rules[0]["relation"] == "founded_by"


class TestExtractSummaryRequest:
    """Tests for ExtractSummaryRequest model."""

    def test_minimal_request(self):
        req = ExtractSummaryRequest(text="Long document to summarize")
        assert req.text == "Long document to summarize"
        assert req.threshold == 0.5
        assert req.prompt is None

    def test_request_with_prompt(self):
        req = ExtractSummaryRequest(text="Document", prompt="Extract key findings", threshold=0.7)
        assert req.prompt == "Extract key findings"
        assert req.threshold == 0.7


class TestEntityResponse:
    """Tests for EntityResponse model."""

    def test_entity_creation(self):
        entity = EntityResponse(span="Apple Inc.", entity="organization", start=0, end=10, score=0.95)
        assert entity.span == "Apple Inc."
        assert entity.entity == "organization"
        assert entity.start == 0
        assert entity.end == 10
        assert entity.score == 0.95

    def test_entity_from_dict(self):
        data = {"span": "California", "entity": "location", "start": 20, "end": 30, "score": 0.88}
        entity = EntityResponse.model_validate(data)
        assert entity.span == "California"
        assert entity.entity == "location"


class TestRelationResponse:
    """Tests for RelationResponse model."""

    def test_relation_creation(self):
        source = EntityResponse(span="Steve Jobs", entity="person", start=0, end=10, score=0.95)
        target = EntityResponse(span="Apple Inc.", entity="organization", start=20, end=30, score=0.92)
        relation = RelationResponse(source=source, relation="founded_by", target=target, score=0.89)
        assert relation.source.span == "Steve Jobs"
        assert relation.relation == "founded_by"
        assert relation.target.span == "Apple Inc."
        assert relation.score == 0.89


class TestExtractAPISync:
    """Tests for synchronous ExtractAPI.it() method."""

    def test_basic_entity_extraction(self, extract_api):
        """Test basic entity extraction without optional parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {"span": "Apple Inc.", "entity": "organization", "start": 0, "end": 10, "score": 0.95},
                {"span": "California", "entity": "location", "start": 35, "end": 45, "score": 0.88},
            ]
        }
        mock_response.headers = {}
        extract_api.configs.client.request.return_value = mock_response

        result = extract_api.it(
            text="Apple Inc. announced new products in California.", labels=["person", "organization", "location"]
        )

        assert isinstance(result, ExtractBaseResults)
        assert len(result.results) == 2
        assert result.results[0].span == "Apple Inc."
        assert result.results[0].entity == "organization"
        assert result.results[1].span == "California"
        assert result.results[1].entity == "location"

    def test_extraction_with_threshold(self, extract_api):
        """Test entity extraction with custom threshold."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"span": "Google", "entity": "organization", "start": 0, "end": 6, "score": 0.92}]
        }
        mock_response.headers = {}
        extract_api.configs.client.request.return_value = mock_response

        result = extract_api.it(
            text="Google develops AI technology", labels=["organization", "technology"], threshold=0.8
        )

        assert len(result.results) == 1
        assert result.results[0].entity == "organization"
        assert result.threshold == 0.8

    def test_extraction_with_prompt(self, extract_api):
        """Test entity extraction with prompt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"span": "John Smith", "entity": "person", "start": 0, "end": 10, "score": 0.94}]
        }
        mock_response.headers = {}
        extract_api.configs.client.request.return_value = mock_response

        result = extract_api.it(
            text="John Smith works at Microsoft",
            labels=["person", "organization"],
            prompt="Extract all people and organizations",
        )

        assert result.results[0].span == "John Smith"

    def test_no_entities_above_threshold(self, extract_api):
        """Test when no entities meet the threshold."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": []}
        mock_response.headers = {}
        extract_api.configs.client.request.return_value = mock_response

        result = extract_api.it(text="Ambiguous text", labels=["entity1", "entity2"], threshold=0.9)

        assert len(result.results) == 0

    def test_timeout_exception(self, extract_api):
        """Test that timeout raises TransportTimeout."""
        extract_api.configs.client.request.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(TransportTimeout):
            extract_api.it(text="Test text", labels=["entity1", "entity2"])

    def test_connection_error(self, extract_api):
        """Test that connection errors raise ConnectionFailure."""
        extract_api.configs.client.request.side_effect = httpx.RequestError("connection failed")

        with pytest.raises(ConnectionFailure):
            extract_api.it(text="Test text", labels=["entity1", "entity2"])

    def test_server_error_response(self, extract_api):
        """Test handling of server error responses."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.return_value = {"error": "Server error"}
        mock_response.headers = {"x-request-id": "req-123"}
        extract_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError):
            extract_api.it(text="Test text", labels=["entity1", "entity2"])

    def test_missing_outputs_in_response(self, extract_api):
        """Test that missing 'outputs' in response raises ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "something"}  # No 'outputs' key
        mock_response.headers = {"x-request-id": "req-456"}
        extract_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            extract_api.it(text="Test text", labels=["entity1", "entity2"])
        assert "missing 'outputs'" in str(exc_info.value)

    def test_malformed_entity_in_outputs(self, extract_api):
        """Test that malformed entity items raise ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {"span": "test"}  # Missing required fields
            ]
        }
        mock_response.headers = {"x-request-id": "req-999"}
        extract_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            extract_api.it(text="Test text", labels=["entity1", "entity2"])
        assert "Unexpected entity record format" in str(exc_info.value)


class TestExtractAPIRelations:
    """Tests for ExtractAPI.relations() method."""

    def test_basic_relation_extraction(self, extract_api):
        """Test basic relation extraction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "source": {"span": "Steve Jobs", "entity": "person", "start": 0, "end": 10, "score": 0.95},
                    "relation": "founded_by",
                    "target": {"span": "Apple Inc.", "entity": "organization", "start": 20, "end": 30, "score": 0.92},
                    "score": 0.89,
                }
            ]
        }
        mock_response.headers = {}
        extract_api.configs.client.request.return_value = mock_response

        result = extract_api.relations(
            text="Steve Jobs founded Apple Inc.",
            entities=["person", "organization"],
            relations=["founded_by", "works_for"],
        )

        assert isinstance(result, ExtractRelationsResults)
        assert len(result.results) == 1
        assert result.results[0].relation == "founded_by"
        assert result.results[0].source.span == "Steve Jobs"
        assert result.results[0].target.span == "Apple Inc."

    def test_relation_extraction_with_rules(self, extract_api):
        """Test relation extraction with filtering rules."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "source": {"span": "John", "entity": "person", "start": 0, "end": 4, "score": 0.93},
                    "relation": "works_for",
                    "target": {"span": "Microsoft", "entity": "organization", "start": 15, "end": 24, "score": 0.91},
                    "score": 0.87,
                }
            ]
        }
        mock_response.headers = {}
        extract_api.configs.client.request.return_value = mock_response

        rules = [{"relation": "works_for", "pairs_filter": [["person", "organization"]], "distance": 200}]

        result = extract_api.relations(
            text="John works for Microsoft", entities=["person", "organization"], relations=["works_for"], rules=rules
        )

        assert len(result.results) == 1
        assert result.results[0].relation == "works_for"


class TestExtractAPISummary:
    """Tests for ExtractAPI.summary() method."""

    def test_basic_summarization(self, extract_api):
        """Test basic summarization."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"summary": "This is a summary of the document."}
        mock_response.headers = {}
        extract_api.configs.client.request.return_value = mock_response

        result = extract_api.summary(text="Long document with lots of text that needs to be summarized.")

        assert isinstance(result, ExtractSummaryResults)
        assert result.summary == "This is a summary of the document."

    def test_summarization_with_prompt(self, extract_api):
        """Test summarization with custom prompt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"summary": "Key findings: AI is transforming industries."}
        mock_response.headers = {}
        extract_api.configs.client.request.return_value = mock_response

        result = extract_api.summary(text="Research paper about AI", prompt="Extract key findings")

        assert "Key findings" in result.summary

    def test_summary_missing_field_error(self, extract_api):
        """Test that missing 'summary' field raises ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "something"}
        mock_response.headers = {}
        extract_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            extract_api.summary(text="Test text")
        assert "missing 'summary'" in str(exc_info.value)


class TestExtractAPIAsync:
    """Tests for asynchronous ExtractAPI methods."""

    @pytest.mark.asyncio
    async def test_basic_extraction_async(self, extract_api):
        """Test basic async entity extraction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"span": "Google", "entity": "organization", "start": 0, "end": 6, "score": 0.94}]
        }
        mock_response.headers = {}

        # Mock the send method directly with AsyncMock
        with patch.object(extract_api, "send_async", new=AsyncMock(return_value=mock_response)):
            result = await extract_api.it_async(text="Google develops AI", labels=["organization", "technology"])

            assert result.results[0].span == "Google"
            assert result.results[0].entity == "organization"

    @pytest.mark.asyncio
    async def test_async_timeout(self, extract_api):
        """Test async timeout handling."""
        # Mock the send method to raise timeout
        with (
            patch.object(extract_api, "send_async", new=AsyncMock(side_effect=httpx.TimeoutException("timeout"))),
            pytest.raises(TransportTimeout),
        ):
            await extract_api.it_async(text="Test", labels=["entity1", "entity2"])

    @pytest.mark.asyncio
    async def test_async_relations(self, extract_api):
        """Test async relation extraction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "source": {"span": "Alice", "entity": "person", "start": 0, "end": 5, "score": 0.92},
                    "relation": "works_for",
                    "target": {"span": "IBM", "entity": "organization", "start": 16, "end": 19, "score": 0.90},
                    "score": 0.85,
                }
            ]
        }
        mock_response.headers = {}

        # Mock the send method directly
        with patch.object(extract_api, "send_async", new=AsyncMock(return_value=mock_response)):
            result = await extract_api.relations_async(
                text="Alice works for IBM", entities=["person", "organization"], relations=["works_for"]
            )

            assert len(result.results) == 1
            assert result.results[0].relation == "works_for"

    @pytest.mark.asyncio
    async def test_async_summary(self, extract_api):
        """Test async summarization."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"summary": "Brief summary of the content."}
        mock_response.headers = {}

        # Mock the send method directly
        with patch.object(extract_api, "send_async", new=AsyncMock(return_value=mock_response)):
            result = await extract_api.summary_async(text="Long document to summarize")

            assert result.summary == "Brief summary of the content."
