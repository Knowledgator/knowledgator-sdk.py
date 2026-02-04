"""
Mock API Request Tests for Extract-it Endpoints

This module demonstrates the Extract-it API with mock HTTP requests,
verifying the complete API contract including:
- General entity extraction endpoint
- Relation extraction endpoint
- Summarization endpoint
- Request/response structure
- Threshold and prompt parameter handling
- Error scenarios
"""

from unittest.mock import MagicMock, patch

import pytest

from knowledgator.configs import SDKConfigs, ServersConfigs
from knowledgator.exceptions import ServerError
from knowledgator.apis.extract import ExtractAPI


class TestExtractGeneralAPIMockRequests:
    """Test Extract API general endpoint with mock HTTP requests matching the API contract."""

    @pytest.fixture
    def api(self):
        """Create ExtractAPI instance with mock configuration."""
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return ExtractAPI(configs)

    def test_minimal_request_structure(self, api):
        """Test minimal request with only required 'text' and 'entities' fields."""
        # Mock response matching API contract
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {"entity": "organization", "span": "Apple Inc.", "start": 0, "end": 10, "score": 0.95},
                {"entity": "location", "span": "California", "start": 35, "end": 45, "score": 0.88},
            ]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request
        result = api.it(
            text="Apple Inc. announced new products in California.", labels=["person", "organization", "location"]
        )

        # Verify request was made
        assert api.configs.client.request.called

        # Verify the request was called with json parameter
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "json" in call_kwargs
        assert call_kwargs["json"]["text"] == "Apple Inc. announced new products in California."
        assert call_kwargs["json"]["labels"] == ["person", "organization", "location"]
        # threshold should use default if not provided
        assert "threshold" not in call_kwargs["json"] or call_kwargs["json"]["threshold"] == 0.5
        assert "prompt" not in call_kwargs["json"]

        # Verify response
        assert len(result.results) == 2
        assert result.results[0].entity == "organization"
        assert result.results[0].span == "Apple Inc."
        assert result.results[0].start == 0
        assert result.results[0].end == 10
        assert result.results[0].score == 0.95

    def test_request_with_threshold(self, api):
        """Test request with custom threshold parameter."""
        # Mock response with filtered results
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"entity": "organization", "span": "Google", "start": 0, "end": 6, "score": 0.92}]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request with threshold
        result = api.it(
            text="Google develops AI technology", labels=["organization", "technology", "person"], threshold=0.7
        )

        # Verify request body includes threshold
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "json" in call_kwargs
        assert call_kwargs["json"]["threshold"] == 0.7

        # Verify response
        assert len(result.results) == 1
        assert result.results[0].entity == "organization"
        assert result.results[0].score >= 0.7

    def test_request_with_prompt(self, api):
        """Test request with prompt parameter."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {"entity": "person", "span": "John Smith", "start": 0, "end": 10, "score": 0.94},
                {"entity": "organization", "span": "Microsoft", "start": 20, "end": 29, "score": 0.91},
            ]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request with prompt
        result = api.it(
            text="John Smith works at Microsoft",
            labels=["person", "organization"],
            prompt="Extract all people and organizations involved in employment relationships",
        )

        # Verify request body includes prompt
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "json" in call_kwargs
        assert (
            call_kwargs["json"]["prompt"] == "Extract all people and organizations involved in employment relationships"
        )

        # Verify response
        assert len(result.results) == 2

    def test_multiple_entities_extracted(self, api):
        """Test extraction of multiple entity types."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {"entity": "person", "span": "Elon Musk", "start": 0, "end": 9, "score": 0.97},
                {"entity": "organization", "span": "Tesla", "start": 18, "end": 23, "score": 0.95},
                {"entity": "product", "span": "Model 3", "start": 35, "end": 42, "score": 0.89},
            ]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(
            text="Elon Musk founded Tesla and launched Model 3",
            labels=["person", "organization", "product", "location"],
        )

        assert len(result.results) == 3
        assert result.results[0].entity == "person"
        assert result.results[1].entity == "organization"
        assert result.results[2].entity == "product"

    def test_no_entities_above_threshold(self, api):
        """Test when no entities meet the threshold."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": []}
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(
            text="Ambiguous text with no clear entities", labels=["person", "organization", "location"], threshold=0.9
        )

        assert len(result.results) == 0

    def test_complete_request_all_fields(self, api):
        """Test complete request with all optional fields populated."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"entity": "person", "span": "Dr. Smith", "start": 0, "end": 9, "score": 0.93}],
            "input_text": "Dr. Smith published research on AI",
            "labels": ["person", "organization", "technology"],
            "threshold": 0.6,
            "status": "ok",
        }
        mock_response.headers = {"x-request-id": "req-extract-123"}
        api.configs.client.request.return_value = mock_response

        result = api.it(
            text="Dr. Smith published research on AI",
            labels=["person", "organization", "technology"],
            threshold=0.6,
            prompt="Extract entities related to research",
            timeout_ms=30000,
            http_headers={"X-Custom-Header": "custom-value"},
        )

        # Verify request structure
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "json" in call_kwargs
        assert call_kwargs["json"]["text"] == "Dr. Smith published research on AI"
        assert len(call_kwargs["json"]["labels"]) == 3
        assert call_kwargs["json"]["threshold"] == 0.6
        assert call_kwargs["json"]["prompt"] == "Extract entities related to research"

        # Verify response
        assert len(result.results) == 1
        assert result.results[0].entity == "person"
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
            api.it(text="Test text", labels=["entity1", "entity2"])

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
            api.it(text="Test text", labels=["entity1", "entity2"])

        assert "missing 'outputs'" in str(exc_info.value)

    def test_malformed_entity_item_error(self, api):
        """Test that malformed items in outputs raise ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {"entity": "person"}  # Missing required fields
            ]
        }
        mock_response.headers = {"x-request-id": "req-malformed"}
        api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            api.it(text="Test text", labels=["entity1", "entity2"])

        assert "Unexpected entity record format" in str(exc_info.value)

    def test_server_error_500(self, api):
        """Test handling of 500 Internal Server Error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.return_value = {"error": "Server error occurred"}
        mock_response.headers = {"x-request-id": "req-500-err"}
        api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError):
            api.it(text="Test text", labels=["entity1", "entity2"])


class TestExtractRelationAPIMockRequests:
    """Test Extract API relation-extraction endpoint with mock requests."""

    @pytest.fixture
    def api(self):
        """Create ExtractAPI instance with mock configuration."""
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return ExtractAPI(configs)

    def test_minimal_relation_request(self, api):
        """Test minimal relation extraction request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "source": {"entity": "person", "span": "Steve Jobs", "start": 0, "end": 10, "score": 0.95},
                    "relation": "founded_by",
                    "target": {"entity": "organization", "span": "Apple Inc.", "start": 19, "end": 29, "score": 0.92},
                    "score": 0.89,
                }
            ]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.relations(
            text="Steve Jobs founded Apple Inc.",
            entities=["person", "organization"],
            relations=["founded_by", "works_for"],
        )

        # Verify request
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "json" in call_kwargs
        assert call_kwargs["json"]["text"] == "Steve Jobs founded Apple Inc."
        assert call_kwargs["json"]["entities"] == ["person", "organization"]
        assert call_kwargs["json"]["relations"] == ["founded_by", "works_for"]

        # Verify response
        assert len(result.results) == 1
        assert result.results[0].relation == "founded_by"
        assert result.results[0].source.span == "Steve Jobs"
        assert result.results[0].target.span == "Apple Inc."

    def test_relation_extraction_with_rules(self, api):
        """Test relation extraction with filtering rules."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "source": {"entity": "person", "span": "Alice", "start": 0, "end": 5, "score": 0.93},
                    "relation": "works_for",
                    "target": {"entity": "organization", "span": "Google", "start": 16, "end": 22, "score": 0.91},
                    "score": 0.87,
                }
            ]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        rules = [{"relation": "works_for", "pairs_filter": [["person", "organization"]], "distance": 200}]

        result = api.relations(
            text="Alice works for Google",
            entities=["person", "organization"],
            relations=["works_for", "founded_by"],
            rules=rules,
        )

        # Verify request includes rules
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "json" in call_kwargs
        assert "rules" in call_kwargs["json"]
        assert len(call_kwargs["json"]["rules"]) == 1
        assert call_kwargs["json"]["rules"][0]["relation"] == "works_for"

        # Verify response
        assert len(result.results) == 1
        assert result.results[0].relation == "works_for"

    def test_multiple_relations_extracted(self, api):
        """Test extraction of multiple relations."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "source": {"entity": "person", "span": "John", "start": 0, "end": 4, "score": 0.94},
                    "relation": "works_for",
                    "target": {"entity": "organization", "span": "Microsoft", "start": 15, "end": 24, "score": 0.92},
                    "score": 0.88,
                },
                {
                    "source": {"entity": "organization", "span": "Microsoft", "start": 15, "end": 24, "score": 0.92},
                    "relation": "located_in",
                    "target": {"entity": "location", "span": "Seattle", "start": 28, "end": 35, "score": 0.90},
                    "score": 0.85,
                },
            ]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.relations(
            text="John works for Microsoft in Seattle",
            entities=["person", "organization", "location"],
            relations=["works_for", "located_in"],
        )

        assert len(result.results) == 2
        assert result.results[0].relation == "works_for"
        assert result.results[1].relation == "located_in"

    def test_relation_with_prompt(self, api):
        """Test relation extraction with prompt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "source": {"entity": "organization", "span": "Tesla", "start": 0, "end": 5, "score": 0.96},
                    "relation": "acquired_by",
                    "target": {"entity": "organization", "span": "SolarCity", "start": 15, "end": 24, "score": 0.93},
                    "score": 0.91,
                }
            ]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.relations(
            text="Tesla acquired SolarCity",
            entities=["organization"],
            relations=["acquired_by", "merged_with"],
            prompt="Extract business acquisition relationships",
        )

        # Verify prompt in request
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert call_kwargs["json"]["prompt"] == "Extract business acquisition relationships"

        assert len(result.results) == 1


class TestExtractSummaryAPIMockRequests:
    """Test Extract API summarization endpoint with mock requests."""

    @pytest.fixture
    def api(self):
        """Create ExtractAPI instance with mock configuration."""
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return ExtractAPI(configs)

    def test_minimal_summary_request(self, api):
        """Test minimal summarization request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"summary": "This is a concise summary of the document."}
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.summary(
            text="Long document with lots of detailed information that needs to be summarized into key points."
        )

        # Verify request
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "json" in call_kwargs
        assert "text" in call_kwargs["json"]

        # Verify response
        assert result.summary == "This is a concise summary of the document."

    def test_summary_with_prompt(self, api):
        """Test summarization with custom prompt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "summary": "Key findings: AI is transforming healthcare through predictive diagnostics."
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.summary(
            text="Research paper about AI applications in healthcare...",
            prompt="Extract key findings and main conclusions",
        )

        # Verify prompt in request
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert call_kwargs["json"]["prompt"] == "Extract key findings and main conclusions"

        # Verify response
        assert "Key findings" in result.summary

    def test_summary_with_threshold(self, api):
        """Test summarization with threshold parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"summary": "High-confidence summary of the content."}
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.summary(text="Document to summarize", threshold=0.8)

        # Verify threshold in request
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert call_kwargs["json"]["threshold"] == 0.8

        assert result.threshold == 0.8

    def test_summary_complete_request(self, api):
        """Test complete summarization request with all parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "summary": "Comprehensive summary with all key points.",
            "input_text": "Full document text...",
            "threshold": 0.7,
            "status": "ok",
        }
        mock_response.headers = {"x-request-id": "req-summary-456"}
        api.configs.client.request.return_value = mock_response

        result = api.summary(
            text="Full document text...",
            threshold=0.7,
            prompt="Summarize focusing on risks and limitations",
            timeout_ms=30000,
            http_headers={"X-Custom-Header": "value"},
        )

        # Verify request
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert call_kwargs["json"]["threshold"] == 0.7
        assert call_kwargs["json"]["prompt"] == "Summarize focusing on risks and limitations"

        # Verify response
        assert result.summary == "Comprehensive summary with all key points."
        assert result.status == "ok"


class TestExtractAPIAsyncMockRequests:
    """Test async version of Extract API with mock requests."""

    @pytest.fixture
    def api(self):
        """Create ExtractAPI instance with mock configuration."""
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return ExtractAPI(configs)

    @pytest.mark.asyncio
    async def test_async_basic_extraction(self, api):
        """Test basic async entity extraction."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"entity": "organization", "span": "Amazon", "start": 0, "end": 6, "score": 0.94}]
        }
        mock_response.headers = {}

        # Mock the send method directly
        with patch.object(api, "send", new=AsyncMock(return_value=mock_response)):
            result = await api.it_async(text="Amazon launches new service", labels=["organization", "product"])

            assert result.results[0].span == "Amazon"
            assert result.results[0].entity == "organization"

    @pytest.mark.asyncio
    async def test_async_relation_extraction(self, api):
        """Test async relation extraction."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "source": {"entity": "person", "span": "Bob", "start": 0, "end": 3, "score": 0.92},
                    "relation": "works_for",
                    "target": {"entity": "organization", "span": "Apple", "start": 14, "end": 19, "score": 0.90},
                    "score": 0.85,
                }
            ]
        }
        mock_response.headers = {}

        # Mock the send method directly
        with patch.object(api, "send", new=AsyncMock(return_value=mock_response)):
            result = await api.relations_async(
                text="Bob works for Apple", entities=["person", "organization"], relations=["works_for"]
            )

            assert len(result.results) == 1
            assert result.results[0].relation == "works_for"

    @pytest.mark.asyncio
    async def test_async_summarization(self, api):
        """Test async summarization."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"summary": "Brief async summary of the document."}
        mock_response.headers = {}

        # Mock the send method directly
        with patch.object(api, "send", new=AsyncMock(return_value=mock_response)):
            result = await api.summary_async(text="Long document to summarize asynchronously")

            assert result.summary == "Brief async summary of the document."
