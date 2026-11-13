"""
Mock API Request Tests for Instruct-it Endpoint

This module demonstrates the Instruct-it API with mock HTTP requests,
verifying the complete API contract including:
- Request/response structure
- Template handling
- File references
- Different output types
- Error scenarios
"""

import json
from unittest.mock import MagicMock

import pytest

from knowledgator.configs import SDKConfigs, ServersConfigs
from knowledgator.exceptions import BadRequest, ServerError
from knowledgator.apis.instruct import InstructAPI
from knowledgator.models.common import FileMetadata


class TestInstructAPIMockRequests:
    """Test Instruct API with mock HTTP requests matching the API contract."""

    @pytest.fixture
    def api(self):
        """Create InstructAPI instance with mock configuration."""
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return InstructAPI(configs)

    def test_minimal_request_structure(self, api):
        """Test minimal request with only required 'text' field."""
        # Mock response matching API contract
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": "This is a simple text response"}
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Make request
        result = api.it(text="Summarize this policy in 3 bullet points.", template='{"summary": ""}')

        # Verify request was made
        assert api.configs.client.request.called

        # Verify the request was called with data parameter (multipart form-data)
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "data" in call_kwargs
        assert call_kwargs["data"]["text"] == "Summarize this policy in 3 bullet points."
        assert call_kwargs["data"]["template"] == '{"summary": ""}'
        assert "filesmetadata" not in call_kwargs["data"]
        assert "output_type" not in call_kwargs["data"]
        assert not call_kwargs.get("files")  # No file attachments in minimal request

        # Verify response
        assert result.outputs == "This is a simple text response"
        assert result.parsed is True

    def test_request_with_json_template(self, api):
        """Test request with JSON template structure."""
        # Mock response with structured output
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": {
                "summary": "Policy covers data privacy and user rights",
                "key_points": [
                    "Users have right to data access",
                    "Data is encrypted at rest",
                    "Retention period is 7 years",
                ],
                "risk_level": "medium",
            }
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Create template
        template = json.dumps({"summary": "", "key_points": [], "risk_level": ""})

        # Make request
        result = api.it(text="Analyze this policy document", template=template, output_type="json")

        # Verify request body (multipart form-data)
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "data" in call_kwargs
        assert call_kwargs["data"]["text"] == "Analyze this policy document"
        assert call_kwargs["data"]["template"] == template
        assert call_kwargs["data"]["output_type"] == "json"

        # Verify response structure
        assert result.outputs["summary"] == "Policy covers data privacy and user rights"
        assert len(result.outputs["key_points"]) == 3
        assert result.outputs["risk_level"] == "medium"

    def test_request_with_file_references(self, api):
        """Test request with file references (no content)."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": "Analysis complete. File-1 contains financial data, File-2 contains legal terms."
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Create file references (no content)
        files = [FileMetadata(name="file-id-1"), FileMetadata(name="file-id-2")]

        # Make request
        api.it(text="Compare these two documents", template='{"comparison": ""}', files=files)

        # Verify request body includes filesmetadata in data
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "data" in call_kwargs
        assert call_kwargs["data"]["template"] == '{"comparison": ""}'
        assert "filesmetadata" in call_kwargs["data"]
        filesmetadata = json.loads(call_kwargs["data"]["filesmetadata"])
        assert len(filesmetadata) == 2
        assert filesmetadata[0] == "file-id-1"
        assert filesmetadata[1] == "file-id-2"
        # No files parameter since no content
        assert not call_kwargs.get("files")

    def test_request_with_file_content(self, api):
        """Test request with file content transmitted via files parameter."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": ["Step 1: Review document", "Step 2: Extract data", "Step 3: Generate report"]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        # Create files with content
        files = [FileMetadata(name="document.pdf", content=b"PDF binary content here")]

        # Make request
        result = api.it(text="Process this document", template='{"steps": []}', files=files, output_type="list")

        # Verify filesmetadata in data
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "data" in call_kwargs
        filesmetadata = json.loads(call_kwargs["data"]["filesmetadata"])
        assert filesmetadata == ["document.pdf"]

        # Verify file content in files parameter
        assert "files" in call_kwargs
        assert "document.pdf" in call_kwargs["files"]
        assert call_kwargs["files"]["document.pdf"][0] == "document.pdf"  # filename
        assert call_kwargs["files"]["document.pdf"][1] == b"PDF binary content here"  # content

        # Verify response is a list
        assert isinstance(result.outputs, list)
        assert len(result.outputs) == 3

    def test_output_type_json(self, api):
        """Test output_type='json' returns JSON object."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": {"status": "success", "data": {"count": 42, "items": ["a", "b", "c"]}}
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(text="Return structured data", template='{"status": ""}', output_type="json")

        assert isinstance(result.outputs, dict)
        assert result.outputs["status"] == "success"
        assert result.outputs["data"]["count"] == 42

    def test_output_type_list(self, api):
        """Test output_type='list' returns array."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": ["First item", "Second item", "Third item"]}
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(text="Generate a list of steps", template='{"steps": []}', output_type="list")

        assert isinstance(result.outputs, list)
        assert len(result.outputs) == 3

    def test_output_type_string(self, api):
        """Test output_type='string' returns plain text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": "This is a natural language response explaining the concept."}
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(text="Explain quantum computing", template='{"explanation": ""}', output_type="string")

        assert isinstance(result.outputs, str)
        assert "natural language" in result.outputs

    def test_output_type_markdown(self, api):
        """Test output_type='markdown' returns markdown formatted text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": "# Report Title\n\n## Section 1\n\n- Point A\n- Point B\n\n## Section 2\n\nContent here."
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(text="Generate a report in markdown", template='{"report": ""}', output_type="markdown")

        assert "# Report Title" in result.outputs
        assert "## Section" in result.outputs

    def test_custom_output_type(self, api):
        """Test custom output_type like 'test_cases' or 'code'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [{"input": "hello", "expected": "HELLO"}, {"input": "world", "expected": "WORLD"}]
        }
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(
            text="Generate test cases for uppercase function",
            template='{"test_cases": []}',
            output_type="test_cases",
        )

        assert isinstance(result.outputs, list)
        assert result.outputs[0]["input"] == "hello"
        assert result.outputs[0]["expected"] == "HELLO"

    def test_invalid_template_json_error(self, api):
        """Test that invalid JSON template raises BadRequest (400)."""
        with pytest.raises(BadRequest) as exc_info:
            api.it(text="Test instruction", template="not valid json {")

        assert "Invalid JSON in template" in str(exc_info.value)

    def test_server_error_500(self, api):
        """Test handling of 500 Internal Server Error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.return_value = {"error": "Server error occurred"}
        mock_response.headers = {"x-request-id": "req-abc-123"}
        api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError):
            api.it(text="Test instruction", template='{"summary": ""}')

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
            api.it(text="Test instruction", template='{"summary": ""}')

        assert "missing 'outputs'" in str(exc_info.value)

    def test_non_json_response_fallback(self, api):
        """Test that non-JSON response is wrapped as string with parsed=False."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Not JSON")
        mock_response.text = "Plain text response from the model"
        mock_response.headers = {}
        api.configs.client.request.return_value = mock_response

        result = api.it(text="Test instruction", template='{"summary": ""}')

        assert result.outputs == "Plain text response from the model"
        assert result.parsed is False

    def test_complete_request_all_fields(self, api):
        """Test complete request with all optional fields populated (mixed files)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": {"analysis": "Complete analysis", "recommendations": ["Rec 1", "Rec 2"]},
            "parsed": True,
        }
        mock_response.headers = {"x-request-id": "req-full-123"}
        api.configs.client.request.return_value = mock_response

        template = json.dumps({"analysis": "", "recommendations": []})

        # Mixed: one file without content, one with content
        files = [FileMetadata(name="doc1.pdf"), FileMetadata(name="doc2.docx", content=b"content")]

        result = api.it(
            text="Analyze these documents comprehensively",
            template=template,
            files=files,
            output_type="json",
            timeout_ms=30000,
            http_headers={"X-Custom-Header": "custom-value"},
        )

        # Verify request structure (multipart)
        call_kwargs = api.configs.client.request.call_args.kwargs
        assert "data" in call_kwargs
        assert call_kwargs["data"]["text"] == "Analyze these documents comprehensively"
        assert call_kwargs["data"]["template"] == template
        assert call_kwargs["data"]["output_type"] == "json"

        # Verify filesmetadata contains both files
        filesmetadata = json.loads(call_kwargs["data"]["filesmetadata"])
        assert len(filesmetadata) == 2
        assert "doc1.pdf" in filesmetadata
        assert "doc2.docx" in filesmetadata

        # Verify only doc2.docx has content in files parameter
        assert "files" in call_kwargs
        assert "doc2.docx" in call_kwargs["files"]
        assert "doc1.pdf" not in call_kwargs["files"]  # No content

        # Verify response
        assert result.outputs["analysis"] == "Complete analysis"
        assert len(result.outputs["recommendations"]) == 2
        assert result.parsed is True


class TestInstructAPIAsyncMockRequests:
    """Test async version of Instruct API with mock requests."""

    @pytest.fixture
    def api(self):
        """Create InstructAPI instance with mock configuration."""
        configs = SDKConfigs()
        configs.server_config = ServersConfigs(servers={"main": "https://api.knowledgator.com"})
        configs.client = MagicMock()
        configs.async_client = MagicMock()
        configs.user_agent = "knowledgator-sdk/1.0"
        return InstructAPI(configs)

    @pytest.mark.asyncio
    async def test_async_basic_request(self, api):
        """Test basic async request."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": {"answer": "The answer is 42"}}
        mock_response.headers = {}

        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await api.it_async(text="What is the answer?", template='{"answer": ""}')

        assert result.outputs["answer"] == "The answer is 42"
        assert result.parsed is True

    @pytest.mark.asyncio
    async def test_async_with_all_parameters(self, api):
        """Test async request with all parameters."""
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": ["Result 1", "Result 2", "Result 3"]}
        mock_response.headers = {}

        api.configs.async_client.request = AsyncMock(return_value=mock_response)

        template = json.dumps({"items": []})
        files = [FileMetadata(name="file.txt")]

        result = await api.it_async(text="Process this", template=template, files=files, output_type="list")

        assert isinstance(result.outputs, list)
        assert len(result.outputs) == 3
