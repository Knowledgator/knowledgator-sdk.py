import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import ValidationError

from knowledgator.configs import SDKConfigs, ServersConfigs
from knowledgator.exceptions import (
    BadRequest,
    ServerError,
    TransportTimeout,
    ConnectionFailure,
)
from knowledgator.apis.instruct import InstructAPI
from knowledgator.models.common import FileMetadata
from knowledgator.models.inputs import InstructRequest
from knowledgator.models.outputs import InstructResults


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
def instruct_api(mock_configs):
    """Create InstructAPI instance with mocked configs."""
    return InstructAPI(mock_configs)


class TestInstructRequest:
    """Tests for InstructRequest model."""

    def test_minimal_request(self):
        req = InstructRequest(text="Summarize this document", template='{"summary": ""}')
        assert req.text == "Summarize this document"
        assert req.template == '{"summary": ""}'
        assert req.files is None
        assert req.output_type is None

    def test_missing_template_raises_validation_error(self):
        with pytest.raises(ValidationError):
            InstructRequest(text="Summarize this document")

    def test_full_request(self):
        files = [FileMetadata(name="doc.pdf", content=b"pdf content")]
        req = InstructRequest(text="Extract key points", template='{"points": []}', files=files, output_type="json")
        assert req.text == "Extract key points"
        assert req.template == '{"points": []}'
        assert len(req.files) == 1
        assert req.files[0].name == "doc.pdf"
        assert req.output_type == "json"

    def test_request_with_dict_template(self):
        template = {"name": "", "company": ""}
        req = InstructRequest(text="Extract company info", template=template)
        assert req.template == template

    def test_request_with_file_reference_only(self):
        files = [FileMetadata(name="report.docx")]
        req = InstructRequest(text="Summarize the report", template='{"summary": ""}', files=files)
        assert req.files[0].content is None


class TestInstructResults:
    """Tests for InstructResults model."""

    def test_parsed_json_object(self):
        result = InstructResults(
            outputs={"summary": "This is a summary", "key_points": ["point1", "point2"]}, parsed=True
        )
        assert result.outputs["summary"] == "This is a summary"
        assert result.parsed is True

    def test_parsed_json_list(self):
        result = InstructResults(outputs=["item1", "item2", "item3"], parsed=True)
        assert len(result.outputs) == 3
        assert result.parsed is True

    def test_unparsed_string(self):
        result = InstructResults(outputs="This is raw text that couldn't be parsed as JSON", parsed=False)
        assert isinstance(result.outputs, str)
        assert result.parsed is False

    def test_from_model_response_valid_json(self):
        raw = '{"result": "success", "data": [1, 2, 3]}'
        result = InstructResults.from_model_response(raw)
        assert result.parsed is True
        assert result.outputs["result"] == "success"

    def test_from_model_response_invalid_json(self):
        raw = "This is not valid JSON"
        result = InstructResults.from_model_response(raw)
        assert result.parsed is False
        assert result.outputs == raw


class TestInstructAPISync:
    """Tests for synchronous InstructAPI.it() method."""

    def test_basic_instruction(self, instruct_api):
        """Test basic instruction with string template."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": {"summary": "Document summary here"}}
        mock_response.headers = {}
        instruct_api.configs.client.request.return_value = mock_response

        result = instruct_api.it(text="Summarize this document", template='{"summary": ""}')

        assert isinstance(result, InstructResults)
        assert result.outputs["summary"] == "Document summary here"
        assert result.parsed is True

    def test_instruction_with_dict_template(self, instruct_api):
        """Test instruction with dictionary template."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": {"summary": "Document summary here"}}
        mock_response.headers = {}
        instruct_api.configs.client.request.return_value = mock_response

        result = instruct_api.it(text="Summarize this document", template={"summary": ""})

        assert isinstance(result, InstructResults)
        assert result.outputs["summary"] == "Document summary here"
        assert result.parsed is True

    def test_instruction_with_template(self, instruct_api):
        """Test instruction with JSON template."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": {"summary": "Brief summary", "key_points": ["point1", "point2"], "risk_level": "low"}
        }
        mock_response.headers = {}
        instruct_api.configs.client.request.return_value = mock_response

        template = json.dumps({"summary": "", "key_points": [], "risk_level": ""})

        result = instruct_api.it(text="Analyze this policy document", template=template, output_type="json")

        assert result.outputs["risk_level"] == "low"
        assert len(result.outputs["key_points"]) == 2

    def test_instruction_with_files(self, instruct_api):
        """Test instruction with file metadata."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": "Analysis of file1.pdf and file2.docx completed."}
        mock_response.headers = {}
        instruct_api.configs.client.request.return_value = mock_response

        files = [FileMetadata(name="file1.pdf"), FileMetadata(name="file2.docx", content=b"doc content")]

        result = instruct_api.it(
            text="Compare these two documents",
            template='{"comparison": ""}',
            files=files,
        )

        assert "file1.pdf" in result.outputs or isinstance(result.outputs, str)

    def test_invalid_template_json(self, instruct_api):
        """Test that invalid template JSON raises BadRequest."""
        with pytest.raises(BadRequest) as exc_info:
            instruct_api.it(text="Test instruction", template="not valid json {")
        assert "Invalid JSON in template" in str(exc_info.value)

    def test_list_output(self, instruct_api):
        """Test instruction returning a list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": ["Step 1: Do this", "Step 2: Do that", "Step 3: Complete"]}
        mock_response.headers = {}
        instruct_api.configs.client.request.return_value = mock_response

        result = instruct_api.it(
            text="List the steps to complete this task",
            template='{"steps": []}',
            output_type="list",
        )

        assert isinstance(result.outputs, list)
        assert len(result.outputs) == 3

    def test_timeout_exception(self, instruct_api):
        """Test that timeout raises TransportTimeout."""
        instruct_api.configs.client.request.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(TransportTimeout):
            instruct_api.it(text="Test instruction", template='{"summary": ""}')

    def test_connection_error(self, instruct_api):
        """Test that connection errors raise ConnectionFailure."""
        instruct_api.configs.client.request.side_effect = httpx.RequestError("connection failed")

        with pytest.raises(ConnectionFailure):
            instruct_api.it(text="Test instruction", template='{"summary": ""}')

    def test_server_error_response(self, instruct_api):
        """Test handling of server error responses."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.headers = {"x-request-id": "req-123"}
        mock_response.json.return_value = {"error": "Server error"}
        instruct_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError):
            instruct_api.it(text="Test instruction", template='{"summary": ""}')

    def test_missing_outputs_in_response(self, instruct_api):
        """Test that missing 'outputs' in response raises ServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "something"}  # No 'outputs' key
        mock_response.headers = {"x-request-id": "req-456"}
        instruct_api.configs.client.request.return_value = mock_response

        with pytest.raises(ServerError) as exc_info:
            instruct_api.it(text="Test instruction", template='{"summary": ""}')
        assert "missing 'outputs'" in str(exc_info.value)

    def test_non_json_response_wrapped_as_string(self, instruct_api):
        """Test that non-JSON response is wrapped as string output."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("not json")
        mock_response.text = "Plain text response from model"
        mock_response.headers = {}
        instruct_api.configs.client.request.return_value = mock_response

        result = instruct_api.it(text="Test instruction", template='{"summary": ""}')

        assert result.outputs == "Plain text response from model"
        assert result.parsed is False


class TestInstructAPIAsync:
    """Tests for asynchronous InstructAPI.it_async() method."""

    @pytest.mark.asyncio
    async def test_basic_instruction_async(self, instruct_api):
        """Test basic async instruction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"outputs": {"answer": "42"}}
        mock_response.headers = {}

        instruct_api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await instruct_api.it_async(text="What is the meaning of life?", template='{"answer": ""}')

        assert result.outputs["answer"] == "42"
        assert result.parsed is True

    @pytest.mark.asyncio
    async def test_async_with_template(self, instruct_api):
        """Test async instruction with template."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": {"title": "Report Title", "sections": ["intro", "body", "conclusion"]}
        }
        mock_response.headers = {}

        instruct_api.configs.async_client.request = AsyncMock(return_value=mock_response)

        template = json.dumps({"title": "", "sections": []})
        result = await instruct_api.it_async(text="Generate a report structure", template=template)

        assert result.outputs["title"] == "Report Title"

    @pytest.mark.asyncio
    async def test_async_with_dict_template(self, instruct_api):
        """Test async instruction with dictionary template."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": {"title": "Report Title", "sections": ["intro", "body", "conclusion"]}
        }
        mock_response.headers = {}

        instruct_api.configs.async_client.request = AsyncMock(return_value=mock_response)

        result = await instruct_api.it_async(
            text="Generate a report structure",
            template={"title": "", "sections": []},
        )

        assert result.outputs["title"] == "Report Title"

    @pytest.mark.asyncio
    async def test_async_invalid_template(self, instruct_api):
        """Test that invalid template raises BadRequest in async."""
        with pytest.raises(BadRequest):
            await instruct_api.it_async(text="Test", template="invalid json")

    @pytest.mark.asyncio
    async def test_async_timeout(self, instruct_api):
        """Test async timeout handling."""
        instruct_api.configs.async_client.request = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with pytest.raises(TransportTimeout):
            await instruct_api.it_async(text="Test", template='{"summary": ""}')

    @pytest.mark.asyncio
    async def test_async_connection_error(self, instruct_api):
        """Test async connection error handling."""
        instruct_api.configs.async_client.request = AsyncMock(side_effect=httpx.RequestError("connection failed"))

        with pytest.raises(ConnectionFailure):
            await instruct_api.it_async(text="Test", template='{"summary": ""}')


class TestFileMetadata:
    """Tests for FileMetadata dataclass."""

    def test_file_reference_only(self):
        """Test file with just a reference (no content)."""
        file = FileMetadata(name="document.pdf")
        assert file.name == "document.pdf"
        assert file.content is None

    def test_file_with_content(self):
        """Test file with actual content."""
        content = b"Hello, world!"
        file = FileMetadata(name="hello.txt", content=content)
        assert file.name == "hello.txt"
        assert file.content == content

    def test_file_with_empty_content(self):
        """Test file with empty bytes content."""
        file = FileMetadata(name="empty.bin", content=b"")
        assert file.content == b""


class TestOutputTypes:
    """Tests for different output_type values."""

    @pytest.fixture
    def api_with_mock(self, instruct_api):
        def setup_response(outputs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"outputs": outputs}
            mock_response.headers = {}
            instruct_api.configs.client.request.return_value = mock_response
            return instruct_api

        return setup_response

    def test_output_type_json(self, api_with_mock):
        """Test output_type='json' returns object."""
        api = api_with_mock({"key": "value"})
        result = api.it(text="Return JSON", template='{"key": ""}', output_type="json")
        assert isinstance(result.outputs, dict)

    def test_output_type_string(self, api_with_mock):
        """Test output_type='string' returns string."""
        api = api_with_mock("This is a plain string response")
        result = api.it(text="Return string", template='{"key": ""}', output_type="string")
        assert isinstance(result.outputs, str)

    def test_output_type_list(self, api_with_mock):
        """Test output_type='list' returns array."""
        api = api_with_mock(["item1", "item2", "item3"])
        result = api.it(text="Return list", template='{"items": []}', output_type="list")
        assert isinstance(result.outputs, list)

    def test_output_type_markdown(self, api_with_mock):
        """Test output_type='markdown' returns markdown string."""
        markdown = "# Title\n\n- Point 1\n- Point 2"
        api = api_with_mock(markdown)
        result = api.it(text="Return markdown", template='{"content": ""}', output_type="markdown")
        assert "# Title" in result.outputs

    def test_output_type_custom(self, api_with_mock):
        """Test custom output_type like 'test_cases'."""
        test_cases = [{"input": "a", "expected": "b"}, {"input": "c", "expected": "d"}]
        api = api_with_mock(test_cases)
        result = api.it(text="Generate test cases", template='{"cases": []}', output_type="test_cases")
        assert len(result.outputs) == 2
