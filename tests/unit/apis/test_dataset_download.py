import json
from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest

from knowledgator.apis.datasets import DatasetsAPI
from knowledgator.configs import SDKConfigs, ServersConfigs
from knowledgator.exceptions import NotFound, ServerError, ClientError


@pytest.fixture
def api():
    """Create DatasetsAPI instance with mock configuration."""
    configs = SDKConfigs()
    configs.server_config = ServersConfigs(servers={"main": "https://api.platform.knowledgator.com"})
    configs.client = MagicMock()
    configs.async_client = MagicMock()
    configs.user_agent = "knowledgator-sdk/1.0"
    return DatasetsAPI(configs)


def test_download_dataset_bytes(api):
    """Test downloading dataset returning raw bytes."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.content = b'[{"text": "Sample", "label": "test"}]'
    mock_response.headers = {"content-type": "application/octet-stream"}
    api.configs.client.request.return_value = mock_response

    result = api.download(dataset_id="d655eda9-ca67-4a86-a8bc-5b3911233c79")

    assert result == b'[{"text": "Sample", "label": "test"}]'
    api.configs.client.request.assert_called_once()
    call_kwargs = api.configs.client.request.call_args.kwargs
    assert "/datasets/d655eda9-ca67-4a86-a8bc-5b3911233c79/download" in str(call_kwargs["url"])


def test_download_dataset_return_data(api):
    """Test downloading dataset with return_data=True."""
    payload = [{"text": "Sample A", "label": "pos"}, {"text": "Sample B", "label": "neg"}]
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.content = json.dumps(payload).encode("utf-8")
    mock_response.headers = {"content-type": "application/json"}
    api.configs.client.request.return_value = mock_response

    result = api.download(dataset_id="d655eda9-ca67-4a86-a8bc-5b3911233c79", return_data=True)

    assert result == payload
    assert len(result) == 2


def test_download_dataset_storage_url_redirect(api):
    """Test downloading dataset when API returns JSON redirect to S3 storage."""
    payload = [{"text": "Redirected sample", "label": "cat_1"}]
    redirect_body = json.dumps({"url": "https://storage.knowledgator.com/files/ds.json"}).encode("utf-8")

    mock_redirect_response = MagicMock(spec=httpx.Response)
    mock_redirect_response.status_code = 200
    mock_redirect_response.content = redirect_body
    mock_redirect_response.headers = {"content-type": "application/json"}

    mock_storage_response = MagicMock(spec=httpx.Response)
    mock_storage_response.status_code = 200
    mock_storage_response.content = json.dumps(payload).encode("utf-8")
    mock_storage_response.headers = {"content-type": "application/json"}

    api.configs.client.request.side_effect = [mock_redirect_response, mock_storage_response]

    result = api.download(dataset_id="d655eda9-ca67-4a86-a8bc-5b3911233c79", return_data=True)

    assert result == payload
    assert api.configs.client.request.call_count == 2


def test_download_dataset_not_found(api):
    """Test 404 error handling on download."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.reason_phrase = "Not Found"
    mock_response.content = b'{"message": "Dataset not found"}'
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"message": "Dataset not found"}
    mock_response.text = '{"message": "Dataset not found"}'
    api.configs.client.request.return_value = mock_response

    with pytest.raises(NotFound):
        api.download(dataset_id="non-existent-id")


@pytest.mark.asyncio
async def test_download_dataset_async_bytes(api):
    """Test async downloading dataset returning raw bytes."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.content = b'[{"text": "Async sample", "label": "async"}]'
    mock_response.headers = {"content-type": "application/octet-stream"}
    api.configs.async_client.request = AsyncMock(return_value=mock_response)

    result = await api.download_async(dataset_id="d655eda9-ca67-4a86-a8bc-5b3911233c79")

    assert result == b'[{"text": "Async sample", "label": "async"}]'
    api.configs.async_client.request.assert_called_once()


@pytest.mark.asyncio
async def test_download_dataset_async_return_data(api):
    """Test async downloading dataset with return_data=True."""
    payload = [{"text": "Async sample parsed", "label": "active"}]
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.content = json.dumps(payload).encode("utf-8")
    mock_response.headers = {"content-type": "application/json"}
    api.configs.async_client.request = AsyncMock(return_value=mock_response)

    result = await api.download_async(dataset_id="d655eda9-ca67-4a86-a8bc-5b3911233c79", return_data=True)

    assert result == payload


@pytest.mark.asyncio
async def test_download_dataset_async_storage_url_redirect(api):
    """Test async downloading dataset when API returns JSON redirect to storage."""
    payload = [{"text": "Async redirected sample", "label": "async_cat"}]
    redirect_body = json.dumps({"url": "https://storage.knowledgator.com/files/ds_async.json"}).encode("utf-8")

    mock_redirect_response = MagicMock(spec=httpx.Response)
    mock_redirect_response.status_code = 200
    mock_redirect_response.content = redirect_body
    mock_redirect_response.headers = {"content-type": "application/json"}

    mock_storage_response = MagicMock(spec=httpx.Response)
    mock_storage_response.status_code = 200
    mock_storage_response.content = json.dumps(payload).encode("utf-8")
    mock_storage_response.headers = {"content-type": "application/json"}

    api.configs.async_client.request = AsyncMock(side_effect=[mock_redirect_response, mock_storage_response])

    result = await api.download_async(dataset_id="d655eda9-ca67-4a86-a8bc-5b3911233c79", return_data=True)

    assert result == payload
    assert api.configs.async_client.request.call_count == 2


def test_stream_download_chunks(api):
    """Test stream_download yields chunks without loading whole file in single chunk."""
    data = b"chunk_1_data_chunk_2_data_chunk_3_data"
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.content = data
    mock_response.headers = {"content-type": "application/octet-stream"}
    api.configs.client.request.return_value = mock_response

    chunks = list(api.stream_download("d655eda9-ca67-4a86-a8bc-5b3911233c79", chunk_size=12))

    assert len(chunks) == 4
    assert b"".join(chunks) == data


def test_download_to_file_path(api, tmp_path):
    """Test downloading directly to a destination file path."""
    data = b'[{"sample": "data"}]'
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.content = data
    mock_response.headers = {"content-type": "application/json"}
    api.configs.client.request.return_value = mock_response

    dest = tmp_path / "streamed_dest.json"
    result_path = api.download("d655eda9-ca67-4a86-a8bc-5b3911233c79", file_path=dest)

    assert result_path == dest
    assert dest.is_file()
    assert dest.read_bytes() == data


def test_download_storage_redirect_error_raises_exception(api):
    """Verify that if S3 storage redirect fails with 403/500, it raises an exception instead of silently passing."""
    redirect_body = json.dumps({"url": "https://storage.knowledgator.com/files/ds.json"}).encode("utf-8")

    mock_redirect_response = MagicMock(spec=httpx.Response)
    mock_redirect_response.status_code = 200
    mock_redirect_response.content = redirect_body
    mock_redirect_response.headers = {"content-type": "application/json"}

    mock_storage_response = MagicMock(spec=httpx.Response)
    mock_storage_response.status_code = 403
    mock_storage_response.reason_phrase = "Forbidden"
    mock_storage_response.content = b'{"message": "Signature expired"}'
    mock_storage_response.headers = {"content-type": "application/json"}
    mock_storage_response.json.return_value = {"message": "Signature expired"}
    mock_storage_response.text = '{"message": "Signature expired"}'

    api.configs.client.request.side_effect = [mock_redirect_response, mock_storage_response]

    with pytest.raises(ClientError):
        api.download(dataset_id="d655eda9-ca67-4a86-a8bc-5b3911233c79")
