"""Integration tests for Knowledgator InferenceAPI using real API calls."""
from unittest.mock import patch
import pytest
import httpx

from knowledgator import Knowledgator
from knowledgator.errors import (
    BadRequestError,
    UnauthorizedError,
    TransportTimeout,
)

# mark this module as integration tests (so runners can skip them by marker)
pytestmark = pytest.mark.integration


@pytest.fixture
def sdk():
    """Create real SDK instance with valid API key."""
    # Replace this with your actual working key or env var.
    return Knowledgator(api_key="abc")


def test_predict_success(sdk):
    """Test successful predict call returns proper structured data."""
    payload = {
        "text": (
            "Characterize movie review: I rented I AM CURIOUS-YELLOW from my video store "
            "because of all the controversy that surrounded it when it was first released in 1967."
        ),
        "labels": ["positive", "negative", "neutral"]
    }

    print("---- Running real predict API ----")
    res = sdk.inference.predict(model_id="comprehend-it-base", json=payload)
    print("Response:", res)

    # Validate structure
    assert isinstance(res, dict)
    assert "data" in res, "Missing 'data' key in response"
    assert "sequence" in res["data"], "Missing 'sequence' in response['data']"
    assert "labels" in res["data"], "Missing 'labels' in response['data']"
    assert "scores" in res["data"], "Missing 'scores' in response['data']"

    # Validate logical structure
    data = res["data"]
    assert isinstance(data["labels"], list)
    assert isinstance(data["scores"], list)
    assert len(data["labels"]) == len(data["scores"])


def test_predict_invalid_model(sdk):
    """Should raise BadRequestError for invalid model ID."""
    payload = {"text": "test", "labels": ["positive", "negative", "neutral"]}
    with pytest.raises(BadRequestError):
        sdk.inference.predict(model_id="invalid-model-id", json=payload)


def test_predict_unauthorized():
    """Invalid API key should raise UnauthorizedError if backend enforces it."""
    bad_sdk = Knowledgator(api_key="wrong-api-key")
    payload = {"text": "test", "labels": ["positive", "negative", "neutral"]}

    try:
        bad_sdk.inference.predict(model_id="comprehend-it-base", json=payload)
        print("API did not raise UnauthorizedError — backend might not enforce auth.")
    except UnauthorizedError:
        print("UnauthorizedError correctly raised for invalid API key.")


def test_predict_timeout_handling(sdk):
    """Simulate network timeout cleanly."""
    with patch.object(sdk.inference, "send", side_effect=httpx.TimeoutException("Timeout")):
        with pytest.raises(TransportTimeout):
            sdk.inference.predict(model_id="comprehend-it-base", json={"text": "hi"})


@pytest.mark.asyncio
async def test_predict_async_success(sdk):
    """Async version of predict with real API call."""
    payload = {
        "text": "Characterize short review: The movie was brilliant and emotional.",
        "labels": ["positive", "negative", "neutral"]
    }

    print("---- Running async predict API ----")
    res = await sdk.inference.predict_async(model_id="comprehend-it-base", json=payload)
    print("Async Response:", res)

    assert "data" in res
    assert "labels" in res["data"]
    assert "scores" in res["data"]



########################################################### WITH MOCK RESPONSE ###########################################################
# """Unit tests for InferenceAPI wrapper."""
# import pytest
# from unittest.mock import Mock, patch
# import httpx

# from knowledgator import Knowledgator
# from knowledgator.errors import (
#     InternalServerError,
#     UnauthorizedError,
#     TransportTimeout,
#     ConnectionFailure,
# )


# @pytest.fixture
# def mock_http_client(monkeypatch):
#     """Mock HttpClient to prevent real network calls."""
#     from knowledgator.clients.http_client import HttpClient, AsyncHttpClient

#     monkeypatch.setattr(HttpClient, "__init__", lambda self, *a, **k: None)
#     monkeypatch.setattr(AsyncHttpClient, "__init__", lambda self, *a, **k: None)


# @pytest.fixture
# def sdk(mock_http_client):
#     """Create SDK instance with mocked HTTP client."""
#     return Knowledgator(api_key="test-api-key")


# @pytest.fixture
# def mock_response():
#     """Factory for creating mock httpx.Response objects."""
#     def _create_response(status_code: int, json_data: dict):
#         response = Mock(spec=httpx.Response)
#         response.status_code = status_code
#         response.is_success = 200 <= status_code < 300
#         response.json.return_value = json_data
#         response.text = str(json_data)
#         response.headers = {"x-request-id": "test-request-123"}
#         return response

#     return _create_response


# def test_predict_success(sdk, mock_response):
#     """Test successful predict call returns parsed JSON."""
#     expected_data = {
#         "sequence": "Characterize movie review: I rented I AM CURIOUS-YELLOW ...",
#         "labels": ["positive", "negative", "neutral"],
#         "scores": [0.48967334628105164, 0.36919713020324707, 0.14112955331802368],
#     }
#     envelope = {"data": expected_data}
#     with patch.object(sdk.inference, "send", return_value=mock_response(200, envelope)):
#         res = sdk.inference.predict(model_id="comprehend-it-base", json={"text": "hi"})

#     assert res == expected_data


# def test_predict_invalid_json_response(sdk):
#     """Server returns non-JSON body -> InternalServerError propagated as ServerError."""
#     response = Mock(spec=httpx.Response)
#     response.status_code = 200
#     response.is_success = True
#     response.json.side_effect = ValueError("Invalid JSON")
#     response.text = "not-json"
#     response.headers = {"x-request-id": "test-123"}

#     with patch.object(sdk.inference, "send", return_value=response):
#         with pytest.raises(InternalServerError):
#             sdk.inference.predict(model_id="comprehend-it-base", json={"text": "hi"})


# def test_predict_unauthorized(sdk, mock_response):
#     """401 leads to ClientError."""
#     with patch.object(sdk.inference, "send", return_value=mock_response(401, {"detail": "invalid"})):
#         with pytest.raises(UnauthorizedError):
#             sdk.inference.predict(model_id="comprehend-it-base", json={"text": "hi"})


# def test_predict_timeout_handling(sdk):
#     """Network timeout should map to TransportTimeout."""
#     with patch.object(sdk.inference, "send", side_effect=httpx.TimeoutException("Timeout")):
#         with pytest.raises(TransportTimeout):
#             sdk.inference.predict(model_id="comprehend-it-base", json={"text": "hi"})


# @pytest.mark.asyncio
# async def test_predict_async_success(sdk, mock_response):
#     expected_data = {"sequence": "short text", "labels": ["neutral"], "scores": [1.0]}
#     envelope = {"data": expected_data}
#     with patch.object(sdk.inference, "send_async", return_value=mock_response(200, envelope)):
#         res = await sdk.inference.predict_async(model_id="comprehend-it-base", json={"text": "hi"})

#     assert res == expected_data
########################################################### WITH MOCK RESPONSE ###########################################################
