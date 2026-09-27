from unittest.mock import MagicMock

from knowledgator import Knowledgator


def test_open_model_api_instantiation():
    kg = Knowledgator(api_key="test-key")
    model = kg.model("my-model")
    assert model.model_id == "my-model"
    assert model.method is None


def test_open_model_api_instantiation_with_method():
    kg = Knowledgator(api_key="test-key")
    model = kg.model("my-model", method="generate")
    assert model.model_id == "my-model"
    assert model.method == "generate"


def test_open_model_predict_no_method():
    kg = Knowledgator(api_key="test-key")
    mock_client = MagicMock()
    kg.configs.client = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "success"}
    mock_client.request.return_value = mock_response

    model = kg.model("my-model")
    res = model.predict({"input": "test"})

    assert res == {"result": "success"}
    kwargs = mock_client.request.call_args[1]
    assert str(kwargs["url"]).endswith("/inference/my-model/")
    assert kwargs["method"] == "POST"


def test_open_model_predict_with_default_method():
    kg = Knowledgator(api_key="test-key")
    mock_client = MagicMock()
    kg.configs.client = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "generated"}
    mock_client.request.return_value = mock_response

    model = kg.model("my-model", method="generate")
    res = model.predict({"prompt": "hello"})

    assert res == {"result": "generated"}
    kwargs = mock_client.request.call_args[1]
    # Should call the method-based endpoint
    assert str(kwargs["url"]).endswith("/inference/my-model/generate/")
    assert kwargs["method"] == "POST"


def test_open_model_call():
    kg = Knowledgator(api_key="test-key")
    mock_client = MagicMock()
    kg.configs.client = mock_client
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "called"}
    mock_client.request.return_value = mock_response

    model = kg.model("custom-model")
    res = model.call("generate", {"prompt": "hello"})

    assert res == {"result": "called"}
    kwargs = mock_client.request.call_args[1]
    assert str(kwargs["url"]).endswith("/inference/custom-model/generate/")
    assert kwargs["method"] == "POST"
