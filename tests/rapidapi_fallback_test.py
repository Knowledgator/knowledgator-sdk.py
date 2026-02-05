import pytest
from knowledgator.sdk import KnowledgatorRapidAPI


def test_rapidapi_datasets_finetuning_error():
    # Initialize KnowledgatorRapidAPI with a dummy key
    kg = KnowledgatorRapidAPI(api_key="test-api-key")

    # Verify that accessing datasets attributes raises NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        _ = kg.datasets.configs
    assert "Datasets API is not available via RapidAPI" in str(exc_info.value)

    # Verify that accessing finetuning attributes raises NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        _ = kg.finetuning.configs
    assert "Finetuning API is not available via RapidAPI" in str(exc_info.value)

    # Verify that other services are still using RapidAPI
    c_url, c_headers = kg.comprehend.configs.get_server_details()
    assert "rapidapi.com" in c_url
    assert "x-rapidapi-key" in c_headers
