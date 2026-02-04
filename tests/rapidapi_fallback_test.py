from knowledgator.sdk import KnowledgatorRapidAPI


def test_rapidapi_datasets_finetuning_fallback():
    # Initialize KnowledgatorRapidAPI with a dummy key
    kg = KnowledgatorRapidAPI(api_key="test-api-key")

    # Verify that datasets and finetuning attributes are initialized (not None)
    assert kg.datasets is not None
    assert kg.finetuning is not None

    # Verify that the internal configuration points to the main server
    kg.datasets.configs.get_server_details()
    kg.finetuning.configs.get_server_details()

    # get_server_details returns (url, headers)
    d_url, d_headers = kg.datasets.configs.get_server_details()
    f_url, f_headers = kg.finetuning.configs.get_server_details()

    # Assert URL is the main server
    assert "api.knowledgator.com" in d_url
    assert "api.knowledgator.com" in f_url

    # Assert Headers contain KnowledgatorAPI-Key
    assert "KnowledgatorAPI-Key" in d_headers
    assert d_headers["KnowledgatorAPI-Key"] == "test-api-key"
    assert "KnowledgatorAPI-Key" in f_headers
    assert f_headers["KnowledgatorAPI-Key"] == "test-api-key"

    # Verify that other services are still using RapidAPI
    c_url, c_headers = kg.comprehend.configs.get_server_details()
    assert "rapidapi.com" in c_url
    assert "x-rapidapi-key" in c_headers
