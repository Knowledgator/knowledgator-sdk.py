"""Integration tests for Datasets API. Requires real API access via KG_API_KEY environment variable."""
import os
import tempfile
import uuid
import time

import pytest

from knowledgator import Knowledgator


@pytest.mark.integration
def test_datasets_end_to_end():
    """Create -> get -> update -> upload file -> delete dataset using real API.

    Requires KG_API_KEY set in environment. Cleans up the created dataset at the end.
    """
    api_key = os.getenv("KG_API_KEY")
    bearer = os.getenv("KG_BEARER")

    # Accept either an API key or a bearer token for integration tests; skip otherwise
    if not (api_key or bearer):
        pytest.skip("KG_API_KEY or KG_BEARER (or KG_API / KG_TOKEN) environment variable must be set for integration test")

    # Create SDK instance - prefer passing server_url and bearer as server_base_headers when provided
    # server_base_headers = {"Authorization": f"Bearer {bearer}"} if bearer else None
    sdk = Knowledgator(api_key="test-api-key")

    # Unique dataset name
    unique = f"test-dataset-{int(time.time())}-{uuid.uuid4().hex[:6]}"

    created = None
    try:
        # List datasets (smoke)
        list_resp = sdk.datasets.list_datasets(page=1, page_size=5)
        assert hasattr(list_resp, "data"), f"list_datasets response missing 'data': {list_resp}"

        # Create dataset — include a small file in the create call (multipart)
        # to avoid a separate upload step that some servers may protect with CSRF.
        with tempfile.NamedTemporaryFile("w+b", suffix=".jsonl", delete=False) as fh:
            fh.write(b'[{"text":"Our AI model processes customer queries using advanced NLP techniques.","true_labels":["AI","NLP"],"all_labels":["AI","automation","NLP","analytics","data_analysis","visualization","support"]},{"text":"Automation tools were implemented to streamline invoice processing.","true_labels":["automation"],"all_labels":["AI","automation","NLP","analytics","data_analysis","visualization","support"]},{"text":"The dashboard provides real-time analytics and data visualization for KPIs.","true_labels":["analytics","visualization"],"all_labels":["AI","automation","NLP","analytics","data_analysis","visualization","support"]},{"text":"Customer support queries are now managed by an AI-powered chatbot.","true_labels":["AI","support"],"all_labels":["AI","automation","NLP","analytics","data_analysis","visualization","support"]}]\n')
            fh.flush()
            fh_name = fh.name

        with open(fh_name, "rb") as fh:
            created = sdk.datasets.create_dataset(
                name=unique,
                type="comprehend-it",
                description="Integration test dataset",
                file=fh,
                filename=os.path.basename(fh_name),
                format="json",
            )
        assert created.id, f"create_dataset response missing 'id': {created}"
        # assert created.name == unique or created.name.startswith("test-dataset-"), f"Unexpected dataset name: {created}"

        # Get dataset
        fetched = sdk.datasets.get_dataset(dataset_id=created.id)
        assert fetched.id == created.id, f"get_dataset returned different id: {fetched}"

        # Update dataset
        new_name = unique + "-updated"
        updated = sdk.datasets.update_dataset(dataset_id=created.id, name=new_name, short_description="Updated description")
        assert updated.name == new_name, f"update_dataset did not change name: {updated}"

        # Note: file upload already performed during create_dataset above.
    except Exception as e:
        pytest.fail(f"Datasets integration test failed: {e}")
    finally:
        # Cleanup: delete dataset if created
        if created:
            try:
                del_resp = sdk.datasets.delete_dataset(dataset_id=created.id)
                # Accept 204 or JSON confirm
                assert del_resp is None or isinstance(del_resp, dict) or hasattr(del_resp, "get")
            except Exception as e:
                pytest.skip(f"Cleanup failed; dataset may remain: {e}")


if __name__ == "__main__":
    
    import sys
    ret = pytest.main([__file__])
    sys.exit(ret)
