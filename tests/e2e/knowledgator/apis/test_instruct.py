import json
import pytest


def test_sync_instruct_inference(sync_sdk_client):
    """Verify synchronous instruct_inference endpoint call against live API."""
    template_str = json.dumps({"summary": "", "key_points": []})
    results = sync_sdk_client.instruct.it(
        text="What are the main features of Python 3.12?",
        template=template_str,
        output_type="json",
        timeout_ms=60000,
    )
    assert results is not None
    assert hasattr(results, "outputs")
    assert results.outputs is not None


@pytest.mark.asyncio
async def test_async_instruct_inference(async_sdk_client):
    """Verify asynchronous instruct_inference endpoint call against live API."""
    template_str = json.dumps({"summary": "", "key_points": []})
    results = await async_sdk_client.instruct.it_async(
        text="What are the main features of Python 3.12?",
        template=template_str,
        output_type="json",
        timeout_ms=60000,
    )
    assert results is not None
    assert hasattr(results, "outputs")
    assert results.outputs is not None
