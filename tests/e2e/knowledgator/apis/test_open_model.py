import pytest


def test_sync_open_model_inference(sync_sdk_client):
    """Verify synchronous open model prediction against live API without failure."""
    body = {
        "input": "Steve Jobs founded Apple Inc. in Cupertino, California.",
        "tasks": {
            "topic": {"type": "classify", "labels": ["technology", "sports", "politics"], "threshold": 0.5},
            "entities": {"type": "span", "labels": ["person", "organization", "location"]},
        },
        "options": {"include_usage": True},
    }
    results = sync_sdk_client.model("gliformer-large-v1").predict(body, timeout_ms=60000)
    assert results is not None
    assert isinstance(results, dict)
    assert "outputs" in results


@pytest.mark.asyncio
async def test_async_open_model_inference(async_sdk_client):
    """Verify asynchronous open model prediction against live API without failure."""
    body = {
        "input": "Steve Jobs founded Apple Inc. in Cupertino, California.",
        "tasks": {
            "topic": {"type": "classify", "labels": ["technology", "sports", "politics"], "threshold": 0.5},
            "entities": {"type": "span", "labels": ["person", "organization", "location"]},
        },
        "options": {"include_usage": True},
    }
    results = await async_sdk_client.model("gliformer-large-v1").predict_async(body, timeout_ms=60000)
    assert results is not None
    assert isinstance(results, dict)
    assert "outputs" in results
