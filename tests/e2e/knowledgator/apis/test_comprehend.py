import pytest


def test_sync_comprehend_inference(sync_sdk_client):
    """Verify synchronous comprehend_inference endpoint call against live API."""
    results = sync_sdk_client.comprehend.it(
        text="SpaceX launched another Starship prototype rocket from Texas.",
        labels=["aerospace", "technology", "sports"],
        timeout_ms=60000,
    )
    assert results is not None
    assert hasattr(results, "results")


@pytest.mark.asyncio
async def test_async_comprehend_inference(async_sdk_client):
    """Verify asynchronous comprehend_inference endpoint call against live API."""
    results = await async_sdk_client.comprehend.it_async(
        text="SpaceX launched another Starship prototype rocket from Texas.",
        labels=["aerospace", "technology", "sports"],
        timeout_ms=60000,
    )
    assert results is not None
    assert hasattr(results, "results")
