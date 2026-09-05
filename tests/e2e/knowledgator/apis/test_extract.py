import pytest


def test_sync_extract_inference(sync_sdk_client):
    """Verify synchronous extract_inference endpoint call against live API."""
    results = sync_sdk_client.extract.it(
        text="Elon Musk founded SpaceX in 2002.",
        labels=["person", "organization", "year"],
        timeout_ms=60000,
    )
    assert results is not None
    assert hasattr(results, "results")


@pytest.mark.asyncio
async def test_async_extract_inference(async_sdk_client):
    """Verify asynchronous extract_inference endpoint call against live API."""
    results = await async_sdk_client.extract.it_async(
        text="Elon Musk founded SpaceX in 2002.",
        labels=["person", "organization", "year"],
        timeout_ms=60000,
    )
    assert results is not None
    assert hasattr(results, "results")
