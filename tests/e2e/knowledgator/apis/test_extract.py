import pytest


def test_sync_extract_inference(sync_sdk_client):
    """Verify synchronous extract_inference endpoint call against live API."""
    results = sync_sdk_client.extract.it(
        text="Elon Musk founded SpaceX in 2002.",
        tasks={"entities": {"type": "span", "labels": ["person", "organization", "year"]}},
        timeout_ms=60000,
    )
    assert results is not None
    assert hasattr(results, "results")
    assert "entities" in results

    # Verify that passing labels directly raises TypeError
    with pytest.raises(TypeError) as exc_info:
        sync_sdk_client.extract.it(
            text="Elon Musk founded SpaceX.",
            labels=["person", "organization"],
        )
    assert "The 'labels' argument has been removed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_extract_inference(async_sdk_client):
    """Verify asynchronous extract_inference endpoint call against live API."""
    results = await async_sdk_client.extract.it_async(
        text="Elon Musk founded SpaceX in 2002.",
        tasks={"entities": {"type": "span", "labels": ["person", "organization", "year"]}},
        timeout_ms=60000,
    )
    assert results is not None
    assert hasattr(results, "results")
    assert "entities" in results or hasattr(results, "entities")


def test_sync_extract_multi_task(sync_sdk_client):
    """Verify multi-task extract inference (span, relation) against live API."""
    results = sync_sdk_client.extract.it(
        text="Steve Jobs founded Apple Inc. in Cupertino, California.",
        tasks={
            "entities": {
                "type": "span",
                "labels": ["person", "organization", "location"],
                "threshold": 0.5,
                "flat": False,
            },
            "relations": {
                "type": "relation",
                "labels": ["founded", "located_in"],
                "flat": False,
                "entity_labels": ["person", "organization", "location"],
            },
        },
        options={"include_usage": True},
        timeout_ms=60000,
    )
    assert results is not None
    assert "entities" in results
    assert "relations" in results
    assert results["entities"].type == "span"
    assert results["relations"].type == "relation"
    assert len(results.entities.value) > 0
    assert len(results.relations.value) > 0


@pytest.mark.asyncio
async def test_async_extract_multi_task(async_sdk_client):
    """Verify asynchronous multi-task extract inference against live API."""
    results = await async_sdk_client.extract.it_async(
        text="Steve Jobs founded Apple Inc. in Cupertino, California.",
        tasks={
            "entities": {
                "type": "span",
                "labels": ["person", "organization", "location"],
                "threshold": 0.5,
                "flat": False,
            },
        },
        timeout_ms=60000,
    )
    assert results is not None
    assert "entities" in results

