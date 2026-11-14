import pytest


def test_sync_extract_inference(sync_sdk_client):
    """Verify synchronous extract_inference endpoint call against live API."""
    res = sync_sdk_client.extract.it(
        text="Elon Musk founded SpaceX in 2002.",
        tasks={"entities": {"type": "span", "labels": ["person", "organization", "year"]}},
        timeout_ms=60000,
    )
    assert res is not None
    assert hasattr(res, "outputs")
    assert "entities" in res.outputs

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
    res = await async_sdk_client.extract.it_async(
        text="Elon Musk founded SpaceX in 2002.",
        tasks={"entities": {"type": "span", "labels": ["person", "organization", "year"]}},
        timeout_ms=60000,
    )
    assert res is not None
    assert hasattr(res, "outputs")
    assert "entities" in res.outputs or hasattr(res, "entities")


def test_sync_extract_multi_task(sync_sdk_client):
    """Verify multi-task extract inference (span, relation) against live API."""
    res = sync_sdk_client.extract.it(
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
    assert res is not None
    assert hasattr(res, "outputs")
    assert "entities" in res.outputs
    assert "relations" in res.outputs
    assert res.outputs["entities"].type == "span"
    assert res.outputs["relations"].type == "relation"
    assert len(res.outputs["entities"].value) > 0
    assert len(res.outputs["relations"].value) > 0


@pytest.mark.asyncio
async def test_async_extract_multi_task(async_sdk_client):
    """Verify asynchronous multi-task extract inference against live API."""
    res = await async_sdk_client.extract.it_async(
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
    assert res is not None
    assert hasattr(res, "outputs")
    assert "entities" in res.outputs

