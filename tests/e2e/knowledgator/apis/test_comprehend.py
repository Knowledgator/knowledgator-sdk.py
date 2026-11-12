import pytest


def test_sync_comprehend_inference(sync_sdk_client):
    """Verify synchronous comprehend_inference endpoint call against live API."""
    results = sync_sdk_client.comprehend.it(
        text="SpaceX launched another Starship prototype rocket from Texas.",
        tasks={"topic": {"type": "classify", "labels": ["aerospace", "technology", "sports"]}},
        timeout_ms=60000,
    )
    assert results is not None
    assert hasattr(results, "results")
    assert "topic" in results

    # Verify that passing labels directly raises TypeError
    with pytest.raises(TypeError) as exc_info:
        sync_sdk_client.comprehend.it(
            text="SpaceX launched a rocket.",
            labels=["aerospace", "technology"],
        )
    assert "The 'labels' argument has been removed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_comprehend_inference(async_sdk_client):
    """Verify asynchronous comprehend_inference endpoint call against live API."""
    results = await async_sdk_client.comprehend.it_async(
        text="SpaceX launched another Starship prototype rocket from Texas.",
        tasks={"topic": {"type": "classify", "labels": ["aerospace", "technology", "sports"]}},
        timeout_ms=60000,
    )
    assert results is not None
    assert hasattr(results, "results")
    assert "topic" in results or hasattr(results, "topic")


def test_sync_comprehend_multi_task(sync_sdk_client):
    """Verify multi-task comprehend inference (classify, boolean, score) against live API."""
    results = sync_sdk_client.comprehend.it(
        text="Space telescopes help scientists observe distant galaxies, stars, and planets outside our solar system.",
        tasks={
            "topic": {
                "type": "classify",
                "labels": ["space", "medicine", "technology", "sports"],
                "threshold": 0.5,
            },
            "is_scientific": {
                "type": "boolean",
                "instruction": "Is this text about scientific research?",
            },
            "complexity": {
                "type": "score",
                "labels": ["basic", "intermediate", "advanced"],
                "order": ["basic", "intermediate", "advanced"],
            },
        },
        options={"include_usage": True},
        timeout_ms=60000,
    )
    assert results is not None
    assert "topic" in results
    assert "is_scientific" in results
    assert "complexity" in results
    assert results["topic"].type == "classify"
    assert results["is_scientific"].type == "boolean"
    assert results["complexity"].type == "score"
    assert results.topic.value == "space"
    assert results.is_scientific.value is True


@pytest.mark.asyncio
async def test_async_comprehend_multi_task(async_sdk_client):
    """Verify asynchronous multi-task comprehend inference against live API."""
    results = await async_sdk_client.comprehend.it_async(
        text="Space telescopes help scientists observe distant galaxies, stars, and planets outside our solar system.",
        tasks={
            "topic": {
                "type": "classify",
                "labels": ["space", "medicine", "technology", "sports"],
                "threshold": 0.5,
            },
            "is_scientific": {
                "type": "boolean",
                "instruction": "Is this text about scientific research?",
            },
        },
        timeout_ms=60000,
    )
    assert results is not None
    assert "topic" in results
    assert "is_scientific" in results

