import pytest


def test_e2e_list_datasets(sync_sdk_client):
    res = sync_sdk_client.datasets.list(limit=10)
    assert res is not None
    assert hasattr(res, "datasets") or isinstance(res, dict) or isinstance(res, list)


def test_e2e_list_formats(sync_sdk_client):
    res = sync_sdk_client.datasets.list_formats()
    assert res is not None


def test_e2e_list_data(sync_sdk_client):
    res = sync_sdk_client.datasets.list_data(limit=10)
    assert res is not None


@pytest.mark.asyncio
async def test_e2e_async_list_datasets(async_sdk_client):
    res = await async_sdk_client.datasets.list_async(limit=10)
    assert res is not None


def test_e2e_generate_plan(sync_sdk_client):
    res = sync_sdk_client.datasets.generate_plan(
        prompt="Create a customer feedback classification dataset"
    )
    assert res is not None


def test_e2e_generate_dataset(sync_sdk_client):
    res = sync_sdk_client.datasets.generate(
        name="test-e2e-single-sample",
        prompt="Product review sentiment classification",
        type="classification",
        num_samples=1,
        domains=["e-commerce"],
        classes=["positive", "negative"],
    )
    assert res is not None

