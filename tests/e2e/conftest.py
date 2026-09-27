import os

import pytest
import pytest_asyncio

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    pass

API_KEY = os.getenv("KG_API_KEY")
BASE_URL = os.getenv("KG_BASE_URL")


def pytest_collection_modifyitems(config, items):
    """Skip E2E test items if no API key is set in environment."""
    if not API_KEY:
        skip_marker = pytest.mark.skip(reason="KG_API_KEY environment variable is required to run E2E live API tests")
        for item in items:
            item.add_marker(skip_marker)


@pytest.fixture
def sync_sdk_client():
    from knowledgator.sdk import Knowledgator

    return Knowledgator(api_key=API_KEY, server_url=BASE_URL, timeout_ms=60000)


@pytest_asyncio.fixture
async def async_sdk_client():
    from knowledgator.sdk import Knowledgator

    client = Knowledgator(api_key=API_KEY, server_url=BASE_URL, timeout_ms=60000)
    yield client
    await client.aclose()
