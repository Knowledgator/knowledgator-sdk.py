import asyncio
import httpx
import pytest

from knowledgator.clients.http_client import HttpClient, AsyncHttpClient   # ← change if needed
from knowledgator.configs import RetryPolicyConfigs, ServersConfigs

BASE_URL = ServersConfigs().servers["main"]

def ok_transport(json_payload: dict[str, str] | None = None) -> httpx.MockTransport:
    return httpx.MockTransport(lambda r: httpx.Response(200, json=json_payload or {"ok": True}))


def test_sync_basic_init_and_get():
    tr = ok_transport()
    url = f"https://{BASE_URL}/"
    with HttpClient(client=httpx.Client(transport=tr), base_url=url) as cli:
        resp = cli.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_sync_retry_happens(monkeypatch):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500) if calls == 1 else httpx.Response(200, json={"done": True})

    tr = httpx.MockTransport(handler)

    rp = RetryPolicyConfigs(max_tries=2, base_delay=0, jitter=False)

    monkeypatch.setattr("time.sleep", lambda *_: None)
    url = f"https://{BASE_URL}/"
    with HttpClient(client=httpx.Client(transport=tr), base_url=url, retry_policy=rp) as cli:
        resp = cli.get("/")

    assert calls == 2
    assert resp.status_code == 200
    assert resp.json() == {"done": True}


@pytest.mark.asyncio
async def test_async_basic_init_and_get():
    tr = ok_transport({"hello": "world"})
    url = f"https://{BASE_URL}/"
    async with AsyncHttpClient(client=httpx.AsyncClient(transport=tr), base_url=url) as cli:
        resp = await cli.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"hello": "world"}


@pytest.mark.asyncio
async def test_async_retry_happens(monkeypatch):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500) if calls == 1 else httpx.Response(200)

    tr = httpx.MockTransport(handler)
    rp = RetryPolicyConfigs(max_tries=2, base_delay=0, jitter=False)

    async def fake_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    url = f"https://{BASE_URL}/"
    async with AsyncHttpClient(
        client=httpx.AsyncClient(transport=tr),
        base_url=url,
        retry_policy=rp,
    ) as cli:
        resp = await cli.get("/")

    assert calls == 2
    assert resp.status_code == 200
