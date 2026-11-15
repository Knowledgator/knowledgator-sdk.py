from __future__ import annotations

import time
import asyncio
import urllib.parse
from typing import Any, Union, Iterable, Iterator, Optional

import httpx

from ..configs import RetryPolicyConfigs
from ..utils.dictionaries_op import remove_none_from_dict, remove_omit_from_dict


class BaseHttpClient:
    def __init__(
        self,
        *,
        client: httpx.Client | httpx.AsyncClient | None = None,
        base_url: str | None = None,
        timeout: Union[int, float] = 5,
        retry_policy: RetryPolicyConfigs | None = None,
    ):
        self._client = client
        self._base_url = base_url or ""
        self._timeout = timeout
        self._retry = retry_policy or RetryPolicyConfigs()

    def _url(self, path: str | None) -> str:
        return urllib.parse.urljoin(self._base_url, path) if path else self._base_url

    @property
    def is_closed(self) -> bool:
        return getattr(self._client, "is_closed", False)


class HttpClient(BaseHttpClient):
    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        base_url: str | None = None,
        timeout: Union[int, float] = 5,
        retry_policy: RetryPolicyConfigs | None = None,
    ) -> None:
        super().__init__(
            client=client,
            base_url=base_url,
            timeout=timeout,
            retry_policy=retry_policy,
        )

        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout, follow_redirects=True)

    def __enter__(self):
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout, follow_redirects=True)
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        if self._client is not None and not getattr(self._client, "is_closed", False):
            self._client.close()

    def request(
        self,
        url: str | None = None,
        method: str = "POST",
        path: str | None = None,
        *,
        params: Optional[dict[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        content: Optional[Union[bytes, Iterator[bytes]]] = None,
        files: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        follow_redirects: bool | None = True,
        timeout: Optional[Union[int, float]] = None,
        retries: Optional[int] = None,
        omit: Optional[Iterable[str]] = None,
    ) -> httpx.Response:
        if url is None:
            url = self._url(path)

        if params is not None:
            req_params = remove_omit_from_dict(remove_none_from_dict(params), omit)
        else:
            req_params = None
        max_tries = retries if retries is not None else self._retry.max_tries

        req_data = data
        req_files = files
        req_headers = headers.copy() if headers else {}
        if isinstance(data, dict) and not files:
            req_files = {
                k: (None, str(v)) if not isinstance(v, (tuple, list)) else v
                for k, v in data.items()
            }
            req_data = None
            req_headers.pop("content-type", None)
            req_headers.pop("Content-Type", None)

        attempt = 0
        while True:
            resp = self._client.request(
                method=method,
                url=url,
                params=req_params,
                data=req_data,
                json=json,
                content=content,
                files=req_files,
                headers=req_headers,
                timeout=timeout or self._timeout,
                follow_redirects=follow_redirects if follow_redirects is not None else True,
            )
            if attempt >= max_tries - 1 or not self._retry.should_retry(resp.status_code):
                return resp

            time.sleep(self._retry.backoff(attempt))
            attempt += 1

    # same convenience wrappers as async version
    def get(self, path=None, **kw):
        return self.request(method="GET", path=path, **kw)

    def post(self, path=None, **kw):
        return self.request(method="POST", path=path, **kw)

    def put(self, path=None, **kw):
        return self.request(method="PUT", path=path, **kw)

    def delete(self, path=None, **kw):
        return self.request(method="DELETE", path=path, **kw)


class AsyncHttpClient(BaseHttpClient):
    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str | None = None,
        timeout: Union[int, float] = 5,
        retry_policy: RetryPolicyConfigs | None = None,
    ) -> None:
        super().__init__(
            client=client,
            base_url=base_url,
            timeout=timeout,
            retry_policy=retry_policy,
        )

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)

    async def __aenter__(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()

    async def request(
        self,
        url: str | None = None,
        method: str = "POST",
        path: str | None = None,
        *,
        params: Optional[dict[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        content: Optional[Union[bytes, Iterator[bytes]]] = None,
        files: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        follow_redirects: bool | None = True,
        timeout: Optional[Union[int, float]] = None,
        retries: Optional[int] = None,
        omit: Optional[Iterable[str]] = None,
    ) -> httpx.Response:
        if url is None:
            url = self._url(path)
        if params is not None:
            req_params = remove_omit_from_dict(remove_none_from_dict(params), omit)
        else:
            req_params = None
        max_tries = retries if retries is not None else self._retry.max_tries

        req_data = data
        req_files = files
        req_headers = headers.copy() if headers else {}
        if isinstance(data, dict) and not files:
            req_files = {
                k: (None, str(v)) if not isinstance(v, (tuple, list)) else v
                for k, v in data.items()
            }
            req_data = None
            req_headers.pop("content-type", None)
            req_headers.pop("Content-Type", None)

        attempt = 0
        while True:
            resp = await self._client.request(
                method=method,
                url=url,
                params=req_params,
                data=req_data,
                json=json,
                content=content,
                files=req_files,
                headers=req_headers,
                timeout=timeout or self._timeout,
                follow_redirects=follow_redirects if follow_redirects is not None else True,
            )
            if attempt >= max_tries - 1 or not self._retry.should_retry(resp.status_code):
                return resp  # success or exhausted retries

            await asyncio.sleep(self._retry.backoff(attempt))
            attempt += 1

    # same convenience wrappers as sync version
    async def get(self, path=None, **kw):
        return await self.request(method="GET", path=path, **kw)

    async def post(self, path=None, **kw):
        return await self.request(method="POST", path=path, **kw)

    async def put(self, path=None, **kw):
        return await self.request(method="PUT", path=path, **kw)

    async def delete(self, path=None, **kw):
        return await self.request(method="DELETE", path=path, **kw)
