from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Union, Iterator

import asyncio, httpx, random, time, urllib.parse

from ..utils.dictionaries_op import remove_none_from_dict, remove_omit_from_dict
from ..configs import RetryPolicyConfigs

class BaseHttpClient:
    def __init__(
        self,
        *,
        client: httpx.Client | httpx.AsyncClient | None = None,
        base_url: str | None = None,
        base_headers: Mapping[str, str] | None = None,
        timeout: Union[int, float] = 5,
        retry_policy: RetryPolicyConfigs | None = None,
    ):
        self._client = client
        self._base_url = base_url or ""
        self._base_headers = dict(base_headers or {})
        self._timeout = timeout
        self._retry = retry_policy or RetryPolicyConfigs()

    def _url(self, path: str | None) -> str:
        return urllib.parse.urljoin(self._base_url, path) if path else self._base_url

    def _headers(self, extra: Mapping[str, str] | None) -> dict[str, str]:
        if extra is None:
            return self._base_headers
        return {**self._base_headers, **remove_none_from_dict(extra)}

class HttpClient(BaseHttpClient):
    def __enter__(self):
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self

    def __exit__(self, *exc):
        self._client.close()

    def request(
        self,
        method: str,
        path: str | None = None,
        *,
        params: Optional[dict[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        content: Optional[Union[bytes, Iterator[bytes]]] = None,
        files: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        follow_redirects: bool | None = None,
        timeout: Optional[Union[int, float]] = None,
        retries: Optional[int] = None,
        omit: Optional[Iterable[str]] = None,
    ) -> httpx.Response:

        url = self._url(path)
        req_headers = self._headers(headers)
        if params is not None:
            req_params  = remove_omit_from_dict(remove_none_from_dict(params), omit)
        else:
            req_params = {}
        max_tries   = retries if retries is not None else self._retry.max_tries

        attempt = 0
        while True:
            resp = self._client.request(
                method=method,
                url=url,
                params=req_params,
                data=data,
                json=json,
                content=content,
                files=files,
                headers=req_headers,
                timeout=timeout or self._timeout,
                follow_redirects=follow_redirects,
            )
            if attempt >= max_tries - 1 or not self._retry.should_retry(resp.status_code):
                return resp

            time.sleep(self._retry.backoff(attempt))
            attempt += 1

    # same convenience wrappers as async version
    def get(self, path=None, **kw):   return self.request("GET",  path, **kw)
    def post(self, path=None, **kw):  return self.request("POST", path, **kw)
    def put(self, path=None, **kw):   return self.request("PUT",  path, **kw)
    def delete(self, path=None, **kw):return self.request("DELETE",path, **kw)

class AsyncHttpClient(BaseHttpClient):
    async def __aenter__(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *exc):
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str | None = None,
        *,
        params: Optional[dict[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        content: Optional[Union[bytes, Iterator[bytes]]] = None,
        files: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        follow_redirects: bool | None = None,
        timeout: Optional[Union[int, float]] = None,
        retries: Optional[int] = None,
        omit: Optional[Iterable[str]] = None,
    ) -> httpx.Response:

        url = self._url(path)
        req_headers = self._headers(headers)
        if params is not None:
            req_params  = remove_omit_from_dict(remove_none_from_dict(params), omit)
        else:
            req_params = {}        
        max_tries   = retries if retries is not None else self._retry.max_tries

        attempt = 0
        while True:
            resp = await self._client.request(
                method=method,
                url=url,
                params=req_params,
                data=data,
                json=json,
                content=content,
                files=files,
                headers=req_headers,
                timeout=timeout or self._timeout,
                follow_redirects=follow_redirects,
            )
            if attempt >= max_tries - 1 or not self._retry.should_retry(resp.status_code):
                return resp         # success or exhausted retries

            await asyncio.sleep(self._retry.backoff(attempt))
            attempt += 1

    async def get(self, path=None, **kw):  return await self.request("GET",  path, **kw)
    async def post(self, path=None, **kw): return await self.request("POST", path, **kw)
    async def put(self, path=None, **kw):  return await self.request("PUT",  path, **kw)
    async def delete(self, path=None, **kw):return await self.request("DELETE",path, **kw)
