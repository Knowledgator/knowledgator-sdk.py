from typing import Dict, List, Any, Optional, Union
import httpx
import urllib
import asyncio
import time

from ..utils import remove_none_from_dict, remove_omit_from_dict
from ..configs import HttpConfig

def _should_retry(response: httpx.Response) -> bool:
    retryable_400s = [429, 408, 409]
    return response.status_code >= 500 or response.status_code in retryable_400s

def set_retry_timeout(retries_happened: int) -> float:
    timeout = 0
    if retries_happened>=1:
        timeout = min(HttpClient.initial_retry_delay_seconds*retries_happened**2, HttpClient.max_retry_delay_seconds)
    return timeout if timeout>=0 else 0

class AsyncHttpClient:
    def __init__(   
                    self, 
                    *,
                    httpx_client: httpx.Client = None,
                    base_url: str = None,
                    base_headers: Dict[str, str] = None,
                    timeout: Union[int, float] = 5,
                    ):
        self.httpx_client = httpx_client 
        self.base_url = base_url
        self.base_headers = base_headers
        self.timeout = timeout

    async def request(
        self,
        path: Optional[str] = None,
        *,
        method: str,
        base_url: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        headers: Optional[Dict[str, Any]] = None,
        retries: int = 3,
        timeout: Optional[Union[int, float]] = None,
        omit: Optional[Any] = None,
    ) -> httpx.Response:
        base_url = base_url if base_url is not None else self.base_url
        
        timeout = timeout if timeout is not None else self.timeout

        url = urllib.parse.urljoin(base_url, path) if path else base_url

        headers = {
            **self.base_headers,
            **remove_none_from_dict(headers or {}),
        }

        params = remove_none_from_dict(params or {})
        params = remove_omit_from_dict(params, omit)

        response = await self.httpx_client.request(
            method=method,
            url=url,
            params=params,
            json=json,
            headers=headers,
            timeout=timeout,
        )

        retries = max(HttpClient.max_retries, retries)
        retries_happened = HttpClient.max_retries - retries
        if _should_retry(response) and retries > 0:
            await asyncio.sleep(set_retry_timeout(retries_happened))
            return await self.request(
                path=path,
                method=method,
                base_url=base_url,
                params=params,
                json=json,
                headers=headers,
                retries=retries - 1,
                timeout=timeout,
                omit=omit,
            )


class HttpClient:
    def __init__(   
                    self, 
                    *,
                    httpx_client: httpx.Client = None,
                    base_url: str = None,
                    base_headers: Dict[str, str] = None,
                    timeout: Union[int, float] = 5,
                    ):
        self.httpx_client = httpx_client 
        self.base_url = base_url
        self.base_headers = base_headers
        self.timeout = timeout

    def request(
        self,
        path: Optional[str] = None,
        *,
        method: str,
        base_url: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        headers: Optional[Dict[str, Any]] = None,
        retries: int = 3,
        timeout: Optional[Union[int, float]] = None,
        omit: Optional[Any] = None,
    ) -> httpx.Response:
        base_url = base_url if base_url is not None else self.base_url
        
        timeout = timeout if timeout is not None else self.timeout

        url = urllib.parse.urljoin(base_url, path) if path else base_url

        headers = {
            **self.base_headers,
            **remove_none_from_dict(headers or {}),
        }

        params = remove_none_from_dict(params or {})
        params = remove_omit_from_dict(params, omit)

        response = self.httpx_client.request(
            method=method,
            url=url,
            params=params,
            json=json,
            headers=headers,
            timeout=timeout,
        )

        retries = max(HttpClient.max_retries, retries)
        retries_happened = HttpClient.max_retries - retries
        if _should_retry(response) and retries > 0:
            time.sleep(set_retry_timeout(retries_happened))
            return self.request(
                path=path,
                method=method,
                base_url=base_url,
                params=params,
                json=json,
                headers=headers,
                retries=retries - 1,
                timeout=timeout,
                omit=omit,
            )