from typing import Any, Mapping, MutableMapping, Optional
import urllib.parse
import httpx
import time

from ..clients import AsyncHttpClient, HttpClient
from ..configs import (
    SDKConfigs,
    RetryPolicyConfigs,
)


class BaseAPI:

    def __init__(self, configs: SDKConfigs) -> None:
        self.configs = configs

        self._log = getattr(self.configs, "debug_logger", None)
        
        server_url, server_headers = self.configs.get_server_details()
        retry_cfg: RetryPolicyConfigs = (
            self.configs.retry_config or RetryPolicyConfigs()
        )
        timeout_sec: float | int | None = (
            (self.configs.timeout_ms or 0) / 1000
            if self.configs.timeout_ms is not None
            else None
        )

        if self.configs.client is None:
            self.configs.client = HttpClient(
                base_url=server_url,
                base_headers=server_headers,
                timeout=timeout_sec or 5,
                retry_policy=retry_cfg,
            )

            if self._log:
                self._log.debug(
                    "BaseAPI: Created default HttpClient (timeout=%ss, retries=%s)",
                    timeout_sec or 5,
                    retry_cfg.max_tries,
                )

            self.configs.client_supplied = False

        if self.configs.async_client is None:
            self.configs.async_client = AsyncHttpClient(
                base_url=server_url,
                base_headers=server_headers,
                timeout=timeout_sec or 5,
                retry_policy=retry_cfg,
            )

            if self._log:
                self._log.debug(
                    "BaseAPI: Created default AsyncHttpClient (timeout=%ss, retries=%s)",
                    timeout_sec or 5,
                    retry_cfg.max_tries,
                )

            self.configs.async_client_supplied = False

    def _make_request(  
        self,
        *,
        method: str,
        path: str = "",
        url_variables: Optional[Mapping[str, Any]] = None,
        query_params: Optional[Mapping[str, Any]] = None,
        json_body: Optional[Any] = None,
        data_body: Optional[Any] = None,
        files: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        accept: str | None = "application/json",
        user_agent: str | None = None,
        timeout_ms: Optional[int] = None,
        url_override: Optional[str] = None,
    ) -> httpx.Request:

        server_url, _ = self.configs.get_server_details()
        base_url = url_override or urllib.parse.urljoin(server_url, "/")

        raw_path: str = path or ""
        if url_variables:
            for k, v in url_variables.items():
                safe_val = urllib.parse.quote(str(v), safe="")
                raw_path = raw_path.replace(f"{{{k}}}", safe_val)

        url = urllib.parse.urljoin(base_url, raw_path.lstrip("/"))

        final_headers: MutableMapping[str, str] = {}

        if self.configs.server_config and self.configs.server_config.base_headers:
            final_headers.update(self.configs.server_config.base_headers)

        if headers:
            final_headers.update(headers)

        if accept is not None:
            final_headers["Accept"] = accept
        if user_agent is not None:
            final_headers["User-Agent"] = user_agent

        extensions: dict[str, Any] = {}
        if timeout_ms is not None:
            extensions["timeout"] = timeout_ms / 1000.0

        body_kwargs: dict[str, Any] = {}
        body_kind = None
        if json_body is not None:
            body_kwargs["json"] = json_body
            body_kind = "json"
        elif data_body is not None:
            body_kwargs["data"] = data_body
            body_kind = "data"
        if files is not None:
            body_kwargs["files"] = files
            body_kind = "files"

        req = httpx.Request(
            method=method,
            url=url,
            params=query_params,
            headers=final_headers,
            **body_kwargs,
            extensions=extensions or None,
        )

        if self._log:
            self._log.debug(
                "BaseAPI: Prepared %s %s  query=%s body=%s",
                method.upper(),
                url,
                bool(query_params),
                body_kind or "none",
            )

        return req

    def send(self, request: httpx.Request) -> httpx.Response:
        """
        Synchronously send *request* using the configured `HttpClient`,
        inheriting its retry / timeout behaviour.
        """
        if self.configs.client is None:
            raise RuntimeError("Synchronous client is not initialised")
        # Re-use the higher-level retry logic already present in HttpClient.
        
        start = time.perf_counter()

        if self._log:
            self._log.debug("BaseAPI: → %s %s", request.method, request.url)

        response =  self.configs.client.request(
            method=request.method,
            path=None,  # url already fully qualified
            params=request.url.params,
            headers=dict(request.headers),
            data=request.content,
            json=request._json,
            files=request._files,
            timeout=request.extensions.get("timeout") if request.extensions else None,
        )

        if self._log:
            self._log.debug(
                "BaseAPI: ← %s %s  %s  (%.0f ms)",
                request.method,
                request.url,
                response.status_code,
                (time.perf_counter() - start) * 1_000,
            )
        
        return response
    
    async def send_async(self, request: httpx.Request) -> httpx.Response:
        """
        Asynchronously send *request* using the configured `AsyncHttpClient`.
        """
        if self.configs.async_client is None:
            raise RuntimeError("Asynchronous client is not initialised")

        start = time.perf_counter()
        if self._log:
            self._log.debug("BaseAPI: → %s %s (async)", request.method, request.url)

        response = await self.configs.async_client.request(
            method=request.method,
            path=None,
            params=request.url.params,
            headers=dict(request.headers),
            data=request.content,
            json=request._json,
            files=request._files,
            timeout=request.extensions.get("timeout") if request.extensions else None,
        )

        if self._log:
            self._log.debug(
                "BaseAPI: ← %s %s  %s  (%.0f ms, async)",
                request.method,
                request.url,
                response.status_code,
                (time.perf_counter() - start) * 1_000,
            )
        return response
    
    def client(self) -> HttpClient:
        """Return the underlying *synchronous* client (for advanced use-cases)."""
        if self.configs.client is None:
            raise RuntimeError("Synchronous client is not initialised")
        return self.configs.client

    def async_client(self) -> AsyncHttpClient:
        """Return the underlying *asynchronous* client."""
        if self.configs.async_client is None:
            raise RuntimeError("Asynchronous client is not initialised")
        return self.configs.async_client
