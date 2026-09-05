from __future__ import annotations

import time
import urllib.parse
from typing import Any, Mapping, Optional, MutableMapping
from dataclasses import dataclass

import httpx

from ..clients import HttpClient, AsyncHttpClient
from ..configs import SDKConfigs, ServersConfigs, RetryPolicyConfigs
from ..utils.dictionaries_op import remove_none_from_dict


@dataclass
class EndpointsConfig:
    path: str = "/"


class BaseAPI:
    PATH: str
    METHOD: str
    CONTENT_TYPE: str

    def __init__(
        self,
        configs: SDKConfigs,
        server_configs: ServersConfigs = None,
        path: Optional[str] = None,
        endpoints_config: Optional[EndpointsConfig] = None,
    ) -> None:
        if path is not None:
            self.PATH = path

        if endpoints_config is not None:
            self.ENDPOINTS_CONFIG = endpoints_config

        self.configs = configs

        if server_configs is not None:
            self.configs.server_config = server_configs

        self._log = getattr(self.configs, "debug_logger", None)

        server_url, server_headers = self.configs.get_server_details()

        self.base_headers = server_headers or {}

        retry_cfg: RetryPolicyConfigs = self.configs.retry_config or RetryPolicyConfigs()
        timeout_sec: float | int | None = (
            (self.configs.timeout_ms or 0) / 1000 if self.configs.timeout_ms is not None else None
        )

        if self.configs.client is None:
            self.configs.client = HttpClient(
                base_url=server_url,
                timeout=timeout_sec or 10,
                retry_policy=retry_cfg,
            )

            if self._log:
                self._log.debug(
                    "BaseAPI: Created default HttpClient (timeout=%ss, retries=%s)",
                    timeout_sec or 10,
                    retry_cfg.max_tries,
                )

            self.configs.client_supplied = False

        if self.configs.async_client is None:
            self.configs.async_client = AsyncHttpClient(
                base_url=server_url,
                timeout=timeout_sec or 10,
                retry_policy=retry_cfg,
            )

            if self._log:
                self._log.debug(
                    "BaseAPI: Created default AsyncHttpClient (timeout=%ss, retries=%s)",
                    timeout_sec or 10,
                    retry_cfg.max_tries,
                )

            self.configs.async_client_supplied = False

    def _headers(self, extra: Mapping[str, str] | None) -> list[tuple[bytes, bytes, bytes]]:
        if extra is None:
            headers = self.base_headers
        else:
            headers = {**self.base_headers, **remove_none_from_dict(extra)}
        dedup_headers: dict[str, tuple[str, str]] = {}
        for k, v in headers.items():
            dedup_headers[k.lower()] = (k, str(v))
        raw_headers: list[tuple[bytes, bytes, bytes]] = [
            (orig_k.encode("ascii"), lower_k.encode("ascii"), val.encode("utf-8"))
            for lower_k, (orig_k, val) in dedup_headers.items()
        ]
        return raw_headers

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

        import re
        clean_path = re.sub(r"/{2,}", "/", raw_path.lstrip("/"))
        url = urllib.parse.urljoin(base_url, clean_path)

        extra_headers: MutableMapping[str, str] = {}
        if headers:
            extra_headers.update(headers)
        if accept is not None:
            extra_headers["Accept"] = accept
        if user_agent is not None:
            extra_headers["User-Agent"] = user_agent
        final_headers = self._headers(extra_headers)

        extensions: dict[str, Any] = {}
        if timeout_ms is not None:
            extensions["timeout"] = timeout_ms / 1000.0

        body_kind: Optional[str] = None
        if json_body is not None:
            body_kind = "json"
            extensions["_body_kind"] = body_kind
            extensions["_body_value"] = json_body
        elif data_body is not None:
            body_kind = "data"
            extensions["_body_kind"] = body_kind
            extensions["_body_value"] = data_body

        # For multipart/form-data, we can have both data and files
        if files is not None:
            if body_kind == "data":
                # Multipart: both data and files
                body_kind = "multipart"
                extensions["_body_kind"] = body_kind
                extensions["_data_value"] = data_body
                extensions["_files_value"] = files
            else:
                # Files only
                body_kind = "files"
                extensions["_body_kind"] = body_kind
                extensions["_body_value"] = files

        # For multipart, don't pass anything to httpx.Request, handle in send()
        if body_kind == "multipart":
            body_kwargs = {}
        else:
            body_kwargs = {extensions.get("_body_kind", "data"): extensions.get("_body_value")}

        req = httpx.Request(
            method=method,
            url=url,
            params=query_params,
            headers={k.decode("ascii"): v.decode("utf-8") for k, _, v in final_headers},
            **body_kwargs,
            extensions=extensions or None,
        )
        # Preserve exact byte case of headers for HTTP wire transmission
        req.headers._list = final_headers

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

        start = time.perf_counter()

        if self._log:
            self._log.debug("BaseAPI: → %s %s", request.method, request.url)

        extensions = request.extensions or {}
        body_kind = extensions.get("_body_kind")
        body_value = extensions.get("_body_value")

        body_params = {}
        if body_kind == "json":
            body_params["json"] = body_value
        elif body_kind == "data":
            body_params["data"] = body_value
        elif body_kind == "files":
            body_params["files"] = body_value
        elif body_kind == "multipart":
            # Both data and files for multipart/form-data
            body_params["data"] = extensions.get("_data_value")
            if extensions.get("_files_value") is not None:
                body_params["files"] = extensions.get("_files_value")
        else:
            body_params["data"] = request.content

        headers = request.headers.copy()
        headers.pop("content-length", None)
        headers.pop("Content-Length", None)

        response = self.configs.client.request(
            url=request.url,
            method=request.method,
            path=None,
            params=None,
            headers=headers,
            timeout=extensions.get("timeout"),
            **body_params,
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

        extensions = request.extensions or {}
        body_kind = extensions.get("_body_kind")
        body_value = extensions.get("_body_value")

        body_params = {}
        if body_kind == "json":
            body_params["json"] = body_value
        elif body_kind == "data":
            body_params["data"] = body_value
        elif body_kind == "files":
            body_params["files"] = body_value
        elif body_kind == "multipart":
            # Both data and files for multipart/form-data
            body_params["data"] = extensions.get("_data_value")
            if extensions.get("_files_value") is not None:
                body_params["files"] = extensions.get("_files_value")
        else:
            body_params["data"] = request.content

        headers = request.headers.copy()
        headers.pop("content-length", None)
        headers.pop("Content-Length", None)

        response = await self.configs.async_client.request(
            url=request.url,
            method=request.method,
            path=None,
            params=None,
            headers=headers,
            timeout=extensions.get("timeout"),
            **body_params,
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
