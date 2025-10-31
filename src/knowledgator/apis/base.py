from typing import Any, Mapping, MutableMapping, Optional
from dataclasses import dataclass
import urllib.parse
import httpx
import time

from ..clients import AsyncHttpClient, HttpClient
from ..configs import (
    SDKConfigs,
    RetryPolicyConfigs,
    ServersConfigs
)
from ..utils.dictionaries_op import remove_none_from_dict

@dataclass
class EndpointsConfig:
    path: str = "/"


class BaseAPI:
    PATH: str
    METHOD: str
    CONTENT_TYPE: str

    def __init__(self, configs: SDKConfigs, 
                        server_configs: ServersConfigs = None, 
                        path: Optional[str] = None, 
                        endpoints_config: Optional[EndpointsConfig] = None) -> None:
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

    def _headers(self, extra: Mapping[str, str] | None) -> dict[str, str]:
        """Merge base headers with any extra headers and return a plain dict.

        Note: httpx.Request accepts a ``dict[str, str]`` for headers. We keep
        them as a normal mapping here; encoding happens inside httpx.
        """
        base = dict(self.base_headers or {})
        extra_clean = remove_none_from_dict(extra or {})
        # Do not overwrite an explicit Authorization header with base headers
        if "Authorization" in base and "Authorization" in extra_clean:
            base.pop("Authorization", None)
        merged = {**base, **extra_clean}
        return merged
    
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

        # If caller supplies an absolute URL in path, honor it. Otherwise join.
        if raw_path.startswith("http://") or raw_path.startswith("https://"):
            url = raw_path
        else:
            url = urllib.parse.urljoin(base_url, raw_path.lstrip("/"))

        extra_headers: MutableMapping[str, str] = {}
        if headers:
            extra_headers.update(headers)
        # Distinguish between Content-Type (based on body) and Accept header
        if accept is not None:
            extra_headers["Accept"] = accept
        if user_agent is not None:
            extra_headers["User-Agent"] = user_agent
        final_headers = self._headers(extra_headers)

        extensions: dict[str, Any] = {}
        if timeout_ms is not None:
            extensions["timeout"] = timeout_ms / 1000.0

        # Support passing multiple body parts together (json, data, files).
        # Store each kind separately in extensions so send()/send_async() can
        # reconstruct the correct call to the underlying httpx client.
        if json_body is not None:
            extensions["_json"] = json_body
        if data_body is not None:
            extensions["_data"] = data_body
        if files is not None:
            extensions["_files"] = files

        # Prepare kwargs for httpx.Request constructor depending on provided parts
        body_kwargs: dict[str, Any] = {}
        if "_json" in extensions:
            body_kwargs["json"] = extensions["_json"]
        if "_data" in extensions:
            body_kwargs["data"] = extensions["_data"]
        if "_files" in extensions:
            body_kwargs["files"] = extensions["_files"]

        req = httpx.Request(
            method=method,
            url=url,
            params=query_params,
            headers=final_headers,
            **body_kwargs,
            extensions=extensions or None,
        )

        if self._log:
            present = []
            if "_json" in extensions:
                present.append("json")
            if "_data" in extensions:
                present.append("data")
            if "_files" in extensions:
                present.append("files")
            body_desc = ",".join(present) if present else "none"
            self._log.debug(
                "BaseAPI: Prepared %s %s  query=%s body=%s",
                method.upper(),
                url,
                bool(query_params),
                body_desc,
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

        # Reconstruct body params from extensions (support json, data and files)
        body_params: dict[str, Any] = {}
        if "_json" in extensions:
            body_params["json"] = extensions.get("_json")
        if "_data" in extensions:
            body_params["data"] = extensions.get("_data")
        if "_files" in extensions:
            body_params["files"] = extensions.get("_files")
        # If nothing was set, fall back to raw request.content
        if not body_params:
            body_params["data"] = request.content

        # httpx.Request.url may be an httpx.URL object; extract the raw path
        path = request.url.raw_path.decode() if isinstance(request.url.raw_path, bytes) else request.url.raw_path
        # Call the configured HttpClient using the (method, path) signature
        response = self.configs.client.request(
            method=request.method,
            path=path,
            params=request.url.params,
            headers=dict(request.headers),
            timeout=extensions.get("timeout"),
            follow_redirects=True,
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

        # Determine body parts from request.extensions and populate body_params
        extensions = request.extensions or {}
        body_params = {}
        if "_json" in extensions:
            body_params["json"] = extensions.get("_json")
        if "_data" in extensions:
            body_params["data"] = extensions.get("_data")
        if "_files" in extensions:
            body_params["files"] = extensions.get("_files")
        if not body_params:
            body_params["data"] = request.content

        path = request.url.raw_path.decode() if isinstance(request.url.raw_path, bytes) else request.url.raw_path
        response = await self.configs.async_client.request(
            method=request.method,
            path=path,
            params=request.url.params,
            headers=dict(request.headers),
            timeout=extensions.get("timeout") if extensions else None,
            follow_redirects=True,
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
