from __future__ import annotations

from typing import Optional, Any, Dict
import os

from .apis import ComprehendAPI, ExtractAPI, EndpointsConfig
from .configs import (
    SDKConfigs,
    ServersConfigs,
    RetryPolicyConfigs,
    SERVERS
)

class BaseKnowledgator:
    comprehend: ComprehendAPI
    extract: ExtractAPI
    configs: SDKConfigs

    servers_cfg: ServersConfigs
    server_alias: str

    def __init__(
        self, 
        max_tries: int = 3,
        base_delay: float = 0.25,
        max_delay: float = 10.0,
        retry_config: Optional[RetryPolicyConfigs] = None
    ):
        if retry_config is None:
            self.retry_config = RetryPolicyConfigs(
                max_tries=max_tries,
                base_delay=base_delay,
                max_delay=max_delay
            )
        else:
            self.retry_config = retry_config

    def _init_apis(
        self,
        *,
        comprehend_cfg: SDKConfigs,
        extract_cfg: SDKConfigs,
        comprehend_endpoints: Optional[EndpointsConfig] = None,
        extract_endpoints: Optional[EndpointsConfig] = None,
    ) -> None:
        """Instantiate low‑level API helpers with custom endpoint configs."""
        self.comprehend = ComprehendAPI(
            comprehend_cfg, endpoints_config=comprehend_endpoints
        )
        self.extract = ExtractAPI(extract_cfg, endpoints_config=extract_endpoints)


    def close(self) -> None:
        """Close the underlying synchronous client."""
        if (
            self.configs.client
            and not self.configs.client_supplied
            and not getattr(self.configs.client, "is_closed", False)
        ):
            self.configs.client.close()

    async def aclose(self) -> None:
        """Close the underlying async client."""
        if (
            self.configs.async_client
            and not self.configs.async_client_supplied
            and not getattr(self.configs.async_client, "is_closed", False)
        ):
            await self.configs.async_client.aclose()

    async def __aenter__(self) -> "BaseKnowledgator":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def __enter__(self) -> "BaseKnowledgator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class Knowledgator(BaseKnowledgator):
    """
    High-level façade for Knowledgator API helpers.

    >>> kg = Knowledgator(api_key="sk-...")
    >>> res = kg.comprehend.it("Kyiv is beautiful", ["geography", "sport"])
    """
    server_alias = "main"

    def __init__(
        self,
        *,
        configs: Optional[SDKConfigs] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        async_client: Optional[Any] = None,
        server_url: Optional[str] = None,
        server_base_headers: Optional[Dict[str, str]] = None,
        comprehend_endpoints: Optional[EndpointsConfig] = None,
        extract_endpoints: Optional[EndpointsConfig] = None,
        max_tries: int = 3,
        base_delay: float = 0.25,
        max_delay: float = 10.0,
        retry_config: Optional[RetryPolicyConfigs] = None,
        timeout_ms: Optional[int] = None,
        debug_logger: Any = None,
    ) -> None:
        super().__init__(
            max_tries=max_tries,
            base_delay=base_delay,
            max_delay=max_delay,
            retry_config=retry_config
        )

        if api_key is None:
            api_key = os.getenv("KG_API_KEY")

        if configs is not None:
            self.configs = configs
        else:
            servers_configs = ServersConfigs()
            server_alias = self.server_alias

            if server_url is not None:
                servers_configs.servers[server_alias] = server_url

            base_headers = self._get_base_headers(servers_configs, api_key)
            if server_base_headers:
                base_headers.update(server_base_headers)

            servers_configs.base_headers[server_alias] = base_headers
            
            self.configs = SDKConfigs(
                client=client,
                client_supplied=client is not None,
                async_client=async_client,
                async_client_supplied=async_client is not None,
                debug_logger=debug_logger,
                server=server_alias,
                server_config=servers_configs,
                retry_config=self.retry_config,
                timeout_ms=timeout_ms,
            )

        self._init_apis(
            comprehend_cfg=self.configs,
            extract_cfg=self.configs,
            comprehend_endpoints=comprehend_endpoints,
            extract_endpoints=extract_endpoints,
        )

    def _get_base_headers(
        self, 
        servers_configs: ServersConfigs, 
        api_key: Optional[str]
    ) -> Dict[str, str]:
        """Generate base headers with proper API key."""
        base_headers: Dict[str, str] = {}
        if api_key:
            base_headers["KnowledgatorAPI-Key"] = api_key
        if servers_configs.base_headers:
            base_headers.update(servers_configs.base_headers)
        return base_headers


class KnowledgatorRapidAPI(BaseKnowledgator):
    """RapidAPI variant – separate hosts / headers for each low‑level API."""

    COMPREHEND_ALIAS = "comprehend-it-rapidapi"
    EXTRACT_ALIAS = "extract-it-rapidapi"

    def __init__(
        self,
        *,
        api_key: str,
        configs: Optional[SDKConfigs] = None,
        comprehend_host: Optional[str] = None,
        extract_host: Optional[str] = None,
        comprehend_endpoints: Optional[EndpointsConfig] = None,
        extract_endpoints: Optional[EndpointsConfig] = None,
        client: Optional[Any] = None,
        async_client: Optional[Any] = None,
        timeout_ms: Optional[int] = None,
        debug_logger: Any = None,
        **retry_kwargs,
    ) -> None:
        super().__init__(**retry_kwargs)

        servers_configs = ServersConfigs()

        comprehend_host = comprehend_host or SERVERS[self.COMPREHEND_ALIAS]
        servers_configs.servers[self.COMPREHEND_ALIAS] = f"https://{comprehend_host}"
        servers_configs.base_headers[self.COMPREHEND_ALIAS] = {  # type: ignore[attr-defined]
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": comprehend_host,
            "Content-Type": "application/json",
        }

        extract_host = extract_host or SERVERS[self.EXTRACT_ALIAS]
        servers_configs.servers[self.EXTRACT_ALIAS] = f"https://{extract_host}"
        servers_configs.base_headers[self.EXTRACT_ALIAS] = {  # type: ignore[attr-defined]
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": extract_host,
            "Content-Type": "application/json",
        }

        if configs is None:
            common_configs = SDKConfigs(
                client=client,
                client_supplied=client is not None,
                async_client=async_client,
                async_client_supplied=async_client is not None,
                debug_logger=debug_logger,
                server=self.COMPREHEND_ALIAS,  # default – doesn't really matter
                server_config=servers_configs,
                retry_config=self.retry_config,
                timeout_ms=timeout_ms,
            )
        else:
            common_configs = configs

        self.configs = common_configs

        comprehend_cfg = common_configs.copy(server = self.COMPREHEND_ALIAS)
        extract_cfg = common_configs.copy(server = self.EXTRACT_ALIAS)

        self._init_apis(
            comprehend_cfg=comprehend_cfg,
            extract_cfg=extract_cfg,
            comprehend_endpoints=comprehend_endpoints,
            extract_endpoints=extract_endpoints,
        )