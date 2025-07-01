from __future__ import annotations

from typing import Optional, Any, Dict
import os

from .apis import ComprehendAPI, ExtractAPI
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

    def _init_apis(self) -> None:
        """Initialize API clients after configs are set."""
        self.comprehend = ComprehendAPI(self.configs)
        self.extract = ExtractAPI(self.configs)

    def close(self) -> None:
        """Close the underlying synchronous client."""
        if (
            self.configs.client
            and not self.configs.client_supplied
            and not self.configs.client.is_closed
        ):
            self.configs.client.close()

    async def aclose(self) -> None:
        """Close the underlying async client."""
        if (
            self.configs.async_client
            and not self.configs.async_client_supplied
            and not self.configs.async_client.is_closed
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
            servers_cfg = ServersConfigs()
            server_alias = self.server_alias

            if server_url is not None:
                servers_cfg.servers[server_alias] = server_url

            base_headers = self._get_base_headers(servers_cfg, api_key)
            if server_base_headers:
                base_headers.update(server_base_headers)

            self.configs = SDKConfigs(
                client=client,
                client_supplied=client is not None,
                async_client=async_client,
                async_client_supplied=async_client is not None,
                debug_logger=debug_logger,
                server=server_alias,
                server_config=servers_cfg,
                retry_config=self.retry_config,
                timeout_ms=timeout_ms,
            )

        # Initialize API clients
        self._init_apis()

    def _get_base_headers(
        self, 
        servers_cfg: ServersConfigs, 
        api_key: Optional[str]
    ) -> Dict[str, str]:
        """Generate base headers with proper API key."""
        base_headers: Dict[str, str] = {}
        if api_key:
            base_headers["KnowledgatorAPI-Key"] = api_key
        if servers_cfg.base_headers:
            base_headers.update(servers_cfg.base_headers)
        return base_headers


class KnowledgatorRapidAPI(BaseKnowledgator):
    """
    Knowledgator implementation for RapidAPI with separate hosts for each API.
    
    >>> rapid_kg = KnowledgatorRapidAPI(
    >>>     api_key="your-rapidapi-key",
    >>>     comprehend_host="comprehend-it.p.rapidapi.com",
    >>>     extract_host="extract-it.p.rapidapi.com"
    >>> )
    """
    COMPREHEND_SERVER_ALIAS = "comprehend-rapidapi"
    EXTRACT_SERVER_ALIAS = "extract-rapidapi"

    def __init__(
        self,
        *,
        api_key: str,
        comprehend_host: Optional[str] = None,
        extract_host: Optional[str] = None,
        client: Optional[Any] = None,
        async_client: Optional[Any] = None,
        max_tries: int = 3,
        base_delay: float = 0.25,
        max_delay: float = 10.0,
        retry_config: Optional[RetryPolicyConfigs] = None,
        timeout_ms: Optional[int] = None,
        debug_logger: Any = None,
    ):
        super().__init__(
            max_tries=max_tries,
            base_delay=base_delay,
            max_delay=max_delay,
            retry_config=retry_config
        )

        servers_cfg = ServersConfigs()

        if comprehend_host is None:
            comprehend_host = SERVERS[self.COMPREHEND_SERVER_ALIAS]

        servers_cfg.servers[self.COMPREHEND_SERVER_ALIAS] = f"https://{comprehend_host}"
        comprehend_headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": comprehend_host,
            "Content-Type": "application/json"
        }
        
        if extract_host is None:
            extract_host = SERVERS[self.COMPREHEND_SERVER_ALIAS]

        servers_cfg.servers[self.EXTRACT_SERVER_ALIAS] = f"https://{extract_host}"
        extract_headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": extract_host,
            "Content-Type": "application/json"
        }

        self.configs = SDKConfigs(
            client=client,
            client_supplied=client is not None,
            async_client=async_client,
            async_client_supplied=async_client is not None,
            debug_logger=debug_logger,
            server=self.COMPREHEND_SERVER_ALIAS,  # Default
            server_config=servers_cfg,
            retry_config=self.retry_config,
            timeout_ms=timeout_ms,
        )

        self._init_apis(comprehend_headers, extract_headers)

    def _init_apis(
        self, 
        comprehend_headers: Dict[str, str], 
        extract_headers: Dict[str, str]
    ) -> None:
        """Initialize API clients with custom server configurations."""
        comprehend_config = self.configs.copy(
            update={"server": self.COMPREHEND_SERVER_ALIAS}
        )
        merged_comprehend_headers = {**(comprehend_config.server_config.base_headers or {}), **comprehend_headers}
        comprehend_config.server_config.base_headers = merged_comprehend_headers
        
        extract_config = self.configs.copy(
            update={"server": self.EXTRACT_SERVER_ALIAS}
        )
        merged_extract_headers = {**(extract_config.server_config.base_headers or {}), **extract_headers}
        extract_config.server_config.base_headers = merged_extract_headers
        
        self.comprehend = ComprehendAPI(comprehend_config, path = '/')
        self.extract = ExtractAPI(extract_config, path = '/')