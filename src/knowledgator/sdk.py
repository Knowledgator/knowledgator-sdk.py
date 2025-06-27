from __future__ import annotations

from typing import Optional, Any

from .apis import ComprehendAPI, ExtractAPI
from .configs import (
    SDKConfigs,
    ServersConfigs,
    RetryPolicyConfigs,
)


class Knowledgator:
    """
    High-level façade that bundles all public Knowledgator API helpers
    (classification, extraction, …) behind a single object.

    >>> kg = Knowledgator(api_key="sk-…")
    >>> res = kg.comprehend.run("Kyiv is beautiful", ["geography", "sport"])
    """

    comprehend: ComprehendAPI
    extract: ExtractAPI
    configs: SDKConfigs


    def __init__(
        self,
        *,
        configs: Optional[SDKConfigs] = None,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,          # httpx.Client
        async_client: Optional[Any] = None,    # httpx.AsyncClient
        server_url: Optional[str] = None,
        retry_config: Optional[RetryPolicyConfigs] = None,
        timeout_ms: Optional[int] = None,
        debug_logger: Any = None,
    ) -> None:

        servers_cfg: ServersConfigs
        server_alias: str

        if configs is not None:
            self.configs = configs
            servers_cfg = self.configs.server_config  # type: ignore
            server_alias = self.configs.server or servers_cfg.base_server
        else:
            servers_cfg = ServersConfigs()
            server_alias = servers_cfg.base_server

            if server_url is not None:
                server_alias = "__custom__"
                servers_cfg.servers[server_alias] = server_url

            base_headers: dict[str, str] = {}
            if api_key:
                base_headers["Authorization"] = f"Bearer {api_key}"
            if servers_cfg.base_headers:
                base_headers |= servers_cfg.base_headers     

            servers_cfg.base_headers = base_headers or None

            self.configs = SDKConfigs(
                client=client,
                client_supplied=client is not None,
                async_client=async_client,
                async_client_supplied=async_client is not None,
                debug_logger=debug_logger,
                server=server_alias,
                server_config=servers_cfg,
                retry_config=retry_config,
                timeout_ms=timeout_ms,
            )

        self.comprehend = ComprehendAPI(self.configs)
        self.extract = ExtractAPI(self.configs)


    def close(self) -> None:
        """Close the underlying *synchronous* client (if we created it)."""
        if (
            self.configs.client
            and not self.configs.client_supplied
            and not self.configs.client.is_closed
        ):
            self.configs.client.close()

    async def aclose(self) -> None:
        """Close the underlying *async* client (if we created it)."""
        if (
            self.configs.async_client
            and not self.configs.async_client_supplied
            and not self.configs.async_client.is_closed
        ):  # type: ignore[attr-defined]
            await self.configs.async_client.aclose()  # type: ignore[attr-defined]

    async def __aenter__(self) -> "Knowledgator":  # noqa: D401
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        await self.aclose()

    def __enter__(self) -> "Knowledgator":  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.close()
