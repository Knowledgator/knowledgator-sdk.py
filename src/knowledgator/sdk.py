from __future__ import annotations

import os
import inspect
from typing import Any, Dict, Optional

from .apis import (
    ExtractAPI,
    DatasetsAPI,
    InstructAPI,
    ComprehendAPI,
    FinetuningAPI,
    OpenModelAPI,
    EndpointsConfig,
)
from .configs import SERVERS, SDKConfigs, ServersConfigs, RetryPolicyConfigs


class BaseKnowledgator:
    comprehend: ComprehendAPI
    extract: ExtractAPI
    instruct: InstructAPI
    datasets: Optional[DatasetsAPI]
    finetuning: Optional[FinetuningAPI]
    configs: SDKConfigs

    servers_cfg: ServersConfigs
    server_alias: str

    def __init__(
        self,
        max_tries: int = 3,
        base_delay: float = 0.25,
        max_delay: float = 10.0,
        retry_config: Optional[RetryPolicyConfigs] = None,
    ):
        if retry_config is None:
            self.retry_config = RetryPolicyConfigs(
                max_tries=max_tries, base_delay=base_delay, max_delay=max_delay
            )
        else:
            self.retry_config = retry_config

    def _init_apis(
        self,
        *,
        comprehend_cfg: SDKConfigs,
        extract_cfg: SDKConfigs,
        instruct_cfg: SDKConfigs,
        datasets_cfg: Optional[SDKConfigs] = None,
        finetuning_cfg: Optional[SDKConfigs] = None,
        comprehend_endpoints: Optional[EndpointsConfig] = None,
        extract_endpoints: Optional[EndpointsConfig] = None,
        instruct_endpoints: Optional[EndpointsConfig] = None,
        datasets_endpoints: Optional[EndpointsConfig] = None,
        finetuning_endpoints: Optional[EndpointsConfig] = None,
    ) -> None:
        """Instantiate low‑level API helpers with custom endpoint configs."""
        self.comprehend = ComprehendAPI(
            comprehend_cfg, endpoints_config=comprehend_endpoints
        )
        self.extract = ExtractAPI(extract_cfg, endpoints_config=extract_endpoints)
        self.instruct = InstructAPI(instruct_cfg, endpoints_config=instruct_endpoints)
        if datasets_cfg is not None:
            self.datasets = DatasetsAPI(
                datasets_cfg, endpoints_config=datasets_endpoints
            )
        else:
            self.datasets = None
        if finetuning_cfg is not None:
            self.finetuning = FinetuningAPI(
                finetuning_cfg, endpoints_config=finetuning_endpoints
            )
        else:
            self.finetuning = None

    @staticmethod
    def _is_closed(obj: Any) -> bool:
        flag = getattr(obj, "is_closed", None)
        return flag if isinstance(flag, bool) else False

    def close(self) -> None:
        """Close the underlying synchronous client."""
        client = self.configs.client
        if client and not self.configs.client_supplied and not self._is_closed(client):
            client.close()

    async def aclose(self) -> None:
        """Close the underlying async client (works with real clients *and* mocks)."""
        client = self.configs.async_client
        if (
            client
            and not self.configs.async_client_supplied
            and not self._is_closed(client)
        ):
            if hasattr(client, "aclose"):
                result = client.aclose()
                if inspect.isawaitable(result):
                    await result
            elif hasattr(client, "close"):
                client.close()

    async def __aenter__(self) -> BaseKnowledgator:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def __enter__(self) -> BaseKnowledgator:
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
        instruct_endpoints: Optional[EndpointsConfig] = None,
        datasets_endpoints: Optional[EndpointsConfig] = None,
        finetuning_endpoints: Optional[EndpointsConfig] = None,
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
            retry_config=retry_config,
        )

        if api_key is None:
            api_key = os.getenv("KG_API_KEY") or os.getenv("KG_MCP_API_KEY") or os.getenv("KNOWLEDGATOR_API_KEY")

        if configs is not None:
            self.configs = configs
        else:
            servers_configs = ServersConfigs()
            server_alias = self.server_alias

            if server_url is not None:
                servers_configs.servers[server_alias] = server_url

            base_headers = self._get_base_headers(
                servers_configs, server_alias, api_key
            )
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
            instruct_cfg=self.configs,
            datasets_cfg=self.configs,
            finetuning_cfg=self.configs,
            comprehend_endpoints=comprehend_endpoints,
            extract_endpoints=extract_endpoints,
            instruct_endpoints=instruct_endpoints,
            datasets_endpoints=datasets_endpoints,
            finetuning_endpoints=finetuning_endpoints,
        )

    def _get_base_headers(
        self,
        servers_configs: ServersConfigs,
        server_alias: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, str]:
        """Generate base headers with proper API key."""
        base_headers: Dict[str, str] = {}
        if api_key:
            base_headers["KnowledgatorAPI-Key"] = api_key
            if not api_key.startswith("Bearer "):
                base_headers["Authorization"] = f"Bearer {api_key}"
            else:
                base_headers["Authorization"] = api_key
        if servers_configs.base_headers:
            base_headers.update(servers_configs.base_headers.get(server_alias, {}))
        return base_headers

    def model(self, model_id: str, method: Optional[str] = None) -> OpenModelAPI:
        """
        Create a client for an Open Model (dynamic model ID).

        :param model_id: The ID of the model to access (e.g. 'my-finetuned-model')
        :param method: Optional method name to use for predictions (e.g. 'generate')
        :return: An initialized OpenModelAPI instance.
        """
        return OpenModelAPI(self.configs, model_id=model_id, method=method)


class KnowledgatorRapidAPI(BaseKnowledgator):
    """RapidAPI variant - separate hosts / headers for each low-level API."""

    COMPREHEND_ALIAS = "comprehend-it-rapidapi"
    EXTRACT_ALIAS = "extract-it-rapidapi"
    INSTRUCT_ALIAS = "instruct-it-rapidapi"

    def __init__(
        self,
        *,
        api_key: str,
        configs: Optional[SDKConfigs] = None,
        comprehend_host: Optional[str] = None,
        extract_host: Optional[str] = None,
        instruct_host: Optional[str] = None,
        comprehend_endpoints: Optional[EndpointsConfig] = None,
        extract_endpoints: Optional[EndpointsConfig] = None,
        instruct_endpoints: Optional[EndpointsConfig] = None,
        datasets_endpoints: Optional[EndpointsConfig] = None,
        finetuning_endpoints: Optional[EndpointsConfig] = None,
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

        instruct_host = instruct_host or SERVERS[self.INSTRUCT_ALIAS]
        servers_configs.servers[self.INSTRUCT_ALIAS] = f"https://{instruct_host}"
        servers_configs.base_headers[self.INSTRUCT_ALIAS] = {  # type: ignore[attr-defined]
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": instruct_host,
            "Content-Type": "application/json",
        }

        # Fallback to main server headers using the same API key
        servers_configs.base_headers["main"] = {
            "KnowledgatorAPI-Key": api_key,
            "Content-Type": "application/json",
        }

        if configs is None:
            common_configs = SDKConfigs(
                client=client,
                client_supplied=client is not None,
                async_client=async_client,
                async_client_supplied=async_client is not None,
                debug_logger=debug_logger,
                server=self.COMPREHEND_ALIAS,  # default - doesn't really matter
                server_config=servers_configs,
                retry_config=self.retry_config,
                timeout_ms=timeout_ms,
            )
        else:
            common_configs = configs

        self.configs = common_configs

        comprehend_cfg = common_configs.copy(server=self.COMPREHEND_ALIAS)
        extract_cfg = common_configs.copy(server=self.EXTRACT_ALIAS)
        instruct_cfg = common_configs.copy(server=self.INSTRUCT_ALIAS)

        self._init_apis(
            comprehend_cfg=comprehend_cfg,
            extract_cfg=extract_cfg,
            instruct_cfg=instruct_cfg,
            # Datasets and Finetuning are not available on RapidAPI
            datasets_cfg=None,
            finetuning_cfg=None,
            comprehend_endpoints=comprehend_endpoints,
            extract_endpoints=extract_endpoints,
            instruct_endpoints=instruct_endpoints,
            datasets_endpoints=datasets_endpoints,
            finetuning_endpoints=finetuning_endpoints,
        )

        # Force unavailable services to raise error on access
        self.datasets = UnavailableRapidAPIService("Datasets API")  # type: ignore
        self.finetuning = UnavailableRapidAPIService("Finetuning API")  # type: ignore


class UnavailableRapidAPIService:
    """Helper to raise error when accessing services not available on RapidAPI."""

    def __init__(self, service_name: str):
        self._service_name = service_name

    def __getattr__(self, name: str):
        raise NotImplementedError(
            f"{self._service_name} is not available via RapidAPI. "
            "Please use the main Knowledgator API for this service."
        )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self._service_name} is not available via RapidAPI. "
            "Please use the main Knowledgator API for this service."
        )
