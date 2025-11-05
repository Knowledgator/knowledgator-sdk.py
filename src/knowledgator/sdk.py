from __future__ import annotations

from typing import Optional, Any, Dict
import os
import inspect

from .apis import ComprehendAPI, ExtractAPI, EndpointsConfig, GatewayAPI, TrainingAPI, ModelsAPI, DatasetsAPI, InferenceAPI
from .configs import (
    SDKConfigs,
    ServersConfigs,
    RetryPolicyConfigs,
    SERVERS
)

class BaseKnowledgator:
    comprehend: ComprehendAPI
    extract: ExtractAPI
    gateway: GatewayAPI
    training: TrainingAPI
    models: ModelsAPI
    datasets: DatasetsAPI
    inference: InferenceAPI
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
        gateway_cfg: Optional[SDKConfigs] = None,
        gateway_endpoints: Optional[EndpointsConfig] = None,
        training_cfg: Optional[SDKConfigs] = None,
        training_endpoints: Optional[EndpointsConfig] = None,
        models_cfg: Optional[SDKConfigs] = None,
        models_endpoints: Optional[EndpointsConfig] = None,
        datasets_cfg: Optional[SDKConfigs] = None,
        datasets_endpoints: Optional[EndpointsConfig] = None,
        inference_cfg: Optional[SDKConfigs] = None,
        inference_endpoints: Optional[EndpointsConfig] = None,
    ) -> None:
        """Instantiate low‑level API helpers with custom endpoint configs."""
        self.comprehend = ComprehendAPI(
            comprehend_cfg, endpoints_config=comprehend_endpoints
        )
        self.extract = ExtractAPI(extract_cfg, endpoints_config=extract_endpoints)
        # Gateway is a generic resolver for inference/service endpoints
        if gateway_cfg is None:
            gateway_cfg = comprehend_cfg
        self.gateway = GatewayAPI(gateway_cfg, endpoints_config=gateway_endpoints)
        # Training API for fine-tuning
        if training_cfg is None:
            training_cfg = comprehend_cfg
        self.training = TrainingAPI(training_cfg, endpoints_config=training_endpoints)
        # Models API for managing base models
        if models_cfg is None:
            models_cfg = comprehend_cfg
        self.models = ModelsAPI(models_cfg, endpoints_config=models_endpoints)
        # Datasets API for managing datasets
        if datasets_cfg is None:
            datasets_cfg = comprehend_cfg
        self.datasets = DatasetsAPI(datasets_cfg, endpoints_config=datasets_endpoints)
        # Inference / predict API
        if inference_cfg is None:
            inference_cfg = comprehend_cfg
        self.inference = InferenceAPI(inference_cfg, endpoints_config=inference_endpoints)

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
            and not self._is_closed(client)          # helper from previous fix
        ):
            result = client.aclose()                 # call it first
            if inspect.isawaitable(result):          # only await real coroutines
                await result

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
        server_url: Optional[str] = os.getenv("KG_SERVER_URL"),
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
            
            # Resolve server_url: prefer explicit param, then KG_SERVER_URL env var
            env_server = os.getenv("KG_SERVER_URL")
            resolved_server_url = server_url or env_server
            
            # pdb.set_trace()
            if resolved_server_url is not None:
                servers_configs.servers[server_alias] = resolved_server_url
            # pdb.set_trace()
            # Build base headers with runtime API key (KnowledgatorAPI-Key)
            base_headers = self._get_base_headers(servers_configs, server_alias, api_key)

            # If an environment bearer token is present, include it automatically
            env_bearer = os.getenv("KG_BEARER")
            if env_bearer:
                # don't override existing Authorization set by caller; prefer existing
                base_headers.setdefault("Authorization", f"Bearer {env_bearer}")

            # User-supplied server_base_headers override any defaults
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
            gateway_cfg=self.configs,
            gateway_endpoints=None,
            training_cfg=self.configs,
            training_endpoints=None,
            models_cfg=self.configs,
            models_endpoints=None,
            datasets_cfg=self.configs,
            datasets_endpoints=None,
            inference_cfg=self.configs,
            inference_endpoints=None,
        )

    def _get_base_headers(
        self, 
        servers_configs: ServersConfigs,
        server_alias: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate base headers with proper API key."""
        base_headers: Dict[str, str] = {}
        if api_key:
            base_headers["KnowledgatorAPI-Key"] = api_key
        if servers_configs.base_headers:
            base_headers.update(servers_configs.base_headers.get(server_alias, {}))
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
            gateway_cfg=common_configs,
            gateway_endpoints=None,
            training_cfg=common_configs,
            training_endpoints=None,
            models_cfg=common_configs,
            models_endpoints=None,
            datasets_cfg=common_configs,
            datasets_endpoints=None,
            inference_cfg=common_configs,
            inference_endpoints=None,
        )