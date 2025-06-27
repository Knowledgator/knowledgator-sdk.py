from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass, field
from pydantic import Field
import random
import os

from .clients import HttpClient, AsyncHttpClient
from .version import __version__

SERVERS = {"main": "https://knowledgator.com"}

@dataclass(slots=True)
class RetryPolicyConfigs:
    max_tries: int = 3
    base_delay: float = 0.25
    max_delay: float = 10.0
    retryable_statuses: tuple[int, ...] = (408, 409, 429)
    jitter: bool = True

    def should_retry(self, status: int) -> bool:
        return status >= 500 or status in self.retryable_statuses

    def backoff(self, attempt: int) -> float:
        delay = min(self.base_delay * 2**attempt, self.max_delay)
        if self.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay


def _default_servers() -> Dict[str, str]:
    if env := os.getenv("KNOWLEDGATOR_SERVERS"):
        try:
            import json
            return json.loads(env)
        except (ValueError, TypeError):
            pass  
    return SERVERS

@dataclass(slots=True)
class ServersConfigs:
    base_server: str = "main"
    servers: Dict[str, str] = field(default_factory=_default_servers)
    base_headers: Dict[str, str] = None

    def __post_init__(self) -> None:
        if self.base_server not in self.servers:
            raise ValueError(
                f"base_server '{self.base_server}' is not present in servers mapping"
            )

    def url(self, name: str | None = None) -> str:
        """Return the URL by alias; if None uses `base_server`."""
        return self.servers[name or self.base_server]


@dataclass
class SDKConfigs:
    client: Union[HttpClient, None]
    client_supplied: bool
    async_client: Union[AsyncHttpClient, None]
    async_client_supplied: bool
    debug_logger: Any #Logger
    server: Union[str, None] = 'main'
    server_config: Union[ServersConfigs, None]
    language: str = "python"
    sdk_version: str = __version__
    retry_config: Optional[RetryPolicyConfigs] = Field(default_factory=lambda: None)
    timeout_ms: Optional[int] = None

    def get_server_details(self) -> Tuple[str, Dict[str, str]]:
        if self.server_config is None:
            raise ValueError("Server configuration is not set.")
        server_url = self.server_config.url(self.server)
        headers = self.server_config.base_headers or {}
        return server_url, headers

    