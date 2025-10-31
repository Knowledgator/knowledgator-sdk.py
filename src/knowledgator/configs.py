from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass, field, replace
from pydantic import Field
import platform
import random
import os

from .version import __version__
from .utils.logger import Logger

SERVERS = {
    "main": "api.knowledgator.com",
    "comprehend-it-rapidapi": "comprehend-it.p.rapidapi.com",
    "extract-it-rapidapi": "extract-it.p.rapidapi.com",
}


DEFAULT_UA = (
    lambda: f"knowledgator-python/{__version__} "
            f"({platform.system()} {platform.release()}; "
            f"{platform.machine()})"
)

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
    base_headers: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.base_server not in self.servers:
            raise ValueError(
                f"base_server '{self.base_server}' is not present in servers mapping"
            )
        self.base_headers.setdefault(self.base_server, {})

    def url(self, name: Optional[str] = None) -> str:
        """Return a fully-qualified URL for the given server alias.

        Falls back to the ``base_server`` mapping if the alias is unknown.
        Ensures a proper scheme (https://) when only a hostname is provided.
        """
        key = name or self.base_server
        # Prefer explicit mapping, then fallback to base_server mapping
        raw = self.servers.get(key) or self.servers.get(self.base_server) or ""
        
        # If nothing is configured at all, raise an explicit error
        if not raw:
            raise ValueError(f"No server URL configured for alias '{key}' or base '{self.base_server}'")

        # Normalize into a fully-qualified URL
        if not (raw.startswith("http://") or raw.startswith("https://")):
            raw = f"https://{raw}"
        return raw

    def headers(self, name: Optional[str] = None) -> Dict[str, str]:
        """Return headers for a server alias, defaulting to ``base_server``.

        Always returns a dict (possibly empty).
        """
        key = name or self.base_server
        return (self.base_headers or {}).get(key, {})


@dataclass(slots=True)
class SDKConfigs:
    client: Any | None = None
    client_supplied: bool = False
    async_client: Any | None = None
    async_client_supplied: bool = False

    debug_logger: Any = field(default_factory=Logger)     
    language: str = "python"
    sdk_version: str = __version__
    user_agent: str = field(default_factory=DEFAULT_UA)

    server: str = "main"
    server_config: "ServersConfigs" = field(default_factory=ServersConfigs)
    retry_config: "RetryPolicyConfigs" = field(default_factory=RetryPolicyConfigs)
    timeout_ms: Optional[int] = None                    

    def copy(self, **overrides) -> "SDKConfigs":
        """
        Return a *shallow* copy of this SDKConfigs instance, with optional
        field overrides::

            new_cfg = cfg.copy(timeout_ms=5_000, server="comprehend-it-rapidapi")

        Any keyword you supply will replace the corresponding attribute
        on the new instance, leaving the original untouched.
        """
        return replace(self, **overrides)
    
    def get_server_details(self) -> Tuple[str, Dict[str, str]]:
        """Return the resolved base URL and headers for the active server alias."""
        server_url = self.server_config.url(self.server)
        headers = self.server_config.headers(self.server)
        return server_url, headers