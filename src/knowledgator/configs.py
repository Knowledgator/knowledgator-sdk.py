from typing import Iterable, Mapping, Union, Dict
from dataclasses import dataclass, field
import random
import os

SERVERS = {"main": "https://knowledgator.com"}

@dataclass(slots=True)
class RetryPolicyConfig:
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
class ServersConfig:
    base_server: str = "main"
    servers: Dict[str, str] = field(default_factory=_default_servers)

    def __post_init__(self) -> None:
        if self.base_server not in self.servers:
            raise ValueError(
                f"base_server '{self.base_server}' is not present in servers mapping"
            )

    def url(self, name: str | None = None) -> str:
        """Return the URL by alias; if None uses `base_server`."""
        return self.servers[name or self.base_server]