from dataclasses import dataclass

@dataclass
class HttpConfig:
    initial_retry_delay_seconds = 0.5
    max_retry_delay_seconds = 10
    max_retries = 10