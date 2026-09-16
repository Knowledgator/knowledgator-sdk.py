from __future__ import annotations

from typing import Any, Dict, Type


class SDKError(Exception):
    """
    Root of *all* public exceptions raised by the client SDK.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        payload: Dict[str, Any] | None = None,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}
        self.request_id = request_id

    # Nice repr/str for logs
    def __str__(self) -> str:  # pragma: no cover
        base = super().__str__()
        extra = f" [{self.status_code}]" if self.status_code else ""
        rid = f" (request_id={self.request_id})" if self.request_id else ""
        payload_str = f" payload={self.payload}" if self.payload else ""
        return f"{base}{extra}{rid}{payload_str}"


class ConfigurationError(SDKError):
    """Bad env vars, missing tokens, etc."""


class ValidationError(SDKError):
    """Local validation failed before a network request was made."""


class NetworkError(SDKError):
    """Base for transport-level problems."""


class TransportTimeout(NetworkError):
    pass


class ConnectionFailure(NetworkError):
    pass


class APIError(SDKError):
    """
    Base for errors originating from the remote service.

    All subclasses have `status_code` set.
    """


class ClientError(APIError):
    """4xx family."""


class ServerError(APIError):
    """5xx family."""


_STATUS_MAP: dict[int, Type[APIError]] = {
    cls.code: cls
    for cls in APIError.__subclasses__()  # direct
    for cls in cls.__subclasses__()  # one level deeper (Client/Server)
}
