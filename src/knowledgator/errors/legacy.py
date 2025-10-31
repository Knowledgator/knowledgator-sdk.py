"""Legacy exception classes for backward compatibility.

This module provides the original SDKError-based exception hierarchy
for backward compatibility with existing code. New code should use
the status-specific errors from the main errors module.

DEPRECATED: This module exists for backward compatibility only.
New code should import from knowledgator.errors instead.
"""
from __future__ import annotations
from typing import Any, Dict

import warnings


class SDKError(Exception):
    """Root of all public exceptions raised by the client SDK.
    
    DEPRECATED: Use ApiError or status-specific errors instead.
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

    def __str__(self) -> str:
        base = super().__str__()
        extra = f" [{self.status_code}]" if self.status_code else ""
        rid = f" (request_id={self.request_id})" if self.request_id else ""
        return f"{base}{extra}{rid}"


class ConfigurationError(SDKError):
    """Bad env vars, missing tokens, etc."""


class ValidationError(SDKError):
    """Local validation failed before a network request was made."""


class NetworkError(SDKError):
    """Base for transport-level problems."""


class TransportTimeout(NetworkError):
    """Request timeout error."""


class ConnectionFailure(NetworkError):
    """Connection failure error."""


class APIError(SDKError):
    """Base for errors originating from the remote service.
    
    DEPRECATED: Use the new ApiError (note capitalization) instead.
    """


class ClientError(APIError):
    """4xx family errors."""


class ServerError(APIError):
    """5xx family errors."""


# For backward compatibility with old error mapping
_STATUS_MAP: dict[int, type[APIError]] = {}


def from_http_response(resp) -> APIError:
    """Turn an httpx.Response into the corresponding APIError subclass.
    
    DEPRECATED: Use raise_for_status() from the main errors module instead.
    
    Args:
        resp: HTTP response object with .status_code, .json(), .text, .headers
        
    Returns:
        APIError: Appropriate error subclass based on status code
    """
    warnings.warn(
        "from_http_response() is deprecated. Use raise_for_status() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    cls = _STATUS_MAP.get(resp.status_code)
    if cls is None:
        # Fallback class keeps hierarchy correct
        cls = ClientError if 400 <= resp.status_code < 500 else ServerError

    try:
        payload = resp.json()
    except ValueError:
        payload = {"raw": resp.text}

    return cls(
        f"{resp.status_code} {resp.reason_phrase}",
        status_code=resp.status_code,
        payload=payload,
        request_id=resp.headers.get("x-request-id"),
    )
