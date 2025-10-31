"""SDK error classes with HTTP status-code-specific exceptions.

This module provides a hierarchy of error classes for handling different types of API errors.
All errors inherit from the base ApiError class in the core module.

Error Hierarchy (New):
    ApiError (base)
    ├── BadRequestError (400)
    ├── UnauthorizedError (401)
    ├── ForbiddenError (403)
    ├── NotFoundError (404)
    ├── ConflictError (409)
    ├── UnprocessableEntityError (422)
    ├── TooManyRequestsError (429)
    ├── InternalServerError (500)
    └── ServiceUnavailableError (503)

Legacy Exceptions (Deprecated):
    SDKError, ServerError, TransportTimeout, ConnectionFailure, etc.
    These are kept for backward compatibility but new code should use
    the status-specific errors above.
"""

# New error hierarchy (recommended for new code)
from ..core.api_error import ApiError
from .bad_request_error import BadRequestError
from .unauthorized_error import UnauthorizedError
from .forbidden_error import ForbiddenError
from .not_found_error import NotFoundError
from .conflict_error import ConflictError
from .unprocessable_entity_error import UnprocessableEntityError
from .too_many_requests_error import TooManyRequestsError
from .internal_server_error import InternalServerError
from .service_unavailable_error import ServiceUnavailableError
from .utils import raise_for_status, get_error_message

# Legacy exceptions (for backward compatibility)
from .legacy import (
    SDKError,
    ConfigurationError,
    ValidationError,
    NetworkError,
    TransportTimeout,
    ConnectionFailure,
    APIError as LegacyAPIError,
    ClientError,
    ServerError,
    from_http_response,
)

__all__ = [
    # New error hierarchy (recommended)
    "ApiError",
    "BadRequestError",
    "UnauthorizedError",
    "ForbiddenError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "TooManyRequestsError",
    "InternalServerError",
    "ServiceUnavailableError",
    # Utilities
    "raise_for_status",
    "get_error_message",
    # Legacy exceptions (deprecated but supported)
    "SDKError",
    "ConfigurationError",
    "ValidationError",
    "NetworkError",
    "TransportTimeout",
    "ConnectionFailure",
    "ClientError",
    "ServerError",
    "from_http_response",
]

