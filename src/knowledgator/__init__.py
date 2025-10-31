from .sdk import Knowledgator, KnowledgatorRapidAPI

# Core error classes (new hierarchy)
from .core.api_error import ApiError
from .errors import (
    BadRequestError,
    UnauthorizedError,
    ForbiddenError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
    TooManyRequestsError,
    InternalServerError,
    ServiceUnavailableError,
    # Legacy exceptions (backward compatibility)
    SDKError,
    ServerError,
    TransportTimeout,
    ConnectionFailure,
)

__all__ = [
    # Main SDK classes
    "Knowledgator",
    "KnowledgatorRapidAPI",
    # Core error
    "ApiError",
    # 4xx Client Errors
    "BadRequestError",
    "UnauthorizedError",
    "ForbiddenError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "TooManyRequestsError",
    # 5xx Server Errors
    "InternalServerError",
    "ServiceUnavailableError",
    # Legacy exceptions
    "SDKError",
    "ServerError",
    "TransportTimeout",
    "ConnectionFailure",
]