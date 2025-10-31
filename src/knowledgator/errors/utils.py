"""Utility functions for error handling."""
from __future__ import annotations
import httpx
from typing import Any, Dict, Optional, Type

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


# Mapping of HTTP status codes to specific error classes
STATUS_CODE_TO_ERROR: Dict[int, Type[ApiError]] = {
    400: BadRequestError,
    401: UnauthorizedError,
    403: ForbiddenError,
    404: NotFoundError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: TooManyRequestsError,
    500: InternalServerError,
    503: ServiceUnavailableError,
}


def raise_for_status(
    response: httpx.Response,
    *,
    body: Optional[Any] = None,
) -> None:
    """Raise an appropriate error based on the HTTP status code.
    
    This function checks the response status code and raises the corresponding
    error class. If no specific error class is mapped, it raises a generic ApiError.
    
    Args:
        response: The httpx Response object
        body: Optional parsed response body (if already parsed)
        
    Raises:
        ApiError: If the response status code indicates an error (>= 400)
    """
    if response.is_success:
        return
    
    # Parse body if not provided
    if body is None:
        try:
            body = response.json()
        except Exception:
            # If JSON parsing fails, use text content
            body = response.text or f"HTTP {response.status_code}"
    
    # Extract headers
    headers = dict(response.headers)
    
    # Get request ID from headers (common patterns)
    request_id = (
        headers.get("x-request-id") or 
        headers.get("request-id") or 
        headers.get("x-trace-id")
    )
    
    # Get the specific error class for this status code
    error_class = STATUS_CODE_TO_ERROR.get(response.status_code, ApiError)
    
    # Raise the appropriate error
    raise error_class(
        body=body,
        headers=headers,
        request_id=request_id,
    )


def get_error_message(response: httpx.Response) -> str:
    """Extract a human-readable error message from the response.
    
    Args:
        response: The httpx Response object
        
    Returns:
        A formatted error message string
    """
    try:
        data = response.json()
        # Common error message field patterns
        if isinstance(data, dict):
            for key in ["message", "error", "detail", "error_message", "msg"]:
                if key in data:
                    return str(data[key])
            # If no standard field, return the whole dict as string
            return str(data)
        return str(data)
    except Exception:
        # Fallback to text if JSON parsing fails
        return response.text or f"HTTP {response.status_code}"
