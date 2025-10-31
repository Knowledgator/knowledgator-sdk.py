"""Conflict (409) error."""
from ..core.api_error import ApiError
from typing import Any, Dict, Optional


class ConflictError(ApiError):
    """Exception raised for HTTP 409 Conflict responses.
    
    This indicates a request conflict with the current state of the resource.
    """
    
    def __init__(
        self,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
    ):
        """Initialize a ConflictError.
        
        Args:
            body: Response body content (error details)
            headers: Response headers
            request_id: Optional request ID for debugging
        """
        super().__init__(
            status_code=409,
            body=body,
            headers=headers,
            request_id=request_id
        )
