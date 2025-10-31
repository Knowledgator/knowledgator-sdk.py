"""Unauthorized (401) error."""
from ..core.api_error import ApiError
from typing import Any, Dict, Optional


class UnauthorizedError(ApiError):
    """Exception raised for HTTP 401 Unauthorized responses.
    
    This indicates authentication is required or credentials are invalid.
    """
    
    def __init__(
        self,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
    ):
        """Initialize an UnauthorizedError.
        
        Args:
            body: Response body content (error details)
            headers: Response headers
            request_id: Optional request ID for debugging
        """
        super().__init__(
            status_code=401,
            body=body,
            headers=headers,
            request_id=request_id
        )
