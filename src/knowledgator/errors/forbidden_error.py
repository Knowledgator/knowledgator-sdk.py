"""Forbidden (403) error."""
from ..core.api_error import ApiError
from typing import Any, Dict, Optional


class ForbiddenError(ApiError):
    """Exception raised for HTTP 403 Forbidden responses.
    
    This indicates the client does not have permission to access the resource.
    """
    
    def __init__(
        self,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
    ):
        """Initialize a ForbiddenError.
        
        Args:
            body: Response body content (error details)
            headers: Response headers
            request_id: Optional request ID for debugging
        """
        super().__init__(
            status_code=403,
            body=body,
            headers=headers,
            request_id=request_id
        )
