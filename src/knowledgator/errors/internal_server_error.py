"""Internal Server Error (500) error."""
from ..core.api_error import ApiError
from typing import Any, Dict, Optional


class InternalServerError(ApiError):
    """Exception raised for HTTP 500 Internal Server Error responses.
    
    This indicates an unexpected error occurred on the server.
    """
    
    def __init__(
        self,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
    ):
        """Initialize an InternalServerError.
        
        Args:
            body: Response body content (error details)
            headers: Response headers
            request_id: Optional request ID for debugging
        """
        super().__init__(
            status_code=500,
            body=body,
            headers=headers,
            request_id=request_id
        )
