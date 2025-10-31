"""Not Found (404) error."""
from ..core.api_error import ApiError
from typing import Any, Dict, Optional


class NotFoundError(ApiError):
    """Exception raised for HTTP 404 Not Found responses.
    
    This indicates the requested resource does not exist.
    """
    
    def __init__(
        self,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
    ):
        """Initialize a NotFoundError.
        
        Args:
            body: Response body content (error details)
            headers: Response headers
            request_id: Optional request ID for debugging
        """
        super().__init__(
            status_code=404,
            body=body,
            headers=headers,
            request_id=request_id
        )
