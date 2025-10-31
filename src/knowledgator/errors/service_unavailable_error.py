"""Service Unavailable (503) error."""
from ..core.api_error import ApiError
from typing import Any, Dict, Optional


class ServiceUnavailableError(ApiError):
    """Exception raised for HTTP 503 Service Unavailable responses.
    
    This indicates the server is temporarily unable to handle the request.
    """
    
    def __init__(
        self,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
    ):
        """Initialize a ServiceUnavailableError.
        
        Args:
            body: Response body content (error details)
            headers: Response headers
            request_id: Optional request ID for debugging
        """
        super().__init__(
            status_code=503,
            body=body,
            headers=headers,
            request_id=request_id
        )
