"""Too Many Requests (429) error."""
from ..core.api_error import ApiError
from typing import Any, Dict, Optional


class TooManyRequestsError(ApiError):
    """Exception raised for HTTP 429 Too Many Requests responses.
    
    This indicates the client has sent too many requests in a given time period (rate limiting).
    """
    
    def __init__(
        self,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
    ):
        """Initialize a TooManyRequestsError.
        
        Args:
            body: Response body content (error details)
            headers: Response headers
            request_id: Optional request ID for debugging
        """
        super().__init__(
            status_code=429,
            body=body,
            headers=headers,
            request_id=request_id
        )
