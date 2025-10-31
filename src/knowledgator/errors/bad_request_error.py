"""Bad Request (400) error."""
from ..core.api_error import ApiError
from typing import Any, Dict, Optional


class BadRequestError(ApiError):
    """Exception raised for HTTP 400 Bad Request responses.
    
    This typically indicates the request was malformed or contained invalid parameters.
    """
    
    def __init__(
        self,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
    ):
        """Initialize a BadRequestError.
        
        Args:
            body: Response body content (error details)
            headers: Response headers
            request_id: Optional request ID for debugging
        """
        super().__init__(
            status_code=400,
            body=body,
            headers=headers,
            request_id=request_id
        )
