"""Unprocessable Entity (422) error."""
from ..core.api_error import ApiError
from typing import Any, Dict, Optional


class UnprocessableEntityError(ApiError):
    """Exception raised for HTTP 422 Unprocessable Entity responses.
    
    This indicates the request was well-formed but contains semantic errors.
    """
    
    def __init__(
        self,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
    ):
        """Initialize an UnprocessableEntityError.
        
        Args:
            body: Response body content (error details)
            headers: Response headers
            request_id: Optional request ID for debugging
        """
        super().__init__(
            status_code=422,
            body=body,
            headers=headers,
            request_id=request_id
        )
