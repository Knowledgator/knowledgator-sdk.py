"""Base API error class for all SDK exceptions."""
from __future__ import annotations
from typing import Any, Dict, Optional


class ApiError(Exception):
    """Base exception for all API-related errors.
    
    This is the root exception class from which all other SDK exceptions inherit.
    It includes context about the HTTP response, including status code, response body,
    headers, and an optional request ID for debugging.
    
    Attributes:
        status_code: HTTP status code from the response (if applicable)
        body: Response body content (typically error message or JSON)
        headers: HTTP response headers
        request_id: Optional request ID for tracking/debugging
    """
    
    def __init__(
        self,
        *,
        body: Optional[Any] = None,
        status_code: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
    ):
        """Initialize an ApiError.
        
        Args:
            body: Response body content (error message, JSON object, etc.)
            status_code: HTTP status code
            headers: Response headers
            request_id: Optional request ID for debugging
        """
        super().__init__(body)
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        self.request_id = request_id or self.headers.get("x-request-id")
    
    def __str__(self) -> str:
        """Return a formatted string representation of the error."""
        parts = []
        
        # Add main error body/message
        if self.body:
            parts.append(str(self.body))
        
        # Add status code if available
        if self.status_code is not None:
            parts.append(f"[HTTP {self.status_code}]")
        
        # Add request ID if available
        if self.request_id:
            parts.append(f"(request_id={self.request_id})")
        
        return " ".join(parts) if parts else "API Error"
    
    def __repr__(self) -> str:
        """Return a detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"status_code={self.status_code!r}, "
            f"body={self.body!r}, "
            f"request_id={self.request_id!r}"
            ")"
        )
