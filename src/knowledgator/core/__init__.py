"""Core SDK abstractions and utilities.

This module contains the foundational components used throughout the SDK:
- Error handling (ApiError and hierarchy)
"""

from .api_error import ApiError

__all__ = [
    "ApiError",
]
