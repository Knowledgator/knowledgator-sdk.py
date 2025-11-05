"""Inference / prediction API wrapper.

Provides a small wrapper around the generic /predict/{model_id}/ endpoint.
This file implements a synchronous and asynchronous `predict` method which
POSTs JSON to the server and returns the parsed JSON response, raising
SDK-level errors on non-success responses.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional
import httpx

from .base import BaseAPI
from ..errors import (
    raise_for_status,
    InternalServerError,
    TransportTimeout,
    ConnectionFailure,
)


class InferenceAPI(BaseAPI):
    """Wrapper for model inference endpoints.

    Usage:
        res = api.predict(model_id="comprehend-it-base", json={"text": "..."})
    """

    def predict(self, *, model_id: str, json: Mapping[str, Any], headers: Optional[Mapping[str, str]] = None) -> Any:
        """Synchronously call POST /predict/{model_id}/ with a JSON body.

        Args:
            model_id: Model identifier to route the prediction to
            json: JSON-serializable payload for the model
            headers: Optional headers to merge with base headers

        Returns:
            Parsed JSON response from the server

        Raises:
            SDKError-subclasses for non-2xx responses or parsing failures
        """
        request = self._make_request(
            method="POST",
            path=f"/predict/{model_id}/",
            json_body=json,
            headers=headers,
            accept="application/json",
        )
        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            # Map transport timeout to SDK TransportTimeout for consistency
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            # Generic transport-level exception
            raise ConnectionFailure("Transport error") from exc

        if not response.is_success:
            raise_for_status(response)

        try:
            parsed = response.json()
            # Some inference endpoints return an envelope {"data": {...}}.
            # Unwrap and return the inner `data` dict for convenience.
            if isinstance(parsed, dict) and "data" in parsed:
                return parsed
            return parsed
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc

    async def predict_async(self, *, model_id: str, json: Mapping[str, Any], headers: Optional[Mapping[str, str]] = None) -> Any:
        """Async variant of `predict`.
        """
        request = self._make_request(
            method="POST",
            path=f"/predict/{model_id}/",
            json_body=json,
            headers=headers,
            accept="application/json",
        )
        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if not response.is_success:
            raise_for_status(response)

        try:
            parsed = response.json()
            if isinstance(parsed, dict) and "data" in parsed:
                return parsed
            return parsed
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc
