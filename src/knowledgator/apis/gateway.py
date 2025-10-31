from typing import Any, Mapping, Optional, List
from dataclasses import dataclass
import httpx

from .base import BaseAPI, EndpointsConfig
from ..models.inputs import ComprehendRequest
from ..models.outputs import ComprehendResults, ComprehendResponse

# Error handling - using consolidated errors module
from ..errors import (
    raise_for_status,
    InternalServerError,
    BadRequestError,
    ServerError,
    TransportTimeout,
    ConnectionFailure,
)

from pydantic import ValidationError as PydanticValidationError


@dataclass
class ComprehendGatewayEndpoints(EndpointsConfig):
    # gateway resolves to /{service_id}/{path} but for the Comprehend service
    # the main path is typically '/'. Keep configurable for tests.
    main: str = ""


class ComprehendGatewayAPI(BaseAPI):
    """Gateway wrapper for Comprehend and instruct-it inference via the generic gateway route.

    This class provides typed, ergonomic `resolve` and `resolve_async` methods
    which call the gateway `/{service_id}/{path}` route. It accepts the same
    body as the Comprehend endpoint and returns Pydantic `ComprehendResults`.
    """

    PATH = '/'
    METHOD = 'POST'
    CONTENT_TYPE = 'application/json'
    ENDPOINTS_CONFIG = ComprehendGatewayEndpoints()
    DEFAULT_THRESHOLD = 0.5

    def _parse_response(
        self,
        raw: dict[str, Any],
        *,
        text_fallback: str,
        labels_fallback: List[str],
        threshold_fallback: Optional[float],
    ) -> ComprehendResults:
        # Gateway may return either 'results' or 'outputs' depending on the
        # upstream service; accept both for robustness.
        outputs = raw.get("results") or raw.get("outputs")
        if not isinstance(outputs, list):
            raise ServerError(
                "Unexpected response shape (missing 'results' or 'outputs')",
                status_code=200,
                payload={"raw": raw},
            )

        try:
            parsed_results = [ComprehendResponse.model_validate(o) for o in outputs]

        except PydanticValidationError as exc:
            raise ServerError(
                "Unexpected record format inside 'results'",
                status_code=200,
                payload={"raw": raw},
            ) from exc

        return ComprehendResults(
            results=parsed_results,
            input_text=raw.get("text", text_fallback),
            labels=raw.get("labels", labels_fallback),
            threshold=raw.get(
                "threshold",
                threshold_fallback if threshold_fallback is not None else self.DEFAULT_THRESHOLD,
            ),
            status=raw.get("status", "ok"),
        )

    def gateway(
        self,
        http_method: str = "POST",
        text: Optional[str] = None,
        labels: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        examples: Optional[List[dict[str, str]]] = None,
        service_id: str = "comprehend-it",
        path: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
        json_body: Optional[Mapping[str, Any]] = None,
    ) -> Optional[ComprehendResults]:
        """Generic gateway caller for arbitrary HTTP methods to /{service_id}/{path}.

        For POST/PUT/PATCH the body will default to the ComprehendRequest built
        from (text, labels, threshold, examples) unless `json_body` is provided.
        For GET/DELETE the request is sent without a JSON body unless
        `json_body` is explicitly supplied. If the response contains a JSON
        payload with 'results' or 'outputs' it will be parsed and returned as
        `ComprehendResults`. If the response is empty (204) or contains no
        parseable outputs, None is returned.
        """

        method = http_method.upper()

        body_payload = None
        if json_body is not None:
            body_payload = json_body
        elif method in ("POST", "PUT", "PATCH"):
            # Build default comprehend request body
            body = ComprehendRequest(
                text=text or "",
                labels=labels or [],
                threshold=threshold,
                examples=examples,
            )
            body_payload = body.model_dump(exclude_none=True)

        # Build gateway path: /{service_id}/{path}
        gateway_path = f"/{service_id}/{path.lstrip('/') if path else ''}" if service_id else f"/{path or ''}"
        if not gateway_path.endswith("/"):
            gateway_path = gateway_path + "/"

        request = self._make_request(
            method=method,
            path=gateway_path,
            url_override=server_url,
            json_body=body_payload,
            headers=http_headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("ComprehendGatewayAPI: %s %s", request.method, request.url)

        try:
            if method in ("GET", "DELETE") and body_payload is None:
                response = self.send(request)
            else:
                response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("ComprehendGatewayAPI: %s ← %s", response.status_code, request.url)

        # Raise for non-success statuses
        if not response.is_success:
            raise_for_status(response)

        # Try to parse JSON; if none or not parseable return None
        try:
            raw: dict[str, Any] = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={
                    "message": "Response is not valid JSON",
                    "raw_text": response.text,
                },
                headers=dict(response.headers),
            ) from exc

        # If payload contains results/outputs try parse into ComprehendResults
        try:
            return self._parse_response(
                raw,
                text_fallback=(text or ""),
                labels_fallback=(labels or []),
                threshold_fallback=threshold,
            )
        except (ValueError, PydanticValidationError) as exc:
            # If no parseable outputs, treat as server error
            raise InternalServerError(
                body={"message": "Malformed or unexpected response", "raw_data": raw},
                headers=dict(response.headers),
            ) from exc

    # Backwards compatible aliases for the previous API
    def resolve(self, *a, **kw):
        return self.gateway(*a, **kw)

    async def resolve_async(
        self,
        text: str,
        labels: List[str],
        threshold: Optional[float] = None,
        examples: Optional[List[dict[str, str]]] = None,
        service_id: str = "comprehend-it",
        path: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ComprehendResults:

        # Reuse the generic gateway implementation for async calls
        return await self.gateway_async(
            http_method="POST",
            text=text,
            labels=labels,
            threshold=threshold,
            examples=examples,
            service_id=service_id,
            path=path,
            server_url=server_url,
            timeout_ms=timeout_ms,
            http_headers=http_headers,
        )

    async def gateway_async(
        self,
        http_method: str = "POST",
        text: Optional[str] = None,
        labels: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        examples: Optional[List[dict[str, str]]] = None,
        service_id: str = "comprehend-it",
        path: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
        json_body: Optional[Mapping[str, Any]] = None,
    ) -> Optional[ComprehendResults]:
        method = http_method.upper()

        body_payload = None
        if json_body is not None:
            body_payload = json_body
        elif method in ("POST", "PUT", "PATCH"):
            body = ComprehendRequest(
                text=text or "",
                labels=labels or [],
                threshold=threshold,
                examples=examples,
            )
            body_payload = body.model_dump(exclude_none=True)

        gateway_path = f"/{service_id}/{path.lstrip('/') if path else ''}" if service_id else f"/{path or ''}"
        if not gateway_path.endswith("/"):
            gateway_path = gateway_path + "/"

        request = self._make_request(
            method=method,
            path=gateway_path,
            url_override=server_url,
            json_body=body_payload,
            headers=http_headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("ComprehendGatewayAPI(async): %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("ComprehendGatewayAPI(async): %s ← %s", response.status_code, request.url)

        if not response.is_success:
            raise_for_status(response)

        try:
            raw: dict[str, Any] = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc

        try:
            return self._parse_response(
                raw,
                text_fallback=(text or ""),
                labels_fallback=(labels or []),
                threshold_fallback=threshold,
            )
        except (ValueError, PydanticValidationError) as exc:
            raise InternalServerError(
                body={"message": "Malformed or unexpected response", "raw_data": raw},
                headers=dict(response.headers),
            ) from exc

    # Backwards compatible alias
    async def resolve_async(self, *a, **kw):
        return await self.gateway_async(*a, **kw)

    # Convenience wrappers for common HTTP verbs (sync)
    def gateway_get(self, *a, **kw):
        return self.gateway("GET", *a, **kw)

    def gateway_post(self, *a, **kw):
        return self.gateway("POST", *a, **kw)

    def gateway_put(self, *a, **kw):
        return self.gateway("PUT", *a, **kw)

    def gateway_patch(self, *a, **kw):
        return self.gateway("PATCH", *a, **kw)

    def gateway_delete(self, *a, **kw):
        return self.gateway("DELETE", *a, **kw)

    # Convenience wrappers for common HTTP verbs (async)
    async def gateway_get_async(self, *a, **kw):
        return await self.gateway_async("GET", *a, **kw)

    async def gateway_post_async(self, *a, **kw):
        return await self.gateway_async("POST", *a, **kw)

    async def gateway_put_async(self, *a, **kw):
        return await self.gateway_async("PUT", *a, **kw)

    async def gateway_patch_async(self, *a, **kw):
        return await self.gateway_async("PATCH", *a, **kw)

    async def gateway_delete_async(self, *a, **kw):
        return await self.gateway_async("DELETE", *a, **kw)

