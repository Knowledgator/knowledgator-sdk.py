from typing import List, Optional, Dict, Mapping, Any
from dataclasses import dataclass
import httpx

from .base import BaseAPI, EndpointsConfig
from ..configs import RetryPolicyConfigs
from ..utils.dictionaries_op import remove_none_from_dict
from ..models.inputs import ComprehendRequest
from ..models.outputs import ComprehendResults, ComprehendResponse
from ..exceptions import (
    ServerError,
    TransportTimeout,
    ConnectionFailure,
    from_http_response
)
from pydantic import ValidationError as PydanticValidationError

@dataclass
class ComprehendEndpointsConfig(EndpointsConfig):
    main: str = ""

class ComprehendAPI(BaseAPI):
    """
    ComprehendAPI - zero-shot classification API service,"""

    PATH = '/comprehend-it/'
    METHOD = 'POST'
    CONTENT_TYPE = 'application/json'
    ENDPOINTS_CONFIG = ComprehendEndpointsConfig()
    SUCCESS_STATUS = "ok"        
    DEFAULT_THRESHOLD = 0.5

    def it(
        self,
        text: str,
        labels: List[str],
        threshold: Optional[float] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ComprehendResults:
        body = ComprehendRequest(
            text=text,
            labels=labels,
            threshold=threshold,
            examples=examples,
        )
        path = f"{self.PATH}/{self.ENDPOINTS_CONFIG.main}"
        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=body.model_dump(exclude_none=True),
            headers=http_headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("ComprehendAPI: %s %s", request.method, request.url)

        # ── Transport layer ─────────────────────────────────────────────────────────
        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "ComprehendAPI: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
        except ValueError as exc:
            raise ServerError(
                "Response is not valid JSON",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        outputs = raw.get("outputs")
        if not isinstance(outputs, list):
            raise ServerError(
                "Unexpected response shape (missing 'outputs')",
                status_code=response.status_code,
                payload={"raw": raw},
                request_id=response.headers.get("x-request-id"),
            )

        try:
            parsed_results = [ComprehendResponse.model_validate(o) for o in outputs]
        except PydanticValidationError as exc:
            raise ServerError(
                "Unexpected record format inside 'outputs'",
                status_code=response.status_code,
                payload={"raw": raw},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        result_obj = ComprehendResults(
            results=parsed_results,
            input_text=raw.get("input_text", text),
            labels=raw.get("labels", labels),
            threshold=raw.get("threshold", threshold if threshold is not None else self.DEFAULT_THRESHOLD),
            status=raw.get("status", self.SUCCESS_STATUS),
        )

        if self._log:
            self._log.debug("ComprehendAPI: classified text successfully")
        return result_obj

    async def it_async(
        self,
        text: str,
        labels: list[str],
        threshold: float | None = None,
        examples: list[dict[str, str]] | None = None,
        url_variables: Mapping[str, Any] | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        http_headers: Mapping[str, str] | None = None,
    ) -> ComprehendResults:

        body = ComprehendRequest(
            text=text,
            labels=labels,
            threshold=threshold,
            examples=examples,
        )
        path = f"{self.PATH}/{self.ENDPOINTS_CONFIG.main}"
        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=body.model_dump(exclude_none=True),
            headers=http_headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("ComprehendAPI: %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("ComprehendAPI: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
        except ValueError as exc:
            raise ServerError(
                "Response is not valid JSON",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        outputs = raw.get("outputs")
        if not isinstance(outputs, list):
            raise ServerError(
                "Unexpected response shape (missing 'outputs')",
                status_code=response.status_code,
                payload={"raw": raw},
                request_id=response.headers.get("x-request-id"),
            )

        try:
            parsed_results = [ComprehendResponse.model_validate(o) for o in outputs]
        except PydanticValidationError as exc:
            raise ServerError(
                "Unexpected record format inside 'outputs'",
                status_code=response.status_code,
                payload={"raw": raw},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        result_obj = ComprehendResults(
            results=parsed_results,
            input_text=raw.get("input_text", text),
            labels=raw.get("labels", labels),
            threshold=raw.get(
                "threshold",
                threshold if threshold is not None else self.DEFAULT_THRESHOLD,
            ),
            status=raw.get("status", self.SUCCESS_STATUS),
        )

        if self._log:
            self._log.debug("ComprehendAPI: classified text successfully")
        return result_obj