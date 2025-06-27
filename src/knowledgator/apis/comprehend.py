from typing import List, Optional, Dict, Mapping, Any
import httpx

from .base import BaseAPI
from ..configs import RetryPolicyConfigs
from ..utils.dictionaries_op import remove_none_from_dict
from ..models.inputs import ComprehendRequest
from ..models.outputs import ComprehendResults
from ..exceptions.base import (
    ServerError,
    TransportTimeout,
    ConnectionFailure,
    from_http_response,
)
from pydantic import ValidationError as PydanticValidationError

class ComprehendAPI(BaseAPI):
    """
    ComprehendAPI - zero-shot classification API service,"""

    PATH = '/comprehend_it/classify'
    METHOD = 'POST'
    CONTENT_TYPE = 'application/json'

    def run(
        self,
        text: str,
        labels: List[str],
        threshold: Optional[float] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        retries: Optional[RetryPolicyConfigs] = None,
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
        request = self._make_request(
            method=self.METHOD,
            path=self.PATH,
            url_override=server_url,
            url_variables=url_variables,
            json_body=body.model_dump(exclude_none=True),
            headers=http_headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
            retries=retries or self.configs.retry_config,
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
            result_obj = ComprehendResults.model_validate(response.json())
        except (ValueError, PydanticValidationError) as exc:
            # Bad/malformed JSON *or* schema mismatch → treat as server fault
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        if self._log:
            self._log.debug("ComprehendAPI: classified text successfully")
        return result_obj

    async def run_async(
        self,
        text: str,
        labels: List[str],
        threshold: Optional[float] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        retries: Optional[RetryPolicyConfigs] = None,
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
        request = self._make_request(
            method=self.METHOD,
            path=self.PATH,
            url_override=server_url,
            url_variables=url_variables,
            json_body=body.model_dump(exclude_none=True),
            headers=http_headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
            retries=retries or self.configs.retry_config,
        )

        if self._log:
            self._log.debug("ComprehendAPI: %s %s", request.method, request.url)

        # ── Transport layer ─────────────────────────────────────────────────────────
        try:
            response = await self.send_async(request)
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
            result_obj = ComprehendResults.model_validate(response.json())
        except (ValueError, PydanticValidationError) as exc:
            # Bad/malformed JSON *or* schema mismatch → treat as server fault
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        if self._log:
            self._log.debug("ComprehendAPI: classified text successfully")
        return result_obj