from typing import List, Optional, Dict, Mapping, Any
import httpx

from .base import BaseAPI, EndpointsConfig
from ..configs import RetryPolicyConfigs
from ..utils.dictionaries_op import remove_none_from_dict
from ..models.inputs import ExtractBaseRequest, ExtractRelationsRequest, ExtractSummaryRequest
from ..models.outputs import ExtractBaseResults, ExtractRelationsResults, ExtractSummaryResults
from ..exceptions.base import (
    ServerError,
    TransportTimeout,
    ConnectionFailure,
    from_http_response,
)
from pydantic import ValidationError as PydanticValidationError

class ExtractEndpointsConfig(EndpointsConfig):
    main = "general"
    relations = "relation-extraction"
    summary = "summarization"

class ExtractAPI(BaseAPI):
    """
    ExtractAPI - zero-shot NER and information extraction API service."""

    PATH = '/extract_it/'
    METHOD = 'POST'
    CONTENT_TYPE = 'application/json'
    ENDPOINTS_CONFIG = ExtractEndpointsConfig()

    def it(
        self,
        text: str,
        labels: List[str],
        threshold: Optional[float] = None,
        prompt: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        retries: Optional[RetryPolicyConfigs] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ExtractBaseResults:
        body = ExtractBaseRequest(
            text=text,
            labels=labels,
            threshold=threshold,
            prompt=prompt,
        )
        path = "{}/{self.ENDPOINTS_CONFIG.main}".format(self.PATH)
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
            retries=retries or self.configs.retry_config,
        )

        if self._log:
            self._log.debug("ExtractAPI: %s %s", request.method, request.url)

        # ── Transport layer ─────────────────────────────────────────────────────────
        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "ExtractAPI: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            result_obj = ExtractBaseResults.model_validate(response.json())
        except (ValueError, PydanticValidationError) as exc:
            # Bad/malformed JSON *or* schema mismatch → treat as server fault
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        if self._log:
            self._log.debug("ExtractAPI: extracted information successfully")
        return result_obj

    async def it_async(
        self,
        text: str,
        labels: List[str],
        threshold: Optional[float] = None,
        prompt: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        retries: Optional[RetryPolicyConfigs] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ExtractBaseResults:
        body = ExtractBaseRequest(
            text=text,
            labels=labels,
            threshold=threshold,
            prompt=prompt,
        )
        path = "{}/{self.ENDPOINTS_CONFIG.main}".format(self.PATH)
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
            retries=retries or self.configs.retry_config,
        )

        if self._log:
            self._log.debug("ExtractAPI: %s %s", request.method, request.url)

        # ── Transport layer ─────────────────────────────────────────────────────────
        try:
            response = await self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "ExtractAPI: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            result_obj = ExtractBaseResults.model_validate(response.json())
        except (ValueError, PydanticValidationError) as exc:
            # Bad/malformed JSON *or* schema mismatch → treat as server fault
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        if self._log:
            self._log.debug("ExtractAPI: extracted information successfully")
        return result_obj
    

    def relations(
        self,
        text: str,
        entities: List[str],
        relations: List[str],
        threshold: Optional[float] = None,
        prompt: Optional[str] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        retries: Optional[RetryPolicyConfigs] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ExtractBaseResults:
        body = ExtractRelationsRequest(
            text=text,
            entities=entities,
            relations=relations,
            threshold=threshold,
            prompt=prompt,
            rules=rules,
        )
        path = "{}/{self.ENDPOINTS_CONFIG.relations}".format(self.PATH)
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
            retries=retries or self.configs.retry_config,
        )

        if self._log:
            self._log.debug("ExtractAPI: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "ExtractAPI: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            result_obj = ExtractRelationsResults.model_validate(response.json())
        except (ValueError, PydanticValidationError) as exc:
            # Bad/malformed JSON *or* schema mismatch → treat as server fault
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        if self._log:
            self._log.debug("ExtractAPI: extracted relations successfully")
        return result_obj

    async def relations_async(
        self,
        text: str,
        entities: List[str],
        relations: List[str],
        threshold: Optional[float] = None,
        prompt: Optional[str] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        retries: Optional[RetryPolicyConfigs] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ExtractBaseResults:
        body = ExtractRelationsRequest(
            text=text,
            entities=entities,
            relations=relations,
            threshold=threshold,
            prompt=prompt,
            rules=rules,
        )
        path = "{}/{self.ENDPOINTS_CONFIG.relations}".format(self.PATH)
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
            retries=retries or self.configs.retry_config,
        )

        if self._log:
            self._log.debug("ExtractAPI: %s %s", request.method, request.url)

        try:
            response = await self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "ExtractAPI: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            result_obj = ExtractRelationsResults.model_validate(response.json())
        except (ValueError, PydanticValidationError) as exc:
            # Bad/malformed JSON *or* schema mismatch → treat as server fault
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        if self._log:
            self._log.debug("ExtractAPI: extracted relations successfully")
        return result_obj

    def summary(
        self,
        text: str,
        threshold: Optional[float] = None,
        prompt: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        retries: Optional[RetryPolicyConfigs] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ExtractBaseResults:
        body = ExtractSummaryRequest(
            text=text,
            threshold=threshold,
            prompt=prompt,
        )

        path = "{}/{self.ENDPOINTS_CONFIG.summary}".format(self.PATH)
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
            retries=retries or self.configs.retry_config,
        )

        if self._log:
            self._log.debug("ExtractAPI: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "ExtractAPI: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            result_obj = ExtractSummaryResults.model_validate(response.json())
        except (ValueError, PydanticValidationError) as exc:
            # Bad/malformed JSON *or* schema mismatch → treat as server fault
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        if self._log:
            self._log.debug("ExtractAPI: extracted summary successfully")
        return result_obj
    
    async def summary_async(
        self,
        text: str,
        threshold: Optional[float] = None,
        prompt: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        retries: Optional[RetryPolicyConfigs] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ExtractBaseResults:
        body = ExtractSummaryRequest(
            text=text,
            threshold=threshold,
            prompt=prompt,
        )

        path = "{}/{self.ENDPOINTS_CONFIG.summary}".format(self.PATH)
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
            retries=retries or self.configs.retry_config,
        )

        if self._log:
            self._log.debug("ExtractAPI: %s %s", request.method, request.url)

        try:
            response = await self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "ExtractAPI: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            result_obj = ExtractSummaryResults.model_validate(response.json())
        except (ValueError, PydanticValidationError) as exc:
            # Bad/malformed JSON *or* schema mismatch → treat as server fault
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        if self._log:
            self._log.debug("ExtractAPI: extracted summary successfully")
        return result_obj