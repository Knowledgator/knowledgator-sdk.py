from typing import Any, Dict, List, Mapping, Optional
from dataclasses import dataclass

import httpx
from pydantic import ValidationError as PydanticValidationError

from .base import BaseAPI, EndpointsConfig
from ..exceptions import (
    ServerError,
    TransportTimeout,
    ConnectionFailure,
    from_http_response,
)
from ..models.inputs import (
    ExtractBaseRequest,
    ExtractSummaryRequest,
    ExtractRelationsRequest,
)
from ..models.outputs import (
    EntityResponse,
    RelationResponse,
    ExtractBaseResults,
    ExtractSummaryResults,
    ExtractRelationsResults,
)
from ..utils.dictionaries_op import get_model_id_header


@dataclass
class ExtractEndpointsConfig(EndpointsConfig):
    main: str = ""
    relations: str = "relations"
    summary: str = "summary"


class ExtractAPI(BaseAPI):
    """
    ExtractAPI - zero-shot NER and information extraction API service."""

    PATH = "/inference/extract/"
    METHOD = "POST"
    CONTENT_TYPE = "application/json"
    ENDPOINTS_CONFIG = ExtractEndpointsConfig()

    SUCCESS_STATUS = "ok"
    DEFAULT_THRESHOLD = 0.5

    def _parse_entities(
        self,
        raw: dict[str, Any],
        *,
        text_fallback: str,
        labels_fallback: List[str],
        threshold_fallback: Optional[float],
    ) -> ExtractBaseResults:
        outputs = raw.get("outputs")
        if not isinstance(outputs, list):
            raise ServerError(
                "Unexpected response shape (missing 'outputs')",
                payload={"raw": raw},
            )

        try:
            entities = [EntityResponse.model_validate(o) for o in outputs]
        except PydanticValidationError as exc:
            raise ServerError(
                "Unexpected entity record format inside 'outputs'",
                payload={"raw": raw},
            ) from exc

        return ExtractBaseResults(
            results=entities,
            input_text=raw.get("input_text", text_fallback),
            labels=raw.get("labels") or raw.get("entities") or labels_fallback,
            threshold=raw.get(
                "threshold",
                threshold_fallback if threshold_fallback is not None else self.DEFAULT_THRESHOLD,
            ),
            status=raw.get("status", self.SUCCESS_STATUS),
        )

    def _parse_relations(
        self,
        raw: dict[str, Any],
        *,
        text_fallback: str,
        entities_fallback: List[str],
        relations_fallback: List[str],
        threshold_fallback: Optional[float],
    ) -> ExtractRelationsResults:
        outputs = raw.get("outputs")
        if not isinstance(outputs, list):
            raise ServerError(
                "Unexpected response shape (missing 'outputs')",
                payload={"raw": raw},
            )

        try:
            relations = [RelationResponse.model_validate(o) for o in outputs]
        except PydanticValidationError as exc:
            raise ServerError(
                "Unexpected relation record format inside 'outputs'",
                payload={"raw": raw},
            ) from exc

        return ExtractRelationsResults(
            results=relations,
            input_text=raw.get("input_text", text_fallback),
            entities=raw.get("entities", entities_fallback),
            relations=raw.get("relations", relations_fallback),
            threshold=raw.get(
                "threshold",
                threshold_fallback if threshold_fallback is not None else self.DEFAULT_THRESHOLD,
            ),
            status=raw.get("status", self.SUCCESS_STATUS),
        )

    def _parse_summary(
        self,
        raw: dict[str, Any],
        *,
        text_fallback: str,
        threshold_fallback: Optional[float],
    ) -> ExtractSummaryResults:
        summary = raw.get("summary")
        if not isinstance(summary, str):
            raise ServerError(
                "Unexpected response shape (missing 'summary')",
                payload={"raw": raw},
            )

        return ExtractSummaryResults(
            summary=summary,
            input_text=raw.get("input_text", text_fallback),
            threshold=raw.get(
                "threshold",
                threshold_fallback if threshold_fallback is not None else self.DEFAULT_THRESHOLD,
            ),
            status=raw.get("status", self.SUCCESS_STATUS),
        )

    def it(
        self,
        text: str,
        labels: List[str],
        threshold: Optional[float] = None,
        prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
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
        path = f"{self.PATH}/{self.ENDPOINTS_CONFIG.main}"

        headers = {
            **(http_headers or {}),
            **get_model_id_header(model_id),
        }

        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=body.model_dump(exclude_none=True, by_alias=True),
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
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
            self._log.debug("ExtractAPI: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw = response.json()
            result_obj = self._parse_entities(
                raw,
                text_fallback=text,
                labels_fallback=labels,
                threshold_fallback=threshold,
            )
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
        model_id: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
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
        path = f"{self.PATH}/{self.ENDPOINTS_CONFIG.main}"

        headers = {
            **(http_headers or {}),
            **get_model_id_header(model_id),
        }

        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=body.model_dump(exclude_none=True, by_alias=True),
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("ExtractAPI: %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout(f"Request timed-out for {request.url}") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure(f"Transport error for {request.url}: {exc} ({type(exc).__name__})") from exc

        if self._log:
            self._log.debug("ExtractAPI: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw = response.json()
            result_obj = self._parse_entities(
                raw,
                text_fallback=text,
                labels_fallback=labels,
                threshold_fallback=threshold,
            )
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
        model_id: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
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
        path = f"{self.PATH}/{self.ENDPOINTS_CONFIG.relations}"

        headers = {
            **(http_headers or {}),
            **get_model_id_header(model_id),
        }

        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=body.model_dump(exclude_none=True),
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
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
            self._log.debug("ExtractAPI: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw = response.json()
            result_obj = self._parse_relations(
                raw,
                text_fallback=text,
                entities_fallback=entities,
                relations_fallback=relations,
                threshold_fallback=threshold,
            )
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
        model_id: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
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
        path = f"{self.PATH}/{self.ENDPOINTS_CONFIG.relations}"

        headers = {
            **(http_headers or {}),
            **get_model_id_header(model_id),
        }

        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=body.model_dump(exclude_none=True),
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("ExtractAPI: %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout(f"Request timed-out for {request.url}") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure(f"Transport error for {request.url}: {exc} ({type(exc).__name__})") from exc

        if self._log:
            self._log.debug("ExtractAPI: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw = response.json()
            result_obj = self._parse_relations(
                raw,
                text_fallback=text,
                entities_fallback=entities,
                relations_fallback=relations,
                threshold_fallback=threshold,
            )
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
        model_id: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ExtractBaseResults:
        body = ExtractSummaryRequest(
            text=text,
            threshold=threshold,
            prompt=prompt,
        )

        path = f"{self.PATH}/{self.ENDPOINTS_CONFIG.summary}"

        headers = {
            **(http_headers or {}),
            **get_model_id_header(model_id),
        }

        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=body.model_dump(exclude_none=True),
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
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
            self._log.debug("ExtractAPI: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw = response.json()
            result_obj = self._parse_summary(raw, text_fallback=text, threshold_fallback=threshold)
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
        model_id: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ExtractBaseResults:
        body = ExtractSummaryRequest(
            text=text,
            threshold=threshold,
            prompt=prompt,
        )

        path = f"{self.PATH}/{self.ENDPOINTS_CONFIG.summary}"

        headers = {
            **(http_headers or {}),
            **get_model_id_header(model_id),
        }

        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=body.model_dump(exclude_none=True),
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("ExtractAPI: %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout(f"Request timed-out for {request.url}") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure(f"Transport error for {request.url}: {exc} ({type(exc).__name__})") from exc

        if self._log:
            self._log.debug("ExtractAPI: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw = response.json()
            result_obj = self._parse_summary(raw, text_fallback=text, threshold_fallback=threshold)
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
