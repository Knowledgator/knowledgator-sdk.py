from __future__ import annotations

from typing import Any, Dict, List, Union, Mapping, Optional
from dataclasses import dataclass

import httpx
from pydantic import ValidationError as PydanticValidationError

from .base import BaseAPI, EndpointsConfig
from ..models import (
    AnalyzeOptions,
    AnalyzeTaskRelation,
    AnalyzeTaskSpan,
    ExtractBaseRequest,
    ExtractRelationsResults,
    ExtractRequest,
    ExtractResults,
    ExtractSummaryResults,
    RelationResponse,
    parse_task_output,
)
from ..exceptions import (
    ServerError,
    TransportTimeout,
    ConnectionFailure,
    from_http_response,
)
from ..utils.dictionaries_op import get_inference_tasks, get_model_id_header


@dataclass
class ExtractEndpointsConfig(EndpointsConfig):
    main: str = ""
    batch: str = "batch"
    capabilities: str = "capabilities"
    relations: str = "relations"
    summary: str = "summary"


class ExtractAPI(BaseAPI):
    """ExtractAPI - zero-shot NER, information and relation extraction API service."""

    PATH = "/inference/extract/"
    METHOD = "POST"
    CONTENT_TYPE = "application/json"
    ENDPOINTS_CONFIG = ExtractEndpointsConfig()

    SUCCESS_STATUS = "ok"
    DEFAULT_THRESHOLD = 0.5


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
            outputs=relations,
            input_text=raw.get("input_text", text_fallback),
            entities=raw.get("entities", entities_fallback),
            relations=raw.get("relations", relations_fallback),
            threshold=raw.get(
                "threshold",
                threshold_fallback
                if threshold_fallback is not None
                else self.DEFAULT_THRESHOLD,
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
                threshold_fallback
                if threshold_fallback is not None
                else self.DEFAULT_THRESHOLD,
            ),
            status=raw.get("status", self.SUCCESS_STATUS),
        )

    def _parse_extract_response(
        self,
        raw: Dict[str, Any],
        target_input: Any,
        tasks: Optional[Dict[str, Any]] = None,
    ) -> ExtractResults:
        """Parse extract response supporting task dict format."""
        if "outputs" not in raw:
            raise ServerError("Unexpected response shape (missing 'outputs')", payload={"raw": raw})

        tasks_raw = get_inference_tasks(raw)

        task_threshold = None
        task_labels = []
        if tasks and isinstance(tasks, dict):
            for t in tasks.values():
                if isinstance(t, dict):
                    if t.get("labels"):
                        task_labels = t["labels"] if isinstance(t["labels"], list) else list(t["labels"])
                    if t.get("threshold") is not None:
                        task_threshold = t["threshold"]
                    break
                elif hasattr(t, "labels") and t.labels:
                    task_labels = t.labels.ids if hasattr(t.labels, "ids") else list(t.labels)
                    if hasattr(t, "threshold") and t.threshold is not None:
                        task_threshold = t.threshold
                    break

        if isinstance(tasks_raw, dict):
            parsed_tasks = {k: parse_task_output(v, "span") for k, v in tasks_raw.items()}
            return ExtractResults(
                outputs=parsed_tasks,
                raw=raw,
                input_text=str(target_input) if target_input is not None else "",
                usage=raw.get("usage"),
                threshold=task_threshold,
                status="ok",
            )
        raise ServerError("Unexpected response shape: expected task-keyed dictionary in 'outputs'", payload={"raw": raw})

    def _build_extract_request_body(
        self,
        *,
        text: Optional[str] = None,
        input: Optional[Any] = None,
        tasks: Optional[Dict[str, Any]] = None,
        request: Optional[Union[ExtractRequest, ExtractBaseRequest, Dict[str, Any]]] = None,
        options: Optional[Union[AnalyzeOptions, Dict[str, Any]]] = None,
        adapter_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Any]:
        target_input = input if input is not None else text

        if request is not None:
            if isinstance(request, ExtractRequest):
                return request.model_dump(exclude_none=True, by_alias=True), request.input.root
            if isinstance(request, ExtractBaseRequest):
                task = AnalyzeTaskSpan(type="span", labels=request.labels, threshold=request.threshold)
                req = ExtractRequest(input=request.text, tasks={"entities": task})
                return req.model_dump(exclude_none=True, by_alias=True), request.text
            if isinstance(request, dict):
                return request, request.get("input", target_input)

        opt = options
        if adapter_id and opt is None:
            opt = AnalyzeOptions(adapter_id=adapter_id)
        elif adapter_id and isinstance(opt, dict) and "adapter_id" not in opt:
            opt["adapter_id"] = adapter_id

        if tasks is None:
            raise ValueError("Parameter 'tasks' (or 'request') is required for Extract inference.")

        req = ExtractRequest(
            input=target_input,
            tasks=tasks,
            options=opt,
        )
        return req.model_dump(exclude_none=True, by_alias=True), target_input

    def it(
        self,
        text: Optional[str] = None,
        tasks: Optional[Dict[str, Any]] = None,
        request: Optional[Union[ExtractRequest, ExtractBaseRequest, Dict[str, Any]]] = None,
        options: Optional[Union[AnalyzeOptions, Dict[str, Any]]] = None,
        adapter_id: Optional[str] = None,
        input: Optional[Any] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
        **extra_kwargs: Any,
    ) -> ExtractResults:
        """
        Extract tasks from input text.

        Args:
            text: Input text to extract entities from (alias for input).
            tasks: Dictionary of tasks (span, relation).
            request: Pre-built ExtractRequest object.
            options: Request options.
            adapter_id: Optional LoRA adapter ID (attaches KnowledgatorAPI-Model-ID header).
            input: Context input.
            url_variables: Optional URL path variables.
            server_url: Optional server URL override.
            timeout_ms: Optional request timeout in milliseconds.
            http_headers: Optional additional HTTP headers.
            **extra_kwargs: Additional keyword arguments.
        """
        if "labels" in extra_kwargs:
            raise TypeError(
                "The 'labels' argument has been removed from extract.it. "
                "Please use 'tasks={\"entities\": {\"type\": \"span\", \"labels\": [...]}}' instead."
            )

        json_body, target_input = self._build_extract_request_body(
            text=text,
            input=input,
            tasks=tasks,
            request=request,
            options=options,
            adapter_id=adapter_id,
        )

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.main}"

        headers = {
            **(http_headers or {}),
            **get_model_id_header(adapter_id),
        }

        req = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=json_body,
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("ExtractAPI: %s %s", req.method, req.url)

        try:
            response = self.send(req)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("ExtractAPI: %s ← %s", response.status_code, req.url)

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

        return self._parse_extract_response(
            raw,
            target_input=target_input,
            tasks=tasks,
        )

    async def it_async(
        self,
        text: Optional[str] = None,
        tasks: Optional[Dict[str, Any]] = None,
        request: Optional[Union[ExtractRequest, ExtractBaseRequest, Dict[str, Any]]] = None,
        options: Optional[Union[AnalyzeOptions, Dict[str, Any]]] = None,
        adapter_id: Optional[str] = None,
        input: Optional[Any] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
        **extra_kwargs: Any,
    ) -> ExtractResults:
        """Asynchronously extract tasks from input text."""
        if "labels" in extra_kwargs:
            raise TypeError(
                "The 'labels' argument has been removed from extract.it_async. "
                "Please use 'tasks={\"entities\": {\"type\": \"span\", \"labels\": [...]}}' instead."
            )

        json_body, target_input = self._build_extract_request_body(
            text=text,
            input=input,
            tasks=tasks,
            request=request,
            options=options,
            adapter_id=adapter_id,
        )

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.main}"

        headers = {
            **(http_headers or {}),
            **get_model_id_header(adapter_id),
        }

        req = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=json_body,
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("ExtractAPI: %s %s", req.method, req.url)

        try:
            response = await self.send_async(req)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("ExtractAPI: %s ← %s", response.status_code, req.url)

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

        return self._parse_extract_response(
            raw,
            target_input=target_input,
            tasks=tasks,
        )

    def relations(
        self,
        text: str,
        relations: List[str],
        entities: List[str],
        threshold: Optional[float] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        adapter_id: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> Union[ExtractResults, ExtractRelationsResults]:
        """Extract relations between entities."""
        task = AnalyzeTaskRelation(
            type="relation",
            labels=relations,
            entity_labels=entities,
            threshold=threshold,
            instruction=prompt,
        )
        opt = AnalyzeOptions(adapter_id=adapter_id) if adapter_id else None
        req_obj = ExtractRequest(input=text, tasks={"relations": task}, options=opt)
        json_body = req_obj.model_dump(exclude_none=True, by_alias=True)

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.relations}"

        headers = {
            **(http_headers or {}),
            **get_model_id_header(adapter_id),
        }

        req = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=json_body,
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        try:
            response = self.send(req)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        raw: dict[str, Any] = response.json()
        tasks_raw = get_inference_tasks(raw)

        if isinstance(tasks_raw, dict):
            parsed_tasks = {k: parse_task_output(v, "relation") for k, v in tasks_raw.items()}
            return ExtractResults(
                outputs=parsed_tasks,
                raw=raw,
                input_text=text,
                threshold=threshold,
                status="ok",
            )
        return self._parse_relations(
            raw,
            text_fallback=text,
            entities_fallback=entities,
            relations_fallback=relations,
            threshold_fallback=threshold,
        )

    async def relations_async(
        self,
        text: str,
        relations: List[str],
        entities: List[str],
        threshold: Optional[float] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        adapter_id: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> Union[ExtractResults, ExtractRelationsResults]:
        """Asynchronously extract relations between entities."""
        task = AnalyzeTaskRelation(
            type="relation",
            labels=relations,
            entity_labels=entities,
            threshold=threshold,
            instruction=prompt,
        )
        opt = AnalyzeOptions(adapter_id=adapter_id) if adapter_id else None
        req_obj = ExtractRequest(input=text, tasks={"relations": task}, options=opt)
        json_body = req_obj.model_dump(exclude_none=True, by_alias=True)

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.relations}"

        headers = {
            **(http_headers or {}),
            **get_model_id_header(adapter_id),
        }

        req = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=json_body,
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        try:
            response = await self.send_async(req)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        raw: dict[str, Any] = response.json()
        tasks_raw = get_inference_tasks(raw)

        if isinstance(tasks_raw, dict):
            parsed_tasks = {k: parse_task_output(v, "relation") for k, v in tasks_raw.items()}
            return ExtractResults(
                results=parsed_tasks,
                raw=raw,
                input_text=text,
                threshold=threshold,
                status="ok",
            )
        return self._parse_relations(
            raw,
            text_fallback=text,
            entities_fallback=entities,
            relations_fallback=relations,
            threshold_fallback=threshold,
        )

    def summary(
        self,
        text: str,
        threshold: Optional[float] = None,
        prompt: Optional[str] = None,
        adapter_id: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> Union[ExtractResults, ExtractSummaryResults]:
        """Extract extractive summary from input text."""
        task = AnalyzeTaskSpan(
            type="span",
            labels=None,
            threshold=threshold,
            instruction=prompt,
        )
        opt = AnalyzeOptions(adapter_id=adapter_id) if adapter_id else None
        req_obj = ExtractRequest(input=text, tasks={"summary": task}, options=opt)
        json_body = req_obj.model_dump(exclude_none=True, by_alias=True)

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.summary}"

        headers = {
            **(http_headers or {}),
            **get_model_id_header(adapter_id),
        }

        req = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=json_body,
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        try:
            response = self.send(req)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        raw: dict[str, Any] = response.json()
        if "summary" in raw and isinstance(raw["summary"], str):
            return self._parse_summary(raw, text_fallback=text, threshold_fallback=threshold)

        if "outputs" in raw or "results" in raw:
            tasks_raw = get_inference_tasks(raw)
            if isinstance(tasks_raw, dict):
                parsed_tasks = {k: parse_task_output(v, "span") for k, v in tasks_raw.items()}
                return ExtractResults(
                    results=parsed_tasks,
                    raw=raw,
                    input_text=text,
                    threshold=threshold,
                    status="ok",
                )
        return self._parse_summary(raw, text_fallback=text, threshold_fallback=threshold)

    async def summary_async(
        self,
        text: str,
        threshold: Optional[float] = None,
        prompt: Optional[str] = None,
        adapter_id: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> Union[ExtractResults, ExtractSummaryResults]:
        """Asynchronously extract extractive summary from input text."""
        task = AnalyzeTaskSpan(
            type="span",
            labels=None,
            threshold=threshold,
            instruction=prompt,
        )
        opt = AnalyzeOptions(adapter_id=adapter_id) if adapter_id else None
        req_obj = ExtractRequest(input=text, tasks={"summary": task}, options=opt)
        json_body = req_obj.model_dump(exclude_none=True, by_alias=True)

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.summary}"

        headers = {
            **(http_headers or {}),
            **get_model_id_header(adapter_id),
        }

        req = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            json_body=json_body,
            headers=headers,
            accept=self.CONTENT_TYPE,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        try:
            response = await self.send_async(req)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        raw: dict[str, Any] = response.json()
        if "summary" in raw and isinstance(raw["summary"], str):
            return self._parse_summary(raw, text_fallback=text, threshold_fallback=threshold)

        if "outputs" in raw or "results" in raw:
            tasks_raw = get_inference_tasks(raw)
            if isinstance(tasks_raw, dict):
                parsed_tasks = {k: parse_task_output(v, "span") for k, v in tasks_raw.items()}
                return ExtractResults(
                    results=parsed_tasks,
                    raw=raw,
                    input_text=text,
                    threshold=threshold,
                    status="ok",
                )
        return self._parse_summary(raw, text_fallback=text, threshold_fallback=threshold)
