from __future__ import annotations

from typing import Any, Dict, List, Union, Mapping, Optional
from dataclasses import dataclass

import httpx
from pydantic import ValidationError as PydanticValidationError

from .base import BaseAPI, EndpointsConfig
from ..models import (
    AnalyzeOptions,
    ComprehendRequest,
    ComprehendResults,
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
class ComprehendEndpointsConfig(EndpointsConfig):
    main: str = ""
    batch: str = "batch"
    capabilities: str = "capabilities"


class ComprehendAPI(BaseAPI):
    """ComprehendAPI - zero-shot classification and semantic decision API service."""

    PATH = "/inference/comprehend/"
    METHOD = "POST"
    CONTENT_TYPE = "application/json"
    ENDPOINTS_CONFIG = ComprehendEndpointsConfig()
    SUCCESS_STATUS = "ok"
    DEFAULT_THRESHOLD = 0.5

    def _build_request_body(
        self,
        *,
        text: Optional[str] = None,
        input: Optional[Any] = None,
        tasks: Optional[Dict[str, Any]] = None,
        request: Optional[Union[ComprehendRequest, Dict[str, Any]]] = None,
        options: Optional[Union[AnalyzeOptions, Dict[str, Any]]] = None,
        adapter_id: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Any]:
        """Construct serialized JSON request payload and retain target input for result tracking."""
        target_input = input if input is not None else text

        if request is not None:
            if isinstance(request, ComprehendRequest):
                return request.model_dump(exclude_none=True, by_alias=True), request.input.root
            if isinstance(request, dict):
                return request, request.get("input", target_input)
            raise TypeError(f"Invalid request type: {type(request)}. Expected ComprehendRequest or dict.")

        opt = options
        if adapter_id and opt is None:
            opt = AnalyzeOptions(adapter_id=adapter_id)
        elif adapter_id and isinstance(opt, dict) and "adapter_id" not in opt:
            opt["adapter_id"] = adapter_id

        if tasks is None:
            raise ValueError("Parameter 'tasks' (or 'request') is required for Comprehend inference.")

        req = ComprehendRequest(
            input=target_input,
            tasks=tasks,
            options=opt,
        )
        return req.model_dump(exclude_none=True, by_alias=True), target_input

    def _parse_response(
        self,
        raw: Dict[str, Any],
        target_input: Any,
        tasks: Optional[Dict[str, Any]] = None,
    ) -> ComprehendResults:
        """Parse raw response using get_inference_tasks helper."""
        if "outputs" not in raw:
            raise ServerError("Unexpected response shape (missing 'outputs')", payload={"raw": raw})

        tasks_raw = get_inference_tasks(raw)

        task_threshold = None
        if tasks and isinstance(tasks, dict):
            for t in tasks.values():
                if isinstance(t, dict) and t.get("threshold") is not None:
                    task_threshold = t["threshold"]
                    break
                elif hasattr(t, "threshold") and t.threshold is not None:
                    task_threshold = t.threshold
                    break

        if isinstance(tasks_raw, dict):
            parsed_tasks = {k: parse_task_output(v, "classify") for k, v in tasks_raw.items()}
            return ComprehendResults(
                outputs=parsed_tasks,
                raw=raw,
                input_text=str(target_input) if target_input is not None else "",
                usage=raw.get("usage"),
                threshold=task_threshold,
                status="ok",
            )
        raise ServerError("Unexpected response shape: expected task-keyed dictionary in 'outputs'", payload={"raw": raw})

    def it(
        self,
        text: Optional[str] = None,
        tasks: Optional[Dict[str, Any]] = None,
        request: Optional[Union[ComprehendRequest, Dict[str, Any]]] = None,
        options: Optional[Union[AnalyzeOptions, Dict[str, Any]]] = None,
        adapter_id: Optional[str] = None,
        input: Optional[Any] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
        **extra_kwargs: Any,
    ) -> ComprehendResults:
        """
        Evaluate tasks using Comprehend endpoint.

        Args:
            text: Input text to evaluate (alias for input).
            tasks: Dictionary of tasks (e.g. classify, boolean, score).
            request: Pre-built ComprehendRequest object.
            options: Request options (e.g. include_usage, adapter_id).
            adapter_id: Optional LoRA adapter ID (attaches KnowledgatorAPI-Model-ID header).
            input: Context input (string, dict, or list).
            url_variables: Optional URL path variables.
            server_url: Optional server URL override.
            timeout_ms: Optional request timeout in milliseconds.
            http_headers: Optional additional HTTP headers.
            **extra_kwargs: Additional keyword arguments.
        """
        if "labels" in extra_kwargs:
            raise TypeError(
                "The 'labels' argument has been removed from comprehend.it. "
                "Please use 'tasks={\"topic\": {\"type\": \"classify\", \"labels\": [...]}}' instead."
            )

        json_body, target_input = self._build_request_body(
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
            self._log.debug("ComprehendAPI: %s %s", req.method, req.url)

        try:
            response = self.send(req)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("ComprehendAPI: %s ← %s", response.status_code, req.url)

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

        result_obj = self._parse_response(
            raw,
            target_input=target_input,
            tasks=tasks,
        )

        if self._log:
            self._log.debug("ComprehendAPI: evaluated tasks successfully")
        return result_obj

    async def it_async(
        self,
        text: Optional[str] = None,
        tasks: Optional[Dict[str, Any]] = None,
        request: Optional[Union[ComprehendRequest, Dict[str, Any]]] = None,
        options: Optional[Union[AnalyzeOptions, Dict[str, Any]]] = None,
        adapter_id: Optional[str] = None,
        input: Optional[Any] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
        **extra_kwargs: Any,
    ) -> ComprehendResults:
        """Asynchronously evaluate tasks using Comprehend endpoint."""
        if "labels" in extra_kwargs:
            raise TypeError(
                "The 'labels' argument has been removed from comprehend.it_async. "
                "Please use 'tasks={\"topic\": {\"type\": \"classify\", \"labels\": [...]}}' instead."
            )

        json_body, target_input = self._build_request_body(
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
            self._log.debug("ComprehendAPI: %s %s", req.method, req.url)

        try:
            response = await self.send_async(req)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("ComprehendAPI: %s ← %s", response.status_code, req.url)

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

        result_obj = self._parse_response(
            raw,
            target_input=target_input,
            tasks=tasks,
        )

        if self._log:
            self._log.debug("ComprehendAPI: evaluated tasks successfully")
        return result_obj
