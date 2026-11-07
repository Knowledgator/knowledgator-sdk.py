import json
from typing import Any, List, Mapping, Optional
from dataclasses import dataclass

import httpx

from .base import BaseAPI, EndpointsConfig
from ..exceptions import (
    BadRequest,
    ServerError,
    TransportTimeout,
    ConnectionFailure,
    from_http_response,
)
from ..models.common import FileMetadata
from ..models.outputs import InstructResults
from ..utils.dictionaries_op import get_model_id_header


@dataclass
class InstructEndpointsConfig(EndpointsConfig):
    main: str = ""


class InstructAPI(BaseAPI):
    """
    InstructAPI - Instruct-it inference endpoint for instruction-following models.

    This API accepts a text instruction plus optional template, file references,
    and output type hint, and returns a JSON-serializable object.
    """

    PATH = "/inference/instruct/"
    METHOD = "POST"
    CONTENT_TYPE = "multipart/form-data"
    ENDPOINTS_CONFIG = InstructEndpointsConfig()
    SUCCESS_STATUS = "ok"

    def _parse_response(self, raw: dict[str, Any]) -> InstructResults:
        """Parse the API response into InstructResults."""
        outputs = raw.get("outputs")

        if outputs is None:
            raise ServerError(
                "Unexpected response shape (missing 'outputs')",
                status_code=200,
                payload={"raw": raw},
            )

        parsed = raw.get("parsed", True)

        return InstructResults(outputs=outputs, parsed=parsed)

    def it(
        self,
        text: str,
        template: Optional[str] = None,
        files: Optional[List[FileMetadata]] = None,
        output_type: Optional[str] = None,
        adapter_id: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> InstructResults:
        """
        Execute an instruction using the Instruct-it model.

        Args:
            text: The main instruction/prompt for the model (required).
            template: JSON string representing a structure the model should fill.
            files: List of FileMetadata with file references/content.
            output_type: Hint for desired output style (json, string, list, markdown, etc.).
            adapter_id: Optional LoRA adapter ID for fine-tuned adapters (attaches KnowledgatorAPI-Model-ID header).
            url_variables: Optional URL path variables.
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Note:
            For standalone full models (open-models), use client.model(open_model_id).predict(...) instead.

        Returns:
            InstructResults with the model's output.

        Raises:
            BadRequest: If template is not valid JSON.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error or invalid response.
        """
        # Validate template JSON if provided
        if template is not None:
            try:
                json.loads(template)
            except json.JSONDecodeError as exc:
                raise BadRequest(
                    f"Invalid JSON in template: {exc}", payload={"template": template}
                ) from exc

        # Build form data payload
        data_payload = {"text": text}

        if template is not None:
            data_payload["template"] = template

        if output_type is not None:
            data_payload["output_type"] = output_type

        # Build filesmetadata list (names of all files)
        if files:
            filesmetadata = [f.name for f in files]
            data_payload["filesmetadata"] = json.dumps(filesmetadata)

        # Build files payload (always pass dict so httpx uses multipart/form-data)
        files_payload = {}
        if files:
            files_with_content = [f for f in files if f.content is not None]
            if files_with_content:
                files_payload = {
                    f.name: (f.name, f.content) for f in files_with_content
                }

        main_endpoint = self.ENDPOINTS_CONFIG.main.strip("/")
        path = f"{self.PATH.rstrip('/')}/{main_endpoint}/" if main_endpoint else self.PATH

        headers = {
            **(http_headers or {}),
            **get_model_id_header(adapter_id),
        }

        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            data_body=data_payload,
            files=files_payload,
            headers=headers,
            accept="application/json",  # We accept JSON response
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("InstructAPI: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout(f"Request timed-out for {request.url}") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure(f"Transport error for {request.url}: {exc} ({type(exc).__name__})") from exc

        if self._log:
            self._log.debug("InstructAPI: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
        except ValueError:
            raw_text = response.text
            if self._log:
                self._log.debug(
                    "InstructAPI: Non-JSON response, wrapping as string output"
                )
            return InstructResults.from_model_response(raw_text)

        try:
            result_obj = self._parse_response(raw)
        except ServerError:
            raise
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": raw},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        if self._log:
            self._log.debug("InstructAPI: instruction executed successfully")
        return result_obj

    async def it_async(
        self,
        text: str,
        template: Optional[str] = None,
        files: Optional[List[FileMetadata]] = None,
        output_type: Optional[str] = None,
        adapter_id: Optional[str] = None,
        url_variables: Optional[Mapping[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> InstructResults:
        """
        Execute an instruction using the Instruct-it model (async version).

        Args:
            text: The main instruction/prompt for the model (required).
            template: JSON string representing a structure the model should fill.
            files: List of FileMetadata with file references/content.
            output_type: Hint for desired output style (json, string, list, markdown, etc.).
            adapter_id: Optional LoRA adapter ID for fine-tuned adapters (attaches KnowledgatorAPI-Model-ID header).
            url_variables: Optional URL path variables.
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Note:
            For standalone full models (open-models), use client.model(open_model_id).predict_async(...) instead.

        Returns:
            InstructResults with the model's output.

        Raises:
            BadRequest: If template is not valid JSON.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error or invalid response.
        """
        # Validate template JSON if provided
        if template is not None:
            try:
                json.loads(template)
            except json.JSONDecodeError as exc:
                raise BadRequest(
                    f"Invalid JSON in template: {exc}", payload={"template": template}
                ) from exc

        # Build form data payload
        data_payload = {"text": text}

        if template is not None:
            data_payload["template"] = template

        if output_type is not None:
            data_payload["output_type"] = output_type

        # Build filesmetadata list (names of all files)
        if files:
            filesmetadata = [f.name for f in files]
            data_payload["filesmetadata"] = json.dumps(filesmetadata)

        # Build files payload (always pass dict so httpx uses multipart/form-data)
        files_payload = {}
        if files:
            files_with_content = [f for f in files if f.content is not None]
            if files_with_content:
                files_payload = {
                    f.name: (f.name, f.content) for f in files_with_content
                }

        main_endpoint = self.ENDPOINTS_CONFIG.main.strip("/")
        path = f"{self.PATH.rstrip('/')}/{main_endpoint}/" if main_endpoint else self.PATH

        headers = {
            **(http_headers or {}),
            **get_model_id_header(adapter_id),
        }

        request = self._make_request(
            method=self.METHOD,
            path=path,
            url_override=server_url,
            url_variables=url_variables,
            data_body=data_payload,
            files=files_payload,
            headers=headers,
            accept="application/json",  # We accept JSON response
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("InstructAPI: %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout(f"Request timed-out for {request.url}") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure(f"Transport error for {request.url}: {exc} ({type(exc).__name__})") from exc

        if self._log:
            self._log.debug("InstructAPI: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
        except ValueError:
            raw_text = response.text
            if self._log:
                self._log.debug(
                    "InstructAPI: Non-JSON response, wrapping as string output"
                )
            return InstructResults.from_model_response(raw_text)

        try:
            result_obj = self._parse_response(raw)
        except ServerError:
            raise
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": raw},
                request_id=response.headers.get("x-request-id"),
            ) from exc

        if self._log:
            self._log.debug("InstructAPI: instruction executed successfully")
        return result_obj
