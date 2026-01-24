from typing import List, Optional, Mapping, Any
from dataclasses import dataclass
import json
import httpx

from .base import BaseAPI, EndpointsConfig
from ..models.inputs import InstructRequest
from ..models.outputs import InstructResults
from ..models.common import FileMetadata
from ..exceptions import (
    ServerError,
    TransportTimeout,
    ConnectionFailure,
    BadRequest,
    from_http_response
)


@dataclass
class InstructEndpointsConfig(EndpointsConfig):
    main: str = ""


class InstructAPI(BaseAPI):
    """
    InstructAPI - Instruct-it inference endpoint for instruction-following models.

    This API accepts a text instruction plus optional template, file references,
    and output type hint, and returns a JSON-serializable object.
    """

    PATH = '/v1/instruct-it'
    METHOD = 'POST'
    CONTENT_TYPE = 'application/json'
    ENDPOINTS_CONFIG = InstructEndpointsConfig()
    SUCCESS_STATUS = "ok"

    def _build_prompt_context(
        self,
        text: str,
        template: Optional[str],
        files: Optional[List[FileMetadata]],
        output_type: Optional[str],
    ) -> dict[str, Any]:
        """Build the context dictionary for the prompt."""
        context: dict[str, Any] = {"instruction": text}

        if template is not None:
            try:
                parsed_template = json.loads(template)
                context["template"] = parsed_template
            except json.JSONDecodeError as exc:
                raise BadRequest(
                    f"Invalid JSON in template: {exc}",
                    payload={"template": template}
                )

        if files:
            context["files"] = [
                {"name": f.name, "has_content": f.content is not None}
                for f in files
            ]

        if output_type:
            context["output_type"] = output_type

        return context

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
            url_variables: Optional URL path variables.
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            InstructResults with the model's output.

        Raises:
            BadRequest: If template is not valid JSON.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error or invalid response.
        """
        if template is not None:
            try:
                json.loads(template)
            except json.JSONDecodeError as exc:
                raise BadRequest(
                    f"Invalid JSON in template: {exc}",
                    payload={"template": template}
                )

        body = InstructRequest(
            text=text,
            template=template,
            files=files,
            output_type=output_type,
        )

        path = f"{self.PATH}/{self.ENDPOINTS_CONFIG.main}".rstrip('/')
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
            self._log.debug("InstructAPI: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "InstructAPI: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
        except ValueError as exc:
            raw_text = response.text
            if self._log:
                self._log.debug("InstructAPI: Non-JSON response, wrapping as string output")
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
            url_variables: Optional URL path variables.
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            InstructResults with the model's output.

        Raises:
            BadRequest: If template is not valid JSON.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error or invalid response.
        """
        if template is not None:
            try:
                json.loads(template)
            except json.JSONDecodeError as exc:
                raise BadRequest(
                    f"Invalid JSON in template: {exc}",
                    payload={"template": template}
                )

        body = InstructRequest(
            text=text,
            template=template,
            files=files,
            output_type=output_type,
        )

        path = f"{self.PATH}/{self.ENDPOINTS_CONFIG.main}".rstrip('/')
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
            self._log.debug("InstructAPI: %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "InstructAPI: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
        except ValueError as exc:
            raw_text = response.text
            if self._log:
                self._log.debug("InstructAPI: Non-JSON response, wrapping as string output")
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
