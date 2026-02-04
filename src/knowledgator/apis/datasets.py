from typing import Any, Union, Mapping, BinaryIO, Optional
from dataclasses import dataclass

import httpx

from .base import BaseAPI, EndpointsConfig
from ..datasets import (
    Dataset,
    ListFormatsResponse,
    ListDatasetsResponse,
    DatasetCreationResponse,
)
from ..exceptions import (
    BadRequest,
    ServerError,
    TransportTimeout,
    ConnectionFailure,
    from_http_response,
)


@dataclass
class DatasetsEndpointsConfig(EndpointsConfig):
    """Endpoints configuration for Datasets API."""

    list: str = ""
    create: str = ""
    get: str = "{dataset_id}/"
    update: str = "{dataset_id}/"
    delete: str = "{dataset_id}/"
    upload: str = "{dataset_id}/files/upload/"
    list_formats: str = "../files/formats/datasets/"


class DatasetsAPI(BaseAPI):
    """
    DatasetsAPI - Manage datasets for NER, text classification, and relation extraction.

    This API provides CRUD operations for datasets including creating, listing,
    getting, updating, deleting datasets, and uploading files.
    """

    PATH = "/datasets/"
    METHOD = "GET"  # Default method, overridden per operation
    CONTENT_TYPE = "application/json"
    ENDPOINTS_CONFIG = DatasetsEndpointsConfig()

    def list(
        self,
        offset: int = 0,
        limit: int = 100,
        name: Optional[str] = None,
        type: Optional[str] = None,
        format: Optional[str] = None,
        status: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListDatasetsResponse:
        """
        List datasets with optional filtering.

        Args:
            offset: Pagination offset (default: 0).
            limit: Maximum number of results (default: 100, max: 10000).
            name: Filter by dataset name.
            type: Filter by service type (ner, text-classification, relation-extraction).
            format: Filter by dataset format (jsonl, csv, parquet, json).
            status: Filter by dataset status (pending, processing, ready, failed).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            ListDatasetsResponse with datasets list and pagination info.

        Raises:
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        # Build query parameters
        query_params = {"offset": offset, "limit": limit}
        if name is not None:
            query_params["name"] = name
        if type is not None:
            query_params["type"] = type
        if format is not None:
            query_params["format"] = format
        if status is not None:
            query_params["status"] = status

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list}".rstrip("/")
        request = self._make_request(
            method="GET",
            path=path,
            url_override=server_url,
            query_params=query_params,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.list: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.list: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListDatasetsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    def create(
        self,
        name: str,
        type: str,
        format: str,
        description: Optional[str] = None,
        file: Optional[Union[bytes, BinaryIO]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> DatasetCreationResponse:
        """
        Create a new dataset.

        Args:
            name: Dataset name (1-64 chars, alphanumeric with spaces, dots, parentheses, hyphens).
            type: Service type (ner, text-classification, relation-extraction).
            format: Dataset format (jsonl, csv, parquet, json).
            description: Optional dataset description.
            file: Optional file content to upload (bytes or file-like object).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            DatasetCreationResponse with dataset ID and metadata.

        Raises:
            BadRequest: If parameters are invalid.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        # Build form data payload
        data_payload = {
            "name": name,
            "type": type,
            "format": format,
        }

        if description is not None:
            data_payload["description"] = description

        # Build files payload if file is provided
        files_payload = None
        if file is not None:
            if isinstance(file, bytes):
                files_payload = {"file": ("file", file)}
            else:
                files_payload = {"file": ("file", file)}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.create}".rstrip("/")
        request = self._make_request(
            method="POST",
            path=path,
            url_override=server_url,
            data_body=data_payload,
            files=files_payload,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.create: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.create: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return DatasetCreationResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    def get(
        self,
        dataset_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> Dataset:
        """
        Get dataset details by ID.

        Args:
            dataset_id: Dataset identifier (2-64 chars, alphanumeric with hyphens).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            Dataset object with full details.

        Raises:
            BadRequest: If dataset_id is invalid or dataset doesn't exist.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.get}".format(dataset_id=dataset_id).rstrip("/")
        request = self._make_request(
            method="GET",
            path=path,
            url_override=server_url,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.get: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.get: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return Dataset(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    def update(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Update dataset metadata.

        Args:
            dataset_id: Dataset identifier (2-64 chars, alphanumeric with hyphens).
            name: New dataset name.
            description: New dataset description.
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Raises:
            BadRequest: If parameters are invalid or dataset doesn't exist.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        # Build JSON body
        json_body = {}
        if name is not None:
            json_body["name"] = name
        if description is not None:
            json_body["description"] = description

        if not json_body:
            raise BadRequest("At least one field (name or description) must be provided for update")

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.update}".format(dataset_id=dataset_id).rstrip("/")
        request = self._make_request(
            method="PATCH",
            path=path,
            url_override=server_url,
            json_body=json_body,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.update: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.update: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

    def delete(
        self,
        dataset_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Delete a dataset.

        Args:
            dataset_id: Dataset identifier (2-64 chars, alphanumeric with hyphens).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Raises:
            BadRequest: If dataset_id is invalid or dataset doesn't exist.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.delete}".format(dataset_id=dataset_id).rstrip("/")
        request = self._make_request(
            method="DELETE",
            path=path,
            url_override=server_url,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.delete: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.delete: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

    def upload(
        self,
        dataset_id: str,
        file: Union[bytes, BinaryIO],
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> DatasetCreationResponse:
        """
        Upload a file to an existing dataset.

        Args:
            dataset_id: Dataset identifier (2-64 chars, alphanumeric with hyphens).
            file: File content to upload (bytes or file-like object).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            DatasetCreationResponse with updated dataset metadata.

        Raises:
            BadRequest: If parameters are invalid or dataset doesn't exist.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        # Build files payload
        if isinstance(file, bytes):
            files_payload = {"file": ("file", file)}
        else:
            files_payload = {"file": ("file", file)}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.upload}".format(dataset_id=dataset_id).rstrip("/")
        request = self._make_request(
            method="POST",
            path=path,
            url_override=server_url,
            files=files_payload,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.upload: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.upload: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return DatasetCreationResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    def list_formats(
        self,
        offset: int = 0,
        limit: int = 100,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListFormatsResponse:
        """
        List available dataset formats.

        Args:
            offset: Pagination offset (default: 0).
            limit: Maximum number of results (default: 100, max: 10000).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            ListFormatsResponse with available formats.

        Raises:
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        # Build query parameters
        query_params = {"offset": offset, "limit": limit}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list_formats}".rstrip("/")
        request = self._make_request(
            method="GET",
            path=path,
            url_override=server_url,
            query_params=query_params,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.list_formats: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.list_formats: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListFormatsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    # Async methods

    async def list_async(
        self,
        offset: int = 0,
        limit: int = 100,
        name: Optional[str] = None,
        type: Optional[str] = None,
        format: Optional[str] = None,
        status: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListDatasetsResponse:
        """Async version of list()."""
        query_params = {"offset": offset, "limit": limit}
        if name is not None:
            query_params["name"] = name
        if type is not None:
            query_params["type"] = type
        if format is not None:
            query_params["format"] = format
        if status is not None:
            query_params["status"] = status

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list}".rstrip("/")
        request = self._make_request(
            method="GET",
            path=path,
            url_override=server_url,
            query_params=query_params,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.list_async: %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.list_async: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListDatasetsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    async def create_async(
        self,
        name: str,
        type: str,
        format: str,
        description: Optional[str] = None,
        file: Optional[Union[bytes, BinaryIO]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> DatasetCreationResponse:
        """Async version of create()."""
        data_payload = {
            "name": name,
            "type": type,
            "format": format,
        }

        if description is not None:
            data_payload["description"] = description

        files_payload = None
        if file is not None:
            if isinstance(file, bytes):
                files_payload = {"file": ("file", file)}
            else:
                files_payload = {"file": ("file", file)}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.create}".rstrip("/")
        request = self._make_request(
            method="POST",
            path=path,
            url_override=server_url,
            data_body=data_payload,
            files=files_payload,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.create_async: %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.create_async: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return DatasetCreationResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    async def get_async(
        self,
        dataset_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> Dataset:
        """Async version of get()."""
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.get}".format(dataset_id=dataset_id).rstrip("/")
        request = self._make_request(
            method="GET",
            path=path,
            url_override=server_url,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.get_async: %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.get_async: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return Dataset(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    async def update_async(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Async version of update()."""
        json_body = {}
        if name is not None:
            json_body["name"] = name
        if description is not None:
            json_body["description"] = description

        if not json_body:
            raise BadRequest("At least one field (name or description) must be provided for update")

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.update}".format(dataset_id=dataset_id).rstrip("/")
        request = self._make_request(
            method="PATCH",
            path=path,
            url_override=server_url,
            json_body=json_body,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.update_async: %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.update_async: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

    async def delete_async(
        self,
        dataset_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Async version of delete()."""
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.delete}".format(dataset_id=dataset_id).rstrip("/")
        request = self._make_request(
            method="DELETE",
            path=path,
            url_override=server_url,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.delete_async: %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.delete_async: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

    async def upload_async(
        self,
        dataset_id: str,
        file: Union[bytes, BinaryIO],
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> DatasetCreationResponse:
        """Async version of upload()."""
        if isinstance(file, bytes):
            files_payload = {"file": ("file", file)}
        else:
            files_payload = {"file": ("file", file)}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.upload}".format(dataset_id=dataset_id).rstrip("/")
        request = self._make_request(
            method="POST",
            path=path,
            url_override=server_url,
            files=files_payload,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.upload_async: %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.upload_async: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return DatasetCreationResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    async def list_formats_async(
        self,
        offset: int = 0,
        limit: int = 100,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListFormatsResponse:
        """Async version of list_formats()."""
        query_params = {"offset": offset, "limit": limit}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list_formats}".rstrip("/")
        request = self._make_request(
            method="GET",
            path=path,
            url_override=server_url,
            query_params=query_params,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.list_formats_async: %s %s", request.method, request.url)

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug("DatasetsAPI.list_formats_async: %s ← %s", response.status_code, request.url)

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListFormatsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc
