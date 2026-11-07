from typing import Any, Union, Mapping, BinaryIO, Optional, List, Iterator, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
import json
import httpx

from .base import BaseAPI, EndpointsConfig
from ..datasets import (
    Data,
    Dataset,
    ListDataResponse,
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
    download: str = "{dataset_id}/download/"
    generate_plan: str = "generate/plan/"
    generate: str = "generate/"
    list_formats: str = "../files/formats/datasets/"

    # Data endpoints
    list_data: str = "../data/"
    create_data: str = "../data/"
    get_data: str = "../data/{data_id}/"
    update_data: str = "../data/{data_id}/"
    delete_data: str = "../data/{data_id}/"
    upload_data: str = "../data/{data_id}/files/upload/"


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
            self._log.debug(
                "DatasetsAPI.list: %s ← %s", response.status_code, request.url
            )

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
        short_description: Optional[str] = None,
        parent_dataset_id: Optional[str] = None,
        file: Optional[Union[bytes, BinaryIO]] = None,
        file_path: Optional[Union[str, Path]] = None,
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
            short_description: Optional short dataset description (max 1024 characters).
            parent_dataset_id: Optional parent dataset UUID for version tracking.
            file: Optional file content to upload (bytes or file-like object).
            file_path: Optional local filesystem path to stream dataset from.
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

        if short_description is not None:
            data_payload["short_description"] = short_description

        if parent_dataset_id is not None and parent_dataset_id != "none":
            data_payload["parent_dataset_id"] = parent_dataset_id

        # Build files payload
        files_payload = None
        opened_file = None
        if file_path is not None:
            p = Path(file_path).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"Dataset file not found: {p}")
            opened_file = open(p, "rb")
            files_payload = {"file": (p.name, opened_file)}
        elif file is not None:
            files_payload = {"file": ("dataset.json", file)}

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
            try:
                response = self.send(request)
            except httpx.TimeoutException as exc:
                raise TransportTimeout("Request timed-out") from exc
            except httpx.RequestError as exc:
                raise ConnectionFailure("Transport error") from exc

            if self._log:
                self._log.debug(
                    "DatasetsAPI.create: %s ← %s", response.status_code, request.url
                )

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
        finally:
            if opened_file is not None:
                opened_file.close()

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
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.get}".format(
            dataset_id=dataset_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.get: %s ← %s", response.status_code, request.url
            )

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
        short_description: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Update dataset metadata.

        Args:
            dataset_id: Dataset identifier (2-64 chars, alphanumeric with hyphens).
            name: New dataset name.
            short_description: New short dataset description (max 1024 characters).
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
        if short_description is not None:
            json_body["short_description"] = short_description

        if not json_body:
            raise BadRequest(
                "At least one field (name or short_description) must be provided for update"
            )

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.update}".format(
            dataset_id=dataset_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.update: %s ← %s", response.status_code, request.url
            )

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
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.delete}".format(
            dataset_id=dataset_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.delete: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

    def upload(
        self,
        dataset_id: str,
        file: Optional[Union[bytes, BinaryIO]] = None,
        file_path: Optional[Union[str, Path]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> DatasetCreationResponse:
        """
        Upload a file to an existing dataset.

        Args:
            dataset_id: Dataset identifier (2-64 chars, alphanumeric with hyphens).
            file: File content to upload (bytes or file-like object).
            file_path: Optional local filesystem path to stream file from.
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
        opened_file = None
        if file_path is not None:
            p = Path(file_path).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"Dataset file not found: {p}")
            opened_file = open(p, "rb")
            files_payload = {"file": (p.name, opened_file)}
        elif file is not None:
            files_payload = {"file": ("dataset.json", file)}
        else:
            raise ValueError("Either 'file' or 'file_path' must be provided.")

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.upload}".format(
            dataset_id=dataset_id
        ).rstrip("/")
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
            try:
                response = self.send(request)
            except httpx.TimeoutException as exc:
                raise TransportTimeout("Request timed-out") from exc
            except httpx.RequestError as exc:
                raise ConnectionFailure("Transport error") from exc

            if self._log:
                self._log.debug(
                    "DatasetsAPI.upload: %s ← %s", response.status_code, request.url
                )

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
        finally:
            if opened_file is not None:
                opened_file.close()

    def stream_download(
        self,
        dataset_id: str,
        chunk_size: int = 65536,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> Iterator[bytes]:
        """
        Stream dataset file content by ID in chunks.
        Automatically follows HTTP 307 storage redirects and S3 URLs.

        Args:
            dataset_id: Dataset identifier.
            chunk_size: Size of byte chunks to yield (default: 65536).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Yields:
            Byte chunks of the dataset.

        Raises:
            BadRequest: If dataset_id is invalid.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.download}".format(
            dataset_id=dataset_id
        ).rstrip("/")
        request = self._make_request(
            method="GET",
            path=path,
            url_override=server_url,
            headers=http_headers,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug("DatasetsAPI.stream_download: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.stream_download: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 400:
            raise from_http_response(response)

        # Check if the response is a JSON storage redirect pointer {"url": "https://..."}
        content_type = response.headers.get("content-type", "").lower()
        target_url = None
        initial_content = response.content

        if "application/json" in content_type:
            try:
                parsed = json.loads(initial_content.decode("utf-8"))
                if isinstance(parsed, dict) and isinstance(parsed.get("url"), str):
                    target_url = parsed["url"]
                elif isinstance(parsed, str) and parsed.startswith(("http://", "https://")):
                    target_url = parsed
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if self._log:
                    self._log.debug("Content marked as application/json was not a redirect URL: %s", exc)

        if target_url:
            if self._log:
                self._log.debug("DatasetsAPI.stream_download: Following storage URL: %s", target_url)
            storage_req = httpx.Request("GET", target_url)
            try:
                storage_resp = self.send(storage_req)
            except httpx.TimeoutException as exc:
                raise TransportTimeout("Storage request timed-out") from exc
            except httpx.RequestError as exc:
                raise ConnectionFailure("Storage transport error") from exc

            if not 200 <= storage_resp.status_code < 400:
                raise from_http_response(storage_resp)

            # Yield chunks from storage response content
            data = storage_resp.content
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]
        else:
            # Yield chunks from direct response content
            for i in range(0, len(initial_content), chunk_size):
                yield initial_content[i:i + chunk_size]

    def download(
        self,
        dataset_id: str,
        file_path: Optional[Union[str, Path]] = None,
        return_data: bool = False,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> Union[bytes, Any, Path]:
        """
        Download dataset file content by ID.
        Uses stream_download under the hood. If file_path is given, streams chunks
        directly to disk with O(1) memory.

        Args:
            dataset_id: Dataset identifier.
            file_path: Optional local destination file path to save chunks directly to disk.
            return_data: If True, parses JSON response into Python object (list/dict). If False, returns raw bytes or saved Path.
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            Path if file_path is specified, parsed JSON object if return_data=True, otherwise raw bytes.
        """
        chunks = self.stream_download(
            dataset_id=dataset_id,
            server_url=server_url,
            timeout_ms=timeout_ms,
            http_headers=http_headers,
        )

        if file_path is not None:
            dest = Path(file_path).expanduser()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)
            if return_data:
                with open(dest, "r", encoding="utf-8") as f:
                    return json.load(f)
            return dest

        all_bytes = b"".join(chunks)
        if return_data:
            try:
                return json.loads(all_bytes.decode("utf-8"))
            except Exception:
                return all_bytes

        return all_bytes

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
            self._log.debug(
                "DatasetsAPI.list_formats: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.list_formats: %s ← %s", response.status_code, request.url
            )

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

    def list_data(
        self,
        offset: int = 0,
        limit: int = 100,
        name: Optional[str] = None,
        type: Optional[str] = None,
        format: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListDataResponse:
        """
        List raw data with optional filtering.

        Args:
            offset: Pagination offset (default: 0).
            limit: Maximum number of results (default: 100, max: 1000).
            name: Filter by data name.
            type: Filter by service type.
            format: Filter by data format.
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            ListDataResponse with data list and pagination info.

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

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list_data}".rstrip("/")
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
            self._log.debug("DatasetsAPI.list_data: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.list_data: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListDataResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    def create_data(
        self,
        name: str,
        type: str,
        format: str,
        short_description: Optional[str] = None,
        file: Optional[Union[bytes, BinaryIO]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> DatasetCreationResponse:
        """
        Create a new raw data entry.
        """
        data_payload = {
            "name": name,
            "type": type,
            "format": format,
        }

        if short_description is not None:
            data_payload["short_description"] = short_description

        files_payload = None
        if file is not None:
            files_payload = {"file": ("file", file)}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.create_data}".rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.create_data: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.create_data: %s ← %s", response.status_code, request.url
            )

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

    def get_data(
        self,
        data_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> Data:
        """
        Get raw data details by ID.
        """
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.get_data}".format(
            data_id=data_id
        ).rstrip("/")
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
            self._log.debug("DatasetsAPI.get_data: %s %s", request.method, request.url)

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.get_data: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return Data(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    def update_data(
        self,
        data_id: str,
        name: Optional[str] = None,
        short_description: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Update raw data metadata.
        """
        json_body = {}
        if name is not None:
            json_body["name"] = name
        if short_description is not None:
            json_body["short_description"] = short_description

        if not json_body:
            raise BadRequest(
                "At least one field (name or short_description) must be provided for update"
            )

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.update_data}".format(
            data_id=data_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.update_data: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.update_data: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

    def delete_data(
        self,
        data_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Delete a raw data entry.
        """
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.delete_data}".format(
            data_id=data_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.delete_data: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.delete_data: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

    def upload_data(
        self,
        data_id: str,
        file: Union[bytes, BinaryIO],
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> DatasetCreationResponse:
        """
        Upload a file to an existing raw data entry.
        """
        files_payload = {"file": ("file", file)}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.upload_data}".format(
            data_id=data_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.upload_data: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.upload_data: %s ← %s", response.status_code, request.url
            )

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
            self._log.debug(
                "DatasetsAPI.list_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.list_async: %s ← %s", response.status_code, request.url
            )

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
        short_description: Optional[str] = None,
        parent_dataset_id: Optional[str] = None,
        file: Optional[Union[bytes, BinaryIO]] = None,
        file_path: Optional[Union[str, Path]] = None,
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

        if short_description is not None:
            data_payload["short_description"] = short_description

        if parent_dataset_id is not None and parent_dataset_id != "none":
            data_payload["parent_dataset_id"] = parent_dataset_id

        files_payload = None
        opened_file = None
        if file_path is not None:
            p = Path(file_path).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"Dataset file not found: {p}")
            opened_file = open(p, "rb")
            files_payload = {"file": (p.name, opened_file)}
        elif file is not None:
            files_payload = {"file": ("dataset.json", file)}

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
            self._log.debug(
                "DatasetsAPI.create_async: %s %s", request.method, request.url
            )

        try:
            try:
                response = await self.send_async(request)
            except httpx.TimeoutException as exc:
                raise TransportTimeout("Request timed-out") from exc
            except httpx.RequestError as exc:
                raise ConnectionFailure("Transport error") from exc

            if self._log:
                self._log.debug(
                    "DatasetsAPI.create_async: %s ← %s", response.status_code, request.url
                )

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
        finally:
            if opened_file is not None:
                opened_file.close()

    async def get_async(
        self,
        dataset_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> Dataset:
        """Async version of get()."""
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.get}".format(
            dataset_id=dataset_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.get_async: %s ← %s", response.status_code, request.url
            )

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
        short_description: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Async version of update()."""
        json_body = {}
        if name is not None:
            json_body["name"] = name
        if short_description is not None:
            json_body["short_description"] = short_description

        if not json_body:
            raise BadRequest(
                "At least one field (name or short_description) must be provided for update"
            )

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.update}".format(
            dataset_id=dataset_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.update_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.update_async: %s ← %s", response.status_code, request.url
            )

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
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.delete}".format(
            dataset_id=dataset_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.delete_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.delete_async: %s ← %s", response.status_code, request.url
            )

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

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.upload}".format(
            dataset_id=dataset_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.upload_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.upload_async: %s ← %s", response.status_code, request.url
            )

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

    async def stream_download_async(
        self,
        dataset_id: str,
        chunk_size: int = 65536,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> AsyncIterator[bytes]:
        """
        Async stream dataset file content by ID in chunks.
        Automatically follows HTTP 307 storage redirects and S3 URLs.

        Args:
            dataset_id: Dataset identifier.
            chunk_size: Size of byte chunks to yield (default: 65536).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Yields:
            Byte chunks of the dataset.

        Raises:
            BadRequest: If dataset_id is invalid.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.download}".format(
            dataset_id=dataset_id
        ).rstrip("/")
        request = self._make_request(
            method="GET",
            path=path,
            url_override=server_url,
            headers=http_headers,
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug(
                "DatasetsAPI.stream_download_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.stream_download_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 400:
            raise from_http_response(response)

        # Check if the response is a JSON storage redirect pointer {"url": "https://..."}
        content_type = response.headers.get("content-type", "").lower()
        target_url = None
        initial_content = response.content

        if "application/json" in content_type:
            try:
                parsed = json.loads(initial_content.decode("utf-8"))
                if isinstance(parsed, dict) and isinstance(parsed.get("url"), str):
                    target_url = parsed["url"]
                elif isinstance(parsed, str) and parsed.startswith(("http://", "https://")):
                    target_url = parsed
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if self._log:
                    self._log.debug("Content marked as application/json was not a redirect URL: %s", exc)

        if target_url:
            if self._log:
                self._log.debug("DatasetsAPI.stream_download_async: Following storage URL: %s", target_url)
            storage_req = httpx.Request("GET", target_url)
            try:
                storage_resp = await self.send_async(storage_req)
            except httpx.TimeoutException as exc:
                raise TransportTimeout("Storage request timed-out") from exc
            except httpx.RequestError as exc:
                raise ConnectionFailure("Storage transport error") from exc

            if not 200 <= storage_resp.status_code < 400:
                raise from_http_response(storage_resp)

            # Yield chunks from storage response content
            data = storage_resp.content
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]
        else:
            # Yield chunks from direct response content
            for i in range(0, len(initial_content), chunk_size):
                yield initial_content[i:i + chunk_size]

    async def download_async(
        self,
        dataset_id: str,
        file_path: Optional[Union[str, Path]] = None,
        return_data: bool = False,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> Union[bytes, Any, Path]:
        """
        Async version of download().
        Uses stream_download_async under the hood. If file_path is given, streams chunks
        directly to disk with O(1) memory.

        Args:
            dataset_id: Dataset identifier.
            file_path: Optional local destination file path to save chunks directly to disk.
            return_data: If True, parses JSON response into Python object (list/dict). If False, returns raw bytes or saved Path.
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            Path if file_path is specified, parsed JSON object if return_data=True, otherwise raw bytes.
        """
        chunks: List[bytes] = []
        async for chunk in self.stream_download_async(
            dataset_id=dataset_id,
            server_url=server_url,
            timeout_ms=timeout_ms,
            http_headers=http_headers,
        ):
            chunks.append(chunk)

        if file_path is not None:
            dest = Path(file_path).expanduser()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)
            if return_data:
                with open(dest, "r", encoding="utf-8") as f:
                    return json.load(f)
            return dest

        all_bytes = b"".join(chunks)
        if return_data:
            try:
                return json.loads(all_bytes.decode("utf-8"))
            except Exception:
                return all_bytes

        return all_bytes

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
            self._log.debug(
                "DatasetsAPI.list_formats_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.list_formats_async: %s ← %s",
                response.status_code,
                request.url,
            )

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

    async def list_data_async(
        self,
        offset: int = 0,
        limit: int = 100,
        name: Optional[str] = None,
        type: Optional[str] = None,
        format: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListDataResponse:
        """Async version of list_data()."""
        query_params = {"offset": offset, "limit": limit}
        if name is not None:
            query_params["name"] = name
        if type is not None:
            query_params["type"] = type
        if format is not None:
            query_params["format"] = format

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list_data}".rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.list_data_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.list_data_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListDataResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    async def create_data_async(
        self,
        name: str,
        type: str,
        format: str,
        short_description: Optional[str] = None,
        file: Optional[Union[bytes, BinaryIO]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> DatasetCreationResponse:
        """Async version of create_data()."""
        data_payload = {
            "name": name,
            "type": type,
            "format": format,
        }

        if short_description is not None:
            data_payload["short_description"] = short_description

        files_payload = None
        if file is not None:
            files_payload = {"file": ("file", file)}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.create_data}".rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.create_data_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.create_data_async: %s ← %s",
                response.status_code,
                request.url,
            )

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

    async def get_data_async(
        self,
        data_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> Data:
        """Async version of get_data()."""
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.get_data}".format(
            data_id=data_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.get_data_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.get_data_async: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return Data(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    async def update_data_async(
        self,
        data_id: str,
        name: Optional[str] = None,
        short_description: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Async version of update_data()."""
        json_body = {}
        if name is not None:
            json_body["name"] = name
        if short_description is not None:
            json_body["short_description"] = short_description

        if not json_body:
            raise BadRequest(
                "At least one field (name or short_description) must be provided for update"
            )

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.update_data}".format(
            data_id=data_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.update_data_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.update_data_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

    async def delete_data_async(
        self,
        data_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Async version of delete_data()."""
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.delete_data}".format(
            data_id=data_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.delete_data_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.delete_data_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

    async def upload_data_async(
        self,
        data_id: str,
        file: Union[bytes, BinaryIO],
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> DatasetCreationResponse:
        """Async version of upload_data()."""
        files_payload = {"file": ("file", file)}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.upload_data}".format(
            data_id=data_id
        ).rstrip("/")
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
            self._log.debug(
                "DatasetsAPI.upload_data_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "DatasetsAPI.upload_data_async: %s ← %s",
                response.status_code,
                request.url,
            )

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

    def generate_plan(
        self,
        prompt: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> dict[str, Any]:
        """Generate a dataset generation plan from a prompt."""
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.generate_plan}".rstrip("/")
        request = self._make_request(
            method="POST",
            path=path,
            url_override=server_url,
            json_body={"prompt": prompt},
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            return response.json()
        except Exception as exc:
            raise ServerError(
                "Malformed response",
                status_code=response.status_code,
                payload={"raw": response.text},
            ) from exc

    async def generate_plan_async(
        self,
        prompt: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> dict[str, Any]:
        """Generate a dataset generation plan from a prompt asynchronously."""
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.generate_plan}".rstrip("/")
        request = self._make_request(
            method="POST",
            path=path,
            url_override=server_url,
            json_body={"prompt": prompt},
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            return response.json()
        except Exception as exc:
            raise ServerError(
                "Malformed response",
                status_code=response.status_code,
                payload={"raw": response.text},
            ) from exc

    def generate(
        self,
        name: str,
        prompt: str,
        type: str,
        num_samples: int,
        domains: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
        json_template: Optional[str] = None,
        ner_output_format: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> dict[str, Any]:
        """Start async dataset generation pipeline."""
        body: dict[str, Any] = {
            "name": name,
            "prompt": prompt,
            "type": type,
            "num_samples": num_samples,
        }
        if domains is not None:
            body["domains"] = domains
        if classes is not None:
            body["classes"] = classes
        if entity_types is not None:
            body["entity_types"] = entity_types
        if json_template is not None:
            body["json_template"] = json_template
        if ner_output_format is not None:
            body["ner_output_format"] = ner_output_format

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.generate}".rstrip("/")
        request = self._make_request(
            method="POST",
            path=path,
            url_override=server_url,
            json_body=body,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            return response.json()
        except Exception as exc:
            raise ServerError(
                "Malformed response",
                status_code=response.status_code,
                payload={"raw": response.text},
            ) from exc

    async def generate_async(
        self,
        name: str,
        prompt: str,
        type: str,
        num_samples: int,
        domains: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
        json_template: Optional[str] = None,
        ner_output_format: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> dict[str, Any]:
        """Start async dataset generation pipeline asynchronously."""
        body: dict[str, Any] = {
            "name": name,
            "prompt": prompt,
            "type": type,
            "num_samples": num_samples,
        }
        if domains is not None:
            body["domains"] = domains
        if classes is not None:
            body["classes"] = classes
        if entity_types is not None:
            body["entity_types"] = entity_types
        if json_template is not None:
            body["json_template"] = json_template
        if ner_output_format is not None:
            body["ner_output_format"] = ner_output_format

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.generate}".rstrip("/")
        request = self._make_request(
            method="POST",
            path=path,
            url_override=server_url,
            json_body=body,
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            return response.json()
        except Exception as exc:
            raise ServerError(
                "Malformed response",
                status_code=response.status_code,
                payload={"raw": response.text},
            ) from exc
