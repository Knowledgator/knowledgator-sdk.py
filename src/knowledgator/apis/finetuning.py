from typing import Any, Mapping, Optional
from dataclasses import dataclass

import httpx

from .base import BaseAPI, EndpointsConfig
from ..exceptions import (
    BadRequest,
    ConnectionFailure,
    ServerError,
    TransportTimeout,
    from_http_response,
)
from ..finetuning import (
    TrainingModel,
    ListModelsResponse,
    ObjectCreation,
    TrainingObject,
    ListTrainingsResponse,
    ListTagsResponse,
    ListEventsResponse,
    ListMetricsResponse,
)


@dataclass
class FinetuningEndpointsConfig(EndpointsConfig):
    """Endpoints configuration for Finetuning API."""

    # Model Management
    create_model: str = "models/"
    list_models: str = "models/"
    get_model: str = "models/{model_id}/"
    list_model_types: str = "models/types/"

    # Training Management
    start_training: str = ""
    list_trainings: str = ""
    get_training: str = "{training_id}/"
    delete_training: str = "{training_id}/"
    stop_training: str = "{training_id}/stop/"

    # Training Monitoring
    list_events: str = "{training_id}/events/"
    get_metrics: str = "{training_id}/metrics/{metric}/"


class FinetuningAPI(BaseAPI):
    """
    FinetuningAPI - Fine-tune models for NER, text classification, and relation extraction.

    This API provides operations for managing base models, training jobs, and monitoring
    training progress including events and metrics.
    """

    PATH = "/trainings/"
    METHOD = "GET"  # Default method, overridden per operation
    CONTENT_TYPE = "application/json"
    ENDPOINTS_CONFIG = FinetuningEndpointsConfig()

    # ============================================================================
    # Model Management - Synchronous Methods
    # ============================================================================

    def create_base_model(
        self,
        name: str,
        type: str,
        description: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ObjectCreation:
        """
        Create a new base model for training.

        Args:
            name: Model name (2-64 chars, alphanumeric with hyphens).
            type: Service type (ner, text-classification, relation-extraction).
            description: Optional model description.
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            ObjectCreation with model ID and creation timestamp.

        Raises:
            BadRequest: If parameters are invalid.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        json_body = {"name": name, "type": type}
        if description is not None:
            json_body["description"] = description

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.create_model}".rstrip("/")
        request = self._make_request(
            method="POST",
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
                "FinetuningAPI.create_base_model: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.create_base_model: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ObjectCreation(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    def list_models(
        self,
        offset: int = 0,
        limit: int = 100,
        type: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListModelsResponse:
        """
        List available base models with optional filtering.

        Args:
            offset: Pagination offset (default: 0).
            limit: Maximum number of results (default: 100, max: 10000).
            type: Filter by service type (ner, text-classification, relation-extraction).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            ListModelsResponse with models list and pagination info.

        Raises:
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        query_params = {"offset": offset, "limit": limit}
        if type is not None:
            query_params["type"] = type

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list_models}".rstrip("/")
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
                "FinetuningAPI.list_models: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.list_models: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListModelsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    def get_model(
        self,
        model_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> TrainingModel:
        """
        Get model details by ID.

        Args:
            model_id: Model identifier (2-64 chars, alphanumeric with hyphens).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            TrainingModel object with full details.

        Raises:
            BadRequest: If model_id is invalid or model doesn't exist.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.get_model}".format(
            model_id=model_id
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
                "FinetuningAPI.get_model: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.get_model: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return TrainingModel(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    def list_model_types(
        self,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListTagsResponse:
        """
        List available model types (service types).

        Args:
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            ListTagsResponse with available service types.

        Raises:
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list_model_types}".rstrip("/")
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
                "FinetuningAPI.list_model_types: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.list_model_types: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListTagsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    # ============================================================================
    # Training Management - Synchronous Methods
    # ============================================================================

    def start_training(
        self,
        model_id: str,
        dataset_id: str,
        hyperparameters: Optional[dict[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> TrainingObject:
        """
        Start a new training job.

        Args:
            model_id: Base model identifier (2-64 chars, alphanumeric with hyphens).
            dataset_id: Dataset identifier (2-64 chars, alphanumeric with hyphens).
            hyperparameters: Optional training hyperparameters (epochs, batch_size, learning_rate, etc.).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            TrainingObject with training job details.

        Raises:
            BadRequest: If parameters are invalid.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        json_body = {"model_id": model_id, "dataset_id": dataset_id}
        if hyperparameters is not None:
            json_body["hyperparameters"] = hyperparameters

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.start_training}".rstrip("/")
        request = self._make_request(
            method="POST",
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
                "FinetuningAPI.start_training: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.start_training: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return TrainingObject(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    def list_trainings(
        self,
        offset: int = 0,
        limit: int = 100,
        model_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        status: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListTrainingsResponse:
        """
        List training jobs with optional filtering.

        Args:
            offset: Pagination offset (default: 0).
            limit: Maximum number of results (default: 100, max: 10000).
            model_id: Filter by base model ID.
            dataset_id: Filter by dataset ID.
            status: Filter by training status (pending, running, successful, failed, stopped).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            ListTrainingsResponse with trainings list and pagination info.

        Raises:
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        query_params = {"offset": offset, "limit": limit}
        if model_id is not None:
            query_params["model_id"] = model_id
        if dataset_id is not None:
            query_params["dataset_id"] = dataset_id
        if status is not None:
            query_params["status"] = status

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list_trainings}".rstrip("/")
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
                "FinetuningAPI.list_trainings: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.list_trainings: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListTrainingsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    def get_training(
        self,
        training_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> TrainingObject:
        """
        Get training job details by ID.

        Args:
            training_id: Training job identifier (2-64 chars, alphanumeric with hyphens).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            TrainingObject with full training details.

        Raises:
            BadRequest: If training_id is invalid or training doesn't exist.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.get_training}".format(
            training_id=training_id
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
                "FinetuningAPI.get_training: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.get_training: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return TrainingObject(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    def delete_training(
        self,
        training_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Delete a training job.

        Args:
            training_id: Training job identifier (2-64 chars, alphanumeric with hyphens).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Raises:
            BadRequest: If training_id is invalid or training doesn't exist.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.delete_training}".format(
            training_id=training_id
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
                "FinetuningAPI.delete_training: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.delete_training: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

    def stop_training(
        self,
        training_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Stop a running training job.

        Args:
            training_id: Training job identifier (2-64 chars, alphanumeric with hyphens).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Raises:
            BadRequest: If training_id is invalid or training doesn't exist.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.stop_training}".format(
            training_id=training_id
        ).rstrip("/")
        request = self._make_request(
            method="POST",
            path=path,
            url_override=server_url,
            json_body={},  # Empty body for POST
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug(
                "FinetuningAPI.stop_training: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.stop_training: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

    # ============================================================================
    # Training Monitoring - Synchronous Methods
    # ============================================================================

    def list_events(
        self,
        training_id: str,
        offset: int = 0,
        limit: int = 100,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListEventsResponse:
        """
        List events for a training job.

        Args:
            training_id: Training job identifier (2-64 chars, alphanumeric with hyphens).
            offset: Pagination offset (default: 0).
            limit: Maximum number of results (default: 100, max: 10000).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            ListEventsResponse with events list and pagination info.

        Raises:
            BadRequest: If training_id is invalid or training doesn't exist.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        query_params = {"offset": offset, "limit": limit}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list_events}".format(
            training_id=training_id
        ).rstrip("/")
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
                "FinetuningAPI.list_events: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.list_events: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListEventsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    def get_metrics(
        self,
        training_id: str,
        metric: str,
        offset: int = 0,
        limit: int = 100,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListMetricsResponse:
        """
        Get training metrics for a specific metric type.

        Args:
            training_id: Training job identifier (2-64 chars, alphanumeric with hyphens).
            metric: Metric name (currently only 'loss' is supported).
            offset: Pagination offset (default: 0).
            limit: Maximum number of results (default: 100, max: 10000).
            server_url: Optional server URL override.
            timeout_ms: Optional timeout in milliseconds.
            http_headers: Optional additional HTTP headers.

        Returns:
            ListMetricsResponse with metric data points and pagination info.

        Raises:
            BadRequest: If parameters are invalid or training doesn't exist.
            TransportTimeout: If the request times out.
            ConnectionFailure: If there's a transport error.
            ServerError: If the server returns an error.
        """
        query_params = {"offset": offset, "limit": limit}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.get_metrics}".format(
            training_id=training_id, metric=metric
        ).rstrip("/")
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
                "FinetuningAPI.get_metrics: %s %s", request.method, request.url
            )

        try:
            response = self.send(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.get_metrics: %s ← %s", response.status_code, request.url
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListMetricsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    # ============================================================================
    # Model Management - Asynchronous Methods
    # ============================================================================

    async def create_base_model_async(
        self,
        name: str,
        type: str,
        description: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ObjectCreation:
        """Async version of create_base_model()."""
        json_body = {"name": name, "type": type}
        if description is not None:
            json_body["description"] = description

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.create_model}".rstrip("/")
        request = self._make_request(
            method="POST",
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
                "FinetuningAPI.create_base_model_async: %s %s",
                request.method,
                request.url,
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.create_base_model_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ObjectCreation(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    async def list_models_async(
        self,
        offset: int = 0,
        limit: int = 100,
        type: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListModelsResponse:
        """Async version of list_models()."""
        query_params = {"offset": offset, "limit": limit}
        if type is not None:
            query_params["type"] = type

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list_models}".rstrip("/")
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
                "FinetuningAPI.list_models_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.list_models_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListModelsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    async def get_model_async(
        self,
        model_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> TrainingModel:
        """Async version of get_model()."""
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.get_model}".format(
            model_id=model_id
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
                "FinetuningAPI.get_model_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.get_model_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return TrainingModel(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    async def list_model_types_async(
        self,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListTagsResponse:
        """Async version of list_model_types()."""
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list_model_types}".rstrip("/")
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
                "FinetuningAPI.list_model_types_async: %s %s",
                request.method,
                request.url,
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.list_model_types_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListTagsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    # ============================================================================
    # Training Management - Asynchronous Methods
    # ============================================================================

    async def start_training_async(
        self,
        model_id: str,
        dataset_id: str,
        hyperparameters: Optional[dict[str, Any]] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> TrainingObject:
        """Async version of start_training()."""
        json_body = {"model_id": model_id, "dataset_id": dataset_id}
        if hyperparameters is not None:
            json_body["hyperparameters"] = hyperparameters

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.start_training}".rstrip("/")
        request = self._make_request(
            method="POST",
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
                "FinetuningAPI.start_training_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.start_training_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return TrainingObject(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    async def list_trainings_async(
        self,
        offset: int = 0,
        limit: int = 100,
        model_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        status: Optional[str] = None,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListTrainingsResponse:
        """Async version of list_trainings()."""
        query_params = {"offset": offset, "limit": limit}
        if model_id is not None:
            query_params["model_id"] = model_id
        if dataset_id is not None:
            query_params["dataset_id"] = dataset_id
        if status is not None:
            query_params["status"] = status

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list_trainings}".rstrip("/")
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
                "FinetuningAPI.list_trainings_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.list_trainings_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListTrainingsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    async def get_training_async(
        self,
        training_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> TrainingObject:
        """Async version of get_training()."""
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.get_training}".format(
            training_id=training_id
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
                "FinetuningAPI.get_training_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.get_training_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return TrainingObject(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    async def delete_training_async(
        self,
        training_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Async version of delete_training()."""
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.delete_training}".format(
            training_id=training_id
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
                "FinetuningAPI.delete_training_async: %s %s",
                request.method,
                request.url,
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.delete_training_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

    async def stop_training_async(
        self,
        training_id: str,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Async version of stop_training()."""
        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.stop_training}".format(
            training_id=training_id
        ).rstrip("/")
        request = self._make_request(
            method="POST",
            path=path,
            url_override=server_url,
            json_body={},
            headers=http_headers,
            accept="application/json",
            timeout_ms=timeout_ms,
            user_agent=self.configs.user_agent,
        )

        if self._log:
            self._log.debug(
                "FinetuningAPI.stop_training_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.stop_training_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

    # ============================================================================
    # Training Monitoring - Asynchronous Methods
    # ============================================================================

    async def list_events_async(
        self,
        training_id: str,
        offset: int = 0,
        limit: int = 100,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListEventsResponse:
        """Async version of list_events()."""
        query_params = {"offset": offset, "limit": limit}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.list_events}".format(
            training_id=training_id
        ).rstrip("/")
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
                "FinetuningAPI.list_events_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.list_events_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListEventsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc

    async def get_metrics_async(
        self,
        training_id: str,
        metric: str,
        offset: int = 0,
        limit: int = 100,
        server_url: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        http_headers: Optional[Mapping[str, str]] = None,
    ) -> ListMetricsResponse:
        """Async version of get_metrics()."""
        query_params = {"offset": offset, "limit": limit}

        path = f"{self.PATH}{self.ENDPOINTS_CONFIG.get_metrics}".format(
            training_id=training_id, metric=metric
        ).rstrip("/")
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
                "FinetuningAPI.get_metrics_async: %s %s", request.method, request.url
            )

        try:
            response = await self.send_async(request)
        except httpx.TimeoutException as exc:
            raise TransportTimeout("Request timed-out") from exc
        except httpx.RequestError as exc:
            raise ConnectionFailure("Transport error") from exc

        if self._log:
            self._log.debug(
                "FinetuningAPI.get_metrics_async: %s ← %s",
                response.status_code,
                request.url,
            )

        if not 200 <= response.status_code < 300:
            raise from_http_response(response)

        try:
            raw: dict[str, Any] = response.json()
            return ListMetricsResponse(**raw)
        except Exception as exc:
            raise ServerError(
                "Malformed or unexpected response",
                status_code=response.status_code,
                payload={"raw": response.text},
                request_id=response.headers.get("x-request-id"),
            ) from exc
