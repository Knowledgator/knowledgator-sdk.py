"""Training API for fine-tuning models."""
from __future__ import annotations

from typing import Optional
import httpx

from .base import BaseAPI
from ..models.inputs import TrainingCreateRequest
from ..models.outputs import TrainingObject

# Error handling - using consolidated errors module
from ..errors import (
    raise_for_status,
    InternalServerError,
    BadRequestError,
    SDKError,
)

from ..models.inputs import TrainingArgs, TrainingMetadata, ValidationMetadata
from ..models.outputs import (
    TrainingObject,
    TrainingsListResponse,
    TrainingEventsListResponse,
    TrainingMetricResponse,
    EvaluationsListResponse,
)


class TrainingAPI(BaseAPI):
    """API wrapper for training/fine-tuning endpoints."""

    def start_training(
        self,
        *,
        parent_model_id: str,
        dataset_id: str,
        seed: int,
        batch: Optional[int] = None,
        learning_rate_mult: Optional[int] = None,
        num_of_epochs: Optional[int] = None,
        name: Optional[str] = None,
        model_name: Optional[str] = None,
        validation_dataset_id: Optional[str] = None,
    ) -> TrainingObject:
        """
        Start a fine-tuning training job (sync).

        Args:
            parent_model_id: Base model ID to fine-tune
            dataset_id: Training dataset ID
            seed: Random seed for training
            batch: Optional batch size
            learning_rate_mult: Optional learning rate multiplier
            num_of_epochs: Optional number of training epochs
            name: Optional training job name
            model_name: Optional name for the fine-tuned model
            validation_dataset_id: Optional validation dataset ID

        Returns:
            TrainingObject with job details and status

        Raises:
            SDKError: If the request fails
        """
        from ..models.inputs import TrainingArgs, TrainingMetadata, ValidationMetadata

        # Build request
        training_args = TrainingArgs(
            seed=seed,
            batch=batch,
            learning_rate_mult=learning_rate_mult,
            num_of_epochs=num_of_epochs,
        )
        training_metadata = TrainingMetadata(dataset_id=dataset_id, args=training_args)
        validation_metadata = (
            ValidationMetadata(dataset_id=validation_dataset_id)
            if validation_dataset_id
            else None
        )

        request_payload = TrainingCreateRequest(
            name=name,
            model_name=model_name,
            parent_model_id=parent_model_id,
            training=training_metadata,
            validation=validation_metadata,
        )

        # Make request
        request = self._make_request(
            method="POST",
            path="/trainings/",
            json_body=request_payload.model_dump(exclude_none=True),
        )
        response = self.send(request)

        # Parse response
        return self._parse_response(response)

    async def start_training_async(
        self,
        *,
        parent_model_id: str,
        dataset_id: str,
        seed: int,
        batch: Optional[int] = None,
        learning_rate_mult: Optional[int] = None,
        num_of_epochs: Optional[int] = None,
        name: Optional[str] = None,
        model_name: Optional[str] = None,
        validation_dataset_id: Optional[str] = None,
    ) -> TrainingObject:
        """
        Start a fine-tuning training job (async).

        Args:
            parent_model_id: Base model ID to fine-tune
            dataset_id: Training dataset ID
            seed: Random seed for training
            batch: Optional batch size
            learning_rate_mult: Optional learning rate multiplier
            num_of_epochs: Optional number of training epochs
            name: Optional training job name
            model_name: Optional name for the fine-tuned model
            validation_dataset_id: Optional validation dataset ID

        Returns:
            TrainingObject with job details and status

        Raises:
            SDKError: If the request fails
        """

        # Build request
        training_args = TrainingArgs(
            seed=seed,
            batch=batch,
            learning_rate_mult=learning_rate_mult,
            num_of_epochs=num_of_epochs,
        )
        training_metadata = TrainingMetadata(dataset_id=dataset_id, args=training_args)
        validation_metadata = (
            ValidationMetadata(dataset_id=validation_dataset_id)
            if validation_dataset_id
            else None
        )

        request_payload = TrainingCreateRequest(
            name=name,
            model_name=model_name,
            parent_model_id=parent_model_id,
            training=training_metadata,
            validation=validation_metadata,
        )

        # Make request
        request = self._make_request(
            method="POST",
            path="/trainings/",
            json_body=request_payload.model_dump(exclude_none=True),
        )
        response = await self.send_async(request)

        # Parse response
        return self._parse_response(response)

    def _parse_response(self, response: httpx.Response) -> TrainingObject:
        """Parse HTTP response into TrainingObject.
        
        Args:
            response: HTTP response from training endpoint
            
        Returns:
            TrainingObject with job details
            
        Raises:
            BadRequestError: For 400 errors (invalid input)
            UnauthorizedError: For 401 errors (auth issues)
            InternalServerError: For 500 errors or parsing failures
        """
        # Use new error hierarchy for automatic status-specific error raising
        if not response.is_success:
            raise_for_status(response)

        # Parse JSON first, isolate JSON errors
        try:
            data = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={
                    "message": "Response is not valid JSON",
                    "raw_text": response.text,
                },
                headers=dict(response.headers),
            ) from exc

        # Validate/shape into our pydantic model
        try:
            return TrainingObject(**data)
        except Exception as exc:
            raise InternalServerError(
                body={
                    "message": "Failed to parse training response",
                    "error": str(exc),
                    "raw_data": data,
                },
                headers=dict(response.headers),
            ) from exc

    def list_trainings(self, *, offset: int = 0, limit: int = 100) -> TrainingsListResponse:
        """List training jobs (sync)."""
        request = self._make_request(
            method="GET",
            path="/trainings/",
            params={"offset": offset, "limit": limit},
        )
        response = self.send(request)

        if not response.is_success:
            raise_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc

        try:
            return TrainingsListResponse(**data)
        except Exception as exc:
            raise InternalServerError(
                body={"message": "Failed to parse trainings list", "error": str(exc), "raw_data": data},
                headers=dict(response.headers),
            ) from exc

    async def list_trainings_async(self, *, offset: int = 0, limit: int = 100) -> TrainingsListResponse:
        request = self._make_request(
            method="GET",
            path="/trainings/",
            params={"offset": offset, "limit": limit},
        )
        response = await self.send_async(request)

        if not response.is_success:
            raise_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc

        try:
            return TrainingsListResponse(**data)
        except Exception as exc:
            raise InternalServerError(
                body={"message": "Failed to parse trainings list", "error": str(exc), "raw_data": data},
                headers=dict(response.headers),
            ) from exc

    def get_training(self, training_id: str) -> TrainingObject:
        request = self._make_request(method="GET", path=f"/trainings/{training_id}/")
        response = self.send(request)
        if not response.is_success:
            raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc
        try:
            return TrainingObject(**data)
        except Exception as exc:
            raise InternalServerError(
                body={"message": "Failed to parse training object", "error": str(exc), "raw_data": data},
                headers=dict(response.headers),
            ) from exc

    async def get_training_async(self, training_id: str) -> TrainingObject:
        request = self._make_request(method="GET", path=f"/trainings/{training_id}/")
        response = await self.send_async(request)
        if not response.is_success:
            raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc
        try:
            return TrainingObject(**data)
        except Exception as exc:
            raise InternalServerError(
                body={"message": "Failed to parse training object", "error": str(exc), "raw_data": data},
                headers=dict(response.headers),
            ) from exc

    def delete_training(self, training_id: str) -> None:
        request = self._make_request(method="DELETE", path=f"/trainings/{training_id}/")
        response = self.send(request)
        if not response.is_success:
            raise_for_status(response)
        return None

    async def delete_training_async(self, training_id: str) -> None:
        request = self._make_request(method="DELETE", path=f"/trainings/{training_id}/")
        response = await self.send_async(request)
        if not response.is_success:
            raise_for_status(response)
        return None

    def stop_training(self, training_id: str) -> TrainingObject:
        request = self._make_request(method="POST", path=f"/trainings/{training_id}/stop/", json_body={})
        response = self.send(request)
        if not response.is_success:
            raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc
        try:
            return TrainingObject(**data)
        except Exception as exc:
            raise InternalServerError(
                body={"message": "Failed to parse training object", "error": str(exc), "raw_data": data},
                headers=dict(response.headers),
            ) from exc

    async def stop_training_async(self, training_id: str) -> TrainingObject:
        request = self._make_request(method="POST", path=f"/trainings/{training_id}/stop/", json_body={})
        response = await self.send_async(request)
        if not response.is_success:
            raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc
        try:
            return TrainingObject(**data)
        except Exception as exc:
            raise InternalServerError(
                body={"message": "Failed to parse training object", "error": str(exc), "raw_data": data},
                headers=dict(response.headers),
            ) from exc

    def list_events(self, training_id: str, *, offset: int = 0, limit: int = 100) -> TrainingEventsListResponse:
        request = self._make_request(method="GET", path=f"/trainings/{training_id}/events/", params={"offset": offset, "limit": limit})
        response = self.send(request)
        if not response.is_success:
            raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc
        try:
            return TrainingEventsListResponse(**data)
        except Exception as exc:
            raise InternalServerError(
                body={"message": "Failed to parse training events", "error": str(exc), "raw_data": data},
                headers=dict(response.headers),
            ) from exc

    async def list_events_async(self, training_id: str, *, offset: int = 0, limit: int = 100) -> TrainingEventsListResponse:
        request = self._make_request(method="GET", path=f"/trainings/{training_id}/events/", params={"offset": offset, "limit": limit})
        response = await self.send_async(request)
        if not response.is_success:
            raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc
        try:
            return TrainingEventsListResponse(**data)
        except Exception as exc:
            raise InternalServerError(
                body={"message": "Failed to parse training events", "error": str(exc), "raw_data": data},
                headers=dict(response.headers),
            ) from exc

    def get_metric(self, training_id: str, metric: str) -> TrainingMetricResponse:
        request = self._make_request(method="GET", path=f"/trainings/{training_id}/metrics/{metric}/")
        response = self.send(request)
        if not response.is_success:
            raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc
        try:
            return TrainingMetricResponse(**data)
        except Exception as exc:
            raise InternalServerError(
                body={"message": "Failed to parse training metric", "error": str(exc), "raw_data": data},
                headers=dict(response.headers),
            ) from exc

    async def get_metric_async(self, training_id: str, metric: str) -> TrainingMetricResponse:
        request = self._make_request(method="GET", path=f"/trainings/{training_id}/metrics/{metric}/")
        response = await self.send_async(request)
        if not response.is_success:
            raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc
        try:
            return TrainingMetricResponse(**data)
        except Exception as exc:
            raise InternalServerError(
                body={"message": "Failed to parse training metric", "error": str(exc), "raw_data": data},
                headers=dict(response.headers),
            ) from exc

    def list_evaluations(self, training_id: str, *, offset: int = 0, limit: int = 100) -> EvaluationsListResponse:
        request = self._make_request(method="GET", path=f"/trainings/{training_id}/evaluations/", params={"offset": offset, "limit": limit})
        response = self.send(request)
        if not response.is_success:
            raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc
        try:
            return EvaluationsListResponse(**data)
        except Exception as exc:
            raise InternalServerError(
                body={"message": "Failed to parse evaluations", "error": str(exc), "raw_data": data},
                headers=dict(response.headers),
            ) from exc

    async def list_evaluations_async(self, training_id: str, *, offset: int = 0, limit: int = 100) -> EvaluationsListResponse:
        request = self._make_request(method="GET", path=f"/trainings/{training_id}/evaluations/", params={"offset": offset, "limit": limit})
        response = await self.send_async(request)
        if not response.is_success:
            raise_for_status(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={"message": "Response is not valid JSON", "raw_text": response.text},
                headers=dict(response.headers),
            ) from exc
        try:
            return EvaluationsListResponse(**data)
        except Exception as exc:
            raise InternalServerError(
                body={"message": "Failed to parse evaluations", "error": str(exc), "raw_data": data},
                headers=dict(response.headers),
            ) from exc

        
