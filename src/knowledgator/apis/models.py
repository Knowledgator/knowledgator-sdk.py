"""Models API for managing base models for fine-tuning."""
from __future__ import annotations

from typing import Optional
import httpx

from .base import BaseAPI
from ..models.inputs import ModelCreateRequest
from ..models.outputs import (
    ModelObject,
    ModelsListResponse,
    ModelTypesListResponse,
)

# Error handling - using consolidated errors module
from ..errors import (
    raise_for_status,
    InternalServerError,
    BadRequestError,
)


class ModelsAPI(BaseAPI):
    """API wrapper for fine-tuning models management endpoints."""

    def create_model(
        self,
        *,
        name: str,
        type: str,
        description: Optional[str] = None,
    ) -> ModelObject:
        """
        Create a base model for training (sync).

        Args:
            name: Model name (3-64 chars, alphanumeric with spaces, dots, parens, hyphens)
            type: Model type (e.g., 'comprehend-it', 'extract-it')
            description: Optional model description

        Returns:
            ModelObject with created model details

        Raises:
            BadRequestError: If input validation fails
            UnauthorizedError: If API key is invalid
            InternalServerError: If the server encounters an error
        """
        # Build request
        request_payload = ModelCreateRequest(
            name=name,
            type=type,
            description=description,
        )

        # Make request
        request = self._make_request(
            method="POST",
            path="/trainings/models/",
            json_body=request_payload.model_dump(exclude_none=True),
        )
        response = self.send(request)

        # Parse response
        return self._parse_model_response(response)

    async def create_model_async(
        self,
        *,
        name: str,
        type: str,
        description: Optional[str] = None,
    ) -> ModelObject:
        """
        Create a base model for training (async).

        Args:
            name: Model name (3-64 chars, alphanumeric with spaces, dots, parens, hyphens)
            type: Model type (e.g., 'comprehend-it', 'extract-it')
            description: Optional model description

        Returns:
            ModelObject with created model details

        Raises:
            BadRequestError: If input validation fails
            UnauthorizedError: If API key is invalid
            InternalServerError: If the server encounters an error
        """
        # Build request
        request_payload = ModelCreateRequest(
            name=name,
            type=type,
            description=description,
        )

        # Make request
        request = self._make_request(
            method="POST",
            path="/trainings/models/",
            json_body=request_payload.model_dump(exclude_none=True),
        )
        response = await self.send_async(request)

        # Parse response
        return self._parse_model_response(response)

    def list_models(self) -> ModelsListResponse:
        """
        List all available base models for training (sync).

        Returns:
            ModelsListResponse with list of models and total count

        Raises:
            UnauthorizedError: If API key is invalid
            InternalServerError: If the server encounters an error
        """
        # Make request
        request = self._make_request(
            method="GET",
            path="/trainings/models/",
        )
        response = self.send(request)

        # Parse response
        return self._parse_models_list_response(response)

    async def list_models_async(self) -> ModelsListResponse:
        """
        List all available base models for training (async).

        Returns:
            ModelsListResponse with list of models and total count

        Raises:
            UnauthorizedError: If API key is invalid
            InternalServerError: If the server encounters an error
        """
        # Make request
        request = self._make_request(
            method="GET",
            path="/trainings/models/",
        )
        response = await self.send_async(request)

        # Parse response
        return self._parse_models_list_response(response)

    def list_model_types(self) -> ModelTypesListResponse:
        """
        List all available model types for training (sync).

        Returns:
            ModelTypesListResponse with list of model types and total count

        Raises:
            UnauthorizedError: If API key is invalid
            InternalServerError: If the server encounters an error
        """
        # Make request
        request = self._make_request(
            method="GET",
            path="/trainings/models/types/",
        )
        response = self.send(request)

        # Parse response
        return self._parse_model_types_response(response)

    async def list_model_types_async(self) -> ModelTypesListResponse:
        """
        List all available model types for training (async).

        Returns:
            ModelTypesListResponse with list of model types and total count

        Raises:
            UnauthorizedError: If API key is invalid
            InternalServerError: If the server encounters an error
        """
        # Make request
        request = self._make_request(
            method="GET",
            path="/trainings/models/types/",
        )
        response = await self.send_async(request)
        # Parse response
        return self._parse_model_types_response(response)

    def get_model(self, model_id: str) -> ModelObject:
        """
        Get a specific model by ID (sync).

        Args:
            model_id: The ID of the model to retrieve

        Returns:
            ModelObject with model details

        Raises:
            BadRequestError: If model_id is invalid
            NotFoundError: If model doesn't exist
            UnauthorizedError: If API key is invalid
            InternalServerError: If the server encounters an error
        """
        # Make request

        request = self._make_request(
            method="GET",
            path=f"/trainings/models/{model_id}",
        )
        response = self.send(request)

        # Parse response
        return self._parse_model_response(response)

    async def get_model_async(self, model_id: str) -> ModelObject:
        """
        Get a specific model by ID (async).

        Args:
            model_id: The ID of the model to retrieve

        Returns:
            ModelObject with model details

        Raises:
            BadRequestError: If model_id is invalid
            NotFoundError: If model doesn't exist
            UnauthorizedError: If API key is invalid
            InternalServerError: If the server encounters an error
        """
        # Make request
        request = self._make_request(
            method="GET",
            path=f"/trainings/models/{model_id}",
        )
        response = await self.send_async(request)

        # Parse response
        return self._parse_model_response(response)

    # --- Response Parsing Methods ---

    def _parse_model_response(self, response: httpx.Response) -> ModelObject:
        """Parse HTTP response into ModelObject.

        Args:
            response: HTTP response from models endpoint

        Returns:
            ModelObject with model details

        Raises:
            BadRequestError: For 400 errors (invalid input)
            UnauthorizedError: For 401 errors (auth issues)
            NotFoundError: For 404 errors (model not found)
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
            return ModelObject(**data)
        except Exception as exc:
            raise InternalServerError(
                body={
                    "message": "Failed to parse model response",
                    "error": str(exc),
                    "raw_data": data,
                },
                headers=dict(response.headers),
            ) from exc

    def _parse_models_list_response(
        self, response: httpx.Response
    ) -> ModelsListResponse:
        """Parse HTTP response into ModelsListResponse.

        Args:
            response: HTTP response from list models endpoint

        Returns:
            ModelsListResponse with models list

        Raises:
            UnauthorizedError: For 401 errors (auth issues)
            InternalServerError: For 500 errors or parsing failures
        """
        # Use new error hierarchy for automatic status-specific error raising
        if not response.is_success:
            raise_for_status(response)

        # Parse JSON first, isolating JSON errors from validation errors
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
            return ModelsListResponse(**data)
        except Exception as exc:
            raise InternalServerError(
                body={
                    "message": "Failed to parse models list response",
                    "error": str(exc),
                    "raw_data": data,
                },
                headers=dict(response.headers),
            ) from exc

    def _parse_model_types_response(
        self, response: httpx.Response
    ) -> ModelTypesListResponse:
        """Parse HTTP response into ModelTypesListResponse (expects 'data' as list of strings)."""
        if not response.is_success:
            raise_for_status(response)

        try:
            data = response.json()
            # Expecting: {"total": int, "data": [str, ...]}
            # if "data" in data and isinstance(data["data"], list):
            #     return ModelTypesListResponse(data=data["data"], total=data.get("total", len(data["data"])))
            # # Fallback: try to coerce any list field
            # for key, value in data.items():
            #     if isinstance(value, list) and all(isinstance(x, str) for x in value):
            #         return ModelTypesListResponse(data=value, total=data.get("total", len(value)))
            # If the response is already in the expected format
            return ModelTypesListResponse(**data)
        except ValueError as exc:
            raise InternalServerError(
                body={
                    "message": "Response is not valid JSON",
                    "raw_text": response.text,
                },
                headers=dict(response.headers),
            ) from exc
        except Exception as exc:
            raise InternalServerError(
                body={
                    "message": "Failed to parse model types response",
                    "error": str(exc),
                    "raw_data": response.text,
                },
                headers=dict(response.headers),
            ) from exc
