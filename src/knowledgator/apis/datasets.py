"""Datasets API for managing datasets."""
from __future__ import annotations

from typing import Optional, BinaryIO
import httpx

from .base import BaseAPI
from ..models.inputs import DatasetCreateRequest, DatasetUpdateRequest
from ..models.outputs import DatasetCreateObject, DatasetObject, DatasetsListResponse

# Error handling - using consolidated errors module
from ..errors import (
    raise_for_status,
    InternalServerError,
)


class DatasetsAPI(BaseAPI):
    """API wrapper for datasets endpoints."""

    def list_datasets(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        type: Optional[str] = None,
        format: Optional[str] = None,
        status: Optional[str] = None,
    ) -> DatasetsListResponse:
        """
        List all datasets (sync).

        Args:
            page: Optional page number for pagination
            page_size: Optional page size for pagination

        Returns:
            DatasetsListResponse with list of datasets

        Raises:
            SDKError: If the request fails
        """
        params = {}

        # Support both legacy page/page_size and newer offset/limit style.
        # Priority: explicit offset/limit -> compute from page/page_size -> defaults
        # Determine limit
        if limit is None:
            if page_size is not None:
                limit = page_size
            else:
                limit = 100

        # Determine offset
        if offset is None:
            if page is not None:
                # map page+limit to offset (page is 1-based)
                offset = max((page - 1) * limit, 0)
            else:
                offset = 0

        # Populate params using the API's expected query names
        params["offset"] = offset
        params["limit"] = limit
        if type is not None:
            params["type"] = type
        if format is not None:
            params["format"] = format
        if status is not None:
            params["status"] = status

        # Make request
        request = self._make_request(
            method="GET",
            path="/datasets/",
            query_params=params if params else None,
        )
        response = self.send(request)
        # Parse response
        return self._parse_datasets_list_response(response)

    async def list_datasets_async(
        self,
        *,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        type: Optional[str] = None,
        format: Optional[str] = None,
        status: Optional[str] = None,
    ) -> DatasetsListResponse:
        """
        List all datasets (async).

        Args:
            page: Optional page number for pagination
            page_size: Optional page size for pagination

        Returns:
            DatasetsListResponse with list of datasets

        Raises:
            SDKError: If the request fails
        """
        params = {}

        if limit is None:
            if page_size is not None:
                limit = page_size
            else:
                limit = 100

        if offset is None:
            if page is not None:
                offset = max((page - 1) * limit, 0)
            else:
                offset = 0

        params["offset"] = offset
        params["limit"] = limit
        if type is not None:
            params["type"] = type
        if format is not None:
            params["format"] = format
        if status is not None:
            params["status"] = status

        # Make request
        request = self._make_request(
            method="GET",
            path="/datasets/",
            query_params=params if params else None,
        )
        response = await self.send_async(request)

        # Parse response
        return self._parse_datasets_list_response(response)

    def create_dataset(
        self,
        *,
        name: str,
        type: str,
        description: Optional[str] = None,
        # Optional file upload (multipart). If provided, request will be sent
        # as multipart/form-data with the file and the other fields as form parts.
        file: Optional[BinaryIO] = None,
        filename: Optional[str] = None,
        format: Optional[str] = None,
    ) -> DatasetObject:
        """
        Create a new dataset (sync).

        Args:
            name: Dataset name
            type: Dataset type (e.g., 'comprehend-it', 'extract-it')
            description: Optional dataset description

        Returns:
            DatasetObject with dataset details

        Raises:
            SDKError: If the request fails
        """
        # Build payload
        payload = {
            "name": name,
            "type": type,
        }
        if description is not None:
            payload["short_description"] = description
        if format is not None:
            payload["format"] = format

        # If a file is supplied, send multipart/form-data with both file and
        # form fields. Otherwise fall back to JSON body for backward-compat.
        if file is not None:
            fn = filename or getattr(file, "name", "upload")
            files = {"file": (fn, file)}
            request = self._make_request(
                method="POST",
                path="/datasets/",
                files=files,
                data_body=payload,
            )
        else:
            # Keep previous JSON behaviour
            request_payload = DatasetCreateRequest(
                name=name,
                type=type,
                description=description,
            )
            request = self._make_request(
                method="POST",
                path="/datasets/",
                json_body=request_payload.model_dump(exclude_none=True),
            )

        response = self.send(request)

        # Parse response
        return self._parse_dataset_create_response(response)

    async def create_dataset_async(
        self,
        *,
        name: str,
        type: str,
        description: Optional[str] = None,
        file: Optional[BinaryIO] = None,
        filename: Optional[str] = None,
        format: Optional[str] = None,
    ) -> DatasetObject:
        """
        Create a new dataset (async).

        Args:
            name: Dataset name
            type: Dataset type (e.g., 'comprehend-it', 'extract-it')
            description: Optional dataset description

        Returns:
            DatasetObject with dataset details

        Raises:
            SDKError: If the request fails
        """
        payload = {
            "name": name,
            "type": type,
        }
        if description is not None:
            payload["short_description"] = description
        if format is not None:
            payload["format"] = format

        if file is not None:
            fn = filename or getattr(file, "name", "upload")
            files = {"file": (fn, file)}
            request = self._make_request(
                method="POST",
                path="/datasets/",
                files=files,
                data_body=payload,
            )
        else:
            request_payload = DatasetCreateRequest(
                name=name,
                type=type,
                description=description,
            )
            request = self._make_request(
                method="POST",
                path="/datasets/",
                json_body=request_payload.model_dump(exclude_none=True),
            )

        response = await self.send_async(request)

        # Parse response
        return self._parse_dataset_create_response(response)

    def get_dataset(
        self,
        *,
        dataset_id: str,
    ) -> DatasetObject:
        """
        Get a specific dataset by ID (sync).

        Args:
            dataset_id: The dataset ID

        Returns:
            DatasetObject with dataset details

        Raises:
            SDKError: If the request fails
        """
        # Make request
        request = self._make_request(
            method="GET",
            path=f"/datasets/{dataset_id}/",
        )
        response = self.send(request)
        # Parse response
        return self._parse_dataset_response(response)

    async def get_dataset_async(
        self,
        *,
        dataset_id: str,
    ) -> DatasetObject:
        """
        Get a specific dataset by ID (async).

        Args:
            dataset_id: The dataset ID

        Returns:
            DatasetObject with dataset details

        Raises:
            SDKError: If the request fails
        """
        # Make request
        request = self._make_request(
            method="GET",
            path=f"/datasets/{dataset_id}/",
        )
        response = await self.send_async(request)

        # Parse response
        return self._parse_dataset_response(response)

    def update_dataset(
        self,
        *,
        dataset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> DatasetObject:
        """
        Update a dataset (sync).

        Args:
            dataset_id: The dataset ID
            name: Optional new name
            description: Optional new description

        Returns:
            DatasetObject with updated dataset details

        Raises:
            SDKError: If the request fails
        """
        request_payload = DatasetUpdateRequest(
            name=name,
            description=description,
        )

        # Make request
        request = self._make_request(
            method="PATCH",
            path=f"/datasets/{dataset_id}/",
            json_body=request_payload.model_dump(exclude_none=True),
        )
        response = self.send(request)

        # Parse response
        return self._parse_dataset_response(response)

    async def update_dataset_async(
        self,
        *,
        dataset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> DatasetObject:
        """
        Update a dataset (async).

        Args:
            dataset_id: The dataset ID
            name: Optional new name
            description: Optional new description

        Returns:
            DatasetObject with updated dataset details

        Raises:
            SDKError: If the request fails
        """
        request_payload = DatasetUpdateRequest(
            name=name,
            description=description,
        )

        # Make request
        request = self._make_request(
            method="PATCH",
            path=f"/datasets/{dataset_id}/",
            json_body=request_payload.model_dump(exclude_none=True),
        )
        response = await self.send_async(request)

        # Parse response
        return self._parse_dataset_response(response)

    def delete_dataset(
        self,
        *,
        dataset_id: str,
    ) -> dict:
        """
        Delete a dataset (sync).

        Args:
            dataset_id: The dataset ID

        Returns:
            Dictionary with deletion confirmation

        Raises:
            SDKError: If the request fails
        """
        # Make request
        request = self._make_request(
            method="DELETE",
            path=f"/datasets/{dataset_id}/",
        )
        response = self.send(request)

        # Parse response (DELETE typically returns 204 No Content or a simple confirmation)
        if not response.is_success:
            raise_for_status(response)

        # For 204 No Content, return empty dict
        if response.status_code == 204:
            return {"status": "deleted", "dataset_id": dataset_id}

        try:
            return response.json()
        except ValueError:
            return {"status": "deleted", "dataset_id": dataset_id}

    async def delete_dataset_async(
        self,
        *,
        dataset_id: str,
    ) -> dict:
        """
        Delete a dataset (async).

        Args:
            dataset_id: The dataset ID

        Returns:
            Dictionary with deletion confirmation

        Raises:
            SDKError: If the request fails
        """
        # Make request
        request = self._make_request(
            method="DELETE",
            path=f"/datasets/{dataset_id}/",
        )
        response = await self.send_async(request)

        # Parse response
        if not response.is_success:
            raise_for_status(response)

        # For 204 No Content, return empty dict
        if response.status_code == 204:
            return {"status": "deleted", "dataset_id": dataset_id}

        try:
            return response.json()
        except ValueError:
            return {"status": "deleted", "dataset_id": dataset_id}

    def upload_dataset_file(
        self,
        *,
        dataset_id: str,
        file: BinaryIO,
        filename: str,
    ) -> dict:
        """
        Upload a file to a dataset (sync).

        Args:
            dataset_id: The dataset ID
            file: File object to upload
            filename: Name of the file

        Returns:
            Dictionary with upload confirmation

        Raises:
            SDKError: If the request fails
        """
        # Make multipart/form-data request
        files = {"file": (filename, file)}

        request = self._make_request(
            method="POST",
            path=f"/datasets/{dataset_id}/files/upload/",
            files=files,
        )
        response = self.send(request)

        # Parse response
        if not response.is_success:
            raise_for_status(response)

        try:
            return response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={
                    "message": "Response is not valid JSON",
                    "raw_text": response.text,
                },
                headers=dict(response.headers),
            ) from exc

    async def upload_dataset_file_async(
        self,
        *,
        dataset_id: str,
        file: BinaryIO,
        filename: str,
    ) -> dict:
        """
        Upload a file to a dataset (async).

        Args:
            dataset_id: The dataset ID
            file: File object to upload
            filename: Name of the file

        Returns:
            Dictionary with upload confirmation

        Raises:
            SDKError: If the request fails
        """
        # Make multipart/form-data request
        files = {"file": (filename, file)}

        request = self._make_request(
            method="POST",
            path=f"/datasets/{dataset_id}/files/upload/",
            files=files,
        )
        response = await self.send_async(request)

        # Parse response
        if not response.is_success:
            raise_for_status(response)

        try:
            return response.json()
        except ValueError as exc:
            raise InternalServerError(
                body={
                    "message": "Response is not valid JSON",
                    "raw_text": response.text,
                },
                headers=dict(response.headers),
            ) from exc

    def _parse_datasets_list_response(self, response: httpx.Response) -> DatasetsListResponse:
        """Parse HTTP response into DatasetsListResponse.
        
        Args:
            response: HTTP response from datasets list endpoint
            
        Returns:
            DatasetsListResponse with list of datasets
            
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
            return DatasetsListResponse(**data)
        except Exception as exc:
            raise InternalServerError(
                body={
                    "message": "Failed to parse datasets list response",
                    "error": str(exc),
                    "raw_data": data,
                },
                headers=dict(response.headers),
            ) from exc

    def _parse_dataset_response(self, response: httpx.Response) -> DatasetObject:
        """Parse HTTP response into DatasetObject.
        
        Args:
            response: HTTP response from dataset endpoint
            
        Returns:
            DatasetObject with dataset details
            
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
            return DatasetObject(**data)
        except Exception as exc:
            raise InternalServerError(
                body={
                    "message": "Failed to parse dataset response",
                    "error": str(exc),
                    "raw_data": data,
                },
                headers=dict(response.headers),
            ) from exc

    def _parse_dataset_create_response(self, response: httpx.Response) -> DatasetCreateObject:
        """Parse HTTP response into DatasetCreateObject.

        Args:
            response: HTTP response from dataset endpoint
            
        Returns:
            DatasetCreateObject with dataset details
            
        Raises:
            BadRequestError: For 400 errors (invalid input)
            UnauthorizedError: For 401 errors (auth issues)
            InternalServerError: For 500 errors or parsing failures
        """

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
            return DatasetCreateObject(**data)
        except Exception as exc:
            raise InternalServerError(
                body={
                    "message": "Failed to parse dataset response",
                    "error": str(exc),
                    "raw_data": data,
                },
                headers=dict(response.headers),
            ) from exc
