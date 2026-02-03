"""Output models for datasets API responses."""

from typing import List, Optional

from pydantic import Field, BaseModel


class Dataset(BaseModel):
    """Dataset object with full details."""

    id: str = Field(..., description="Unique dataset identifier")
    name: str = Field(..., description="Dataset name")
    type: str = Field(..., description="Service type")
    format: str = Field(..., description="Dataset format")
    status: str = Field(..., description="Dataset status")
    description: Optional[str] = Field(default=None, description="Dataset description")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    file_count: Optional[int] = Field(default=None, description="Number of files in dataset")
    total_size: Optional[int] = Field(default=None, description="Total size in bytes")


class ListDatasetsResponse(BaseModel):
    """Response model for listing datasets."""

    datasets: List[Dataset] = Field(..., description="List of datasets")
    total: int = Field(..., description="Total number of datasets matching filters")
    offset: int = Field(..., description="Current pagination offset")
    limit: int = Field(..., description="Current pagination limit")


class DatasetCreationResponse(BaseModel):
    """Response model after creating or uploading to a dataset."""

    id: str = Field(..., description="Dataset identifier")
    name: str = Field(..., description="Dataset name")
    type: str = Field(..., description="Service type")
    format: str = Field(..., description="Dataset format")
    status: str = Field(..., description="Dataset status")
    created_at: str = Field(..., description="Creation timestamp")


class FormatTag(BaseModel):
    """Format information object."""

    name: str = Field(..., description="Format name")
    description: Optional[str] = Field(default=None, description="Format description")


class ListFormatsResponse(BaseModel):
    """Response model for listing available formats."""

    formats: List[FormatTag] = Field(..., description="List of available formats")
    total: int = Field(..., description="Total number of formats")
    offset: int = Field(..., description="Current pagination offset")
    limit: int = Field(..., description="Current pagination limit")
