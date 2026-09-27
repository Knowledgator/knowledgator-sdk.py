"""Output models for datasets API responses."""

from typing import Any, List, Union, Optional

from pydantic import Field, BaseModel, AliasChoices


class Dataset(BaseModel):
    """Dataset object with full details."""

    id: str = Field(..., description="Unique dataset identifier")
    name: str = Field(..., description="Dataset name")
    type: str = Field(..., description="Service type")
    format: str = Field(..., description="Dataset format")
    status: str = Field(..., description="Dataset status")
    description: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("description", "short_description"),
        description="Dataset description",
    )
    short_description: Optional[str] = Field(default=None, description="Short dataset description")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    file_count: Optional[int] = Field(default=None, description="Number of files in dataset")
    total_size: Optional[int] = Field(default=None, description="Total size in bytes")
    created_by: Optional[str] = Field(default=None, description="Creator user ID")
    examples_count: Optional[int] = Field(default=None, description="Number of examples")
    source: Optional[str] = Field(default=None, description="Dataset source")
    files: Optional[List[Any]] = Field(default=None, description="Dataset files")
    generation_status: Optional[Any] = Field(default=None, description="Generation status")


class ListDatasetsResponse(BaseModel):
    """Response model for listing datasets."""

    datasets: List[Dataset] = Field(
        default_factory=list,
        validation_alias=AliasChoices("datasets", "data"),
        description="List of datasets",
    )
    total: int = Field(0, description="Total number of datasets matching filters")
    offset: Optional[int] = Field(default=0, description="Current pagination offset")
    limit: Optional[int] = Field(default=100, description="Current pagination limit")


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

    formats: List[Union[str, FormatTag]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("formats", "data"),
        description="List of available formats",
    )
    total: int = Field(0, description="Total number of formats")
    offset: Optional[int] = Field(default=0, description="Current pagination offset")
    limit: Optional[int] = Field(default=100, description="Current pagination limit")


class Data(BaseModel):
    """Raw data object with full details."""

    id: str = Field(..., description="Unique data identifier")
    name: str = Field(..., description="Data name")
    type: str = Field(..., description="Service type")
    format: str = Field(..., description="Data format")
    # status: str = Field(..., description="Data status") # Raw data might not have status like datasets
    description: Optional[str] = Field(default=None, description="Data description")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    file_count: Optional[int] = Field(default=None, description="Number of files")
    total_size: Optional[int] = Field(default=None, description="Total size in bytes")


class ListDataResponse(BaseModel):
    """Response model for listing raw data."""

    data: List[Data] = Field(..., description="List of raw data")
    total: int = Field(..., description="Total number of data items matching filters")
    offset: int = Field(..., description="Current pagination offset")
    limit: int = Field(..., description="Current pagination limit")
