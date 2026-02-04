"""Input models for datasets API requests."""

from typing import Optional

from pydantic import Field, BaseModel, conint, constr

from .common import ServiceType, DatasetFormat, DatasetStatus


class ListDatasetsRequest(BaseModel):
    """Request model for listing datasets."""

    offset: conint(ge=0) = Field(default=0, description="Pagination offset")
    limit: conint(ge=0, le=10000) = Field(default=100, description="Maximum number of results")
    name: Optional[constr(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9- .()]+$")] = Field(
        default=None, description="Filter by dataset name"
    )
    type: Optional[ServiceType] = Field(default=None, description="Filter by service type")
    format: Optional[DatasetFormat] = Field(default=None, description="Filter by dataset format")
    status: Optional[DatasetStatus] = Field(default=None, description="Filter by dataset status")


class CreateDatasetRequest(BaseModel):
    """Request model for creating a dataset."""

    name: constr(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9- .()]+$") = Field(..., description="Dataset name")
    type: ServiceType = Field(..., description="Service type for the dataset")
    format: DatasetFormat = Field(..., description="Dataset file format")
    description: Optional[str] = Field(default=None, description="Dataset description")
    file: Optional[bytes] = Field(default=None, description="Optional file content to upload", exclude=True)


class UpdateDatasetRequest(BaseModel):
    """Request model for updating dataset metadata."""

    name: Optional[constr(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9- .()]+$")] = Field(
        default=None, description="New dataset name"
    )
    description: Optional[str] = Field(default=None, description="New dataset description")


class UploadDatasetRequest(BaseModel):
    """Request model for uploading files to a dataset."""

    file: bytes = Field(..., description="File content to upload", exclude=True)


class ListFormatsRequest(BaseModel):
    """Request model for listing available dataset formats."""

    offset: conint(ge=0) = Field(default=0, description="Pagination offset")
    limit: conint(ge=0, le=10000) = Field(default=100, description="Maximum number of results")
