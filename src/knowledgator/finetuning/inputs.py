"""Input models for fine-tuning API requests."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, conint, constr

from .common import ServiceType, TrainingStatus


class CreateBaseModelRequest(BaseModel):
    """Request model for creating a base model for training."""

    name: constr(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9- .()]+$") = Field(
        ..., description="Model name"
    )
    type: ServiceType = Field(..., description="Service type for the model")
    description: Optional[str] = Field(default=None, description="Model description")


class ListModelsRequest(BaseModel):
    """Request model for listing models for training."""

    offset: conint(ge=0) = Field(default=0, description="Pagination offset")
    limit: conint(ge=0, le=10000) = Field(
        default=100, description="Maximum number of results"
    )
    name: Optional[
        constr(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9- .()]+$")
    ] = Field(default=None, description="Filter by model name")
    type: Optional[ServiceType] = Field(
        default=None, description="Filter by service type"
    )
    include_base: bool = Field(
        default=True,
        description="Include base models for finetuning (not trained by a customer)",
    )


class StartTrainingRequest(BaseModel):
    """Request model for starting a training job."""

    model_id: constr(min_length=2, max_length=64, pattern=r"^[a-zA-Z0-9-]+$") = Field(
        ..., description="Base model ID to train"
    )
    dataset_id: constr(min_length=2, max_length=64, pattern=r"^[a-zA-Z0-9-]+$") = Field(
        ..., description="Dataset ID to use for training"
    )
    hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Training hyperparameters"
    )


class ListTrainingsRequest(BaseModel):
    """Request model for listing training jobs."""

    offset: conint(ge=0) = Field(default=0, description="Pagination offset")
    limit: conint(ge=0, le=10000) = Field(
        default=100, description="Maximum number of results"
    )
    status: Optional[TrainingStatus] = Field(
        default=None, description="Filter by training status"
    )
    type: Optional[ServiceType] = Field(
        default=None, description="Filter by service type"
    )


class ListEventsRequest(BaseModel):
    """Request model for listing training events."""

    offset: conint(ge=0) = Field(default=0, description="Pagination offset")
    limit: conint(ge=0, le=10000) = Field(
        default=100, description="Maximum number of results"
    )


class GetMetricsRequest(BaseModel):
    """Request model for getting training metrics."""

    metric: str = Field(
        "loss", description="Metric name (currently only 'loss' is supported)"
    )
    offset: conint(ge=0) = Field(default=0, description="Pagination offset")
    limit: conint(ge=0, le=10000) = Field(
        default=100, description="Maximum number of results"
    )
