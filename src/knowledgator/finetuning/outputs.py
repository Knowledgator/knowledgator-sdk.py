"""Output models for fine-tuning API responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainingModel(BaseModel):
    """Training model object with full details."""

    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Service type")
    description: Optional[str] = Field(default=None, description="Model description")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    is_base: Optional[bool] = Field(
        default=None,
        description="Whether this is a base model (not trained by customer)",
    )


class ListModelsResponse(BaseModel):
    """Response model for listing models."""

    models: List[TrainingModel] = Field(..., description="List of models")
    total: int = Field(..., description="Total number of models matching filters")
    offset: int = Field(..., description="Current pagination offset")
    limit: int = Field(..., description="Current pagination limit")


class ObjectCreation(BaseModel):
    """Response model after creating an object."""

    id: str = Field(..., description="Object identifier")
    created_at: str = Field(..., description="Creation timestamp")


class TrainingObject(BaseModel):
    """Training job object with full details."""

    id: str = Field(..., description="Unique training identifier")
    model_id: str = Field(..., description="Base model ID")
    dataset_id: str = Field(..., description="Dataset ID")
    status: str = Field(..., description="Training status")
    hyperparameters: Optional[Dict[str, Any]] = Field(default=None, description="Training hyperparameters")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    started_at: Optional[str] = Field(default=None, description="Training start timestamp")
    completed_at: Optional[str] = Field(default=None, description="Training completion timestamp")
    error: Optional[str] = Field(default=None, description="Error message if training failed")


class ListTrainingsResponse(BaseModel):
    """Response model for listing training jobs."""

    trainings: List[TrainingObject] = Field(..., description="List of training jobs")
    total: int = Field(..., description="Total number of trainings matching filters")
    offset: int = Field(..., description="Current pagination offset")
    limit: int = Field(..., description="Current pagination limit")


class Tag(BaseModel):
    """Tag information object."""

    name: str = Field(..., description="Tag name")
    description: Optional[str] = Field(default=None, description="Tag description")


class ListTagsResponse(BaseModel):
    """Response model for listing tags."""

    tags: List[Tag] = Field(..., description="List of tags")
    total: int = Field(..., description="Total number of tags")


class TrainingEvent(BaseModel):
    """Training event object."""

    id: str = Field(..., description="Unique event identifier")
    training_id: str = Field(..., description="Training job ID")
    event_type: str = Field(..., description="Event type")
    message: str = Field(..., description="Event message")
    created_at: str = Field(..., description="Event timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional event metadata")


class ListEventsResponse(BaseModel):
    """Response model for listing training events."""

    events: List[TrainingEvent] = Field(..., description="List of training events")
    total: int = Field(..., description="Total number of events")
    offset: int = Field(..., description="Current pagination offset")
    limit: int = Field(..., description="Current pagination limit")


class MetricDataPoint(BaseModel):
    """Metric data point."""

    step: int = Field(..., description="Training step number")
    value: float = Field(..., description="Metric value")
    timestamp: str = Field(..., description="Measurement timestamp")


class ListMetricsResponse(BaseModel):
    """Response model for listing training metrics."""

    metric: str = Field(..., description="Metric name")
    data: List[MetricDataPoint] = Field(..., description="Metric data points")
    total: int = Field(..., description="Total number of data points")
    offset: int = Field(..., description="Current pagination offset")
    limit: int = Field(..., description="Current pagination limit")
