"""Output models for fine-tuning API responses."""

from typing import Any, Dict, List, Optional

from pydantic import Field, BaseModel, ConfigDict, AliasChoices, model_validator


class TrainingModel(BaseModel):
    """Training model object with full details."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("id", "model_id"),
        description="Unique model identifier",
    )
    name: Optional[str] = Field(default=None, description="Model name")
    type: Optional[str] = Field(default=None, description="Service type")
    description: Optional[str] = Field(default=None, description="Model description")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")
    is_base: Optional[bool] = Field(
        default=None,
        description="Whether this is a base model (not trained by customer)",
    )
    training_model: Optional[bool] = Field(default=None)
    training_adapter: Optional[bool] = Field(default=None)


class ListModelsResponse(BaseModel):
    """Response model for listing models."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    models: Optional[List[TrainingModel]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("models", "results", "data"),
        description="List of models",
    )
    total: Optional[int] = Field(default=None, description="Total number of models matching filters")
    offset: Optional[int] = Field(default=None, description="Current pagination offset")
    limit: Optional[int] = Field(default=None, description="Current pagination limit")


class ObjectCreation(BaseModel):
    """Response model after creating an object."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("id", "training_id", "model_id"),
        description="Object identifier",
    )
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")


class TrainingObject(BaseModel):
    """Training job object with full details."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("id", "training_id", "name"),
        description="Unique training identifier",
    )
    model_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("model_id", "parent_model_id"),
        description="Base model ID",
    )
    dataset_id: Optional[str] = Field(default=None, description="Dataset ID")
    status: Optional[str] = Field(default=None, description="Training status")
    name: Optional[str] = Field(default=None, description="Training name")
    type: Optional[str] = Field(default=None, description="Training type")
    hyperparameters: Optional[Any] = Field(
        default=None,
        validation_alias=AliasChoices("hyperparameters", "args", "training"),
        description="Training hyperparameters",
    )
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("updated_at", "last_updated_at"),
        description="Last update timestamp",
    )
    started_at: Optional[str] = Field(default=None, description="Training start timestamp")
    completed_at: Optional[str] = Field(default=None, description="Training completion timestamp")
    error: Optional[str] = Field(default=None, description="Error message if training failed")

    @model_validator(mode="before")
    @classmethod
    def _extract_nested_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if ("dataset_id" not in data or data["dataset_id"] is None) and isinstance(data.get("training"), dict):
                data["dataset_id"] = data["training"].get("dataset_id")
        return data


class ListTrainingsResponse(BaseModel):
    """Response model for listing training jobs."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    trainings: Optional[List[TrainingObject]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("trainings", "results", "data"),
        description="List of training jobs",
    )
    total: Optional[int] = Field(default=None, description="Total number of trainings matching filters")
    offset: Optional[int] = Field(default=None, description="Current pagination offset")
    limit: Optional[int] = Field(default=None, description="Current pagination limit")


class Tag(BaseModel):
    """Tag information object."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: str = Field(..., description="Tag name")
    description: Optional[str] = Field(default=None, description="Tag description")


class ListTagsResponse(BaseModel):
    """Response model for listing tags."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    tags: Optional[List[Tag]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("tags", "types", "results", "data"),
        description="List of tags",
    )
    total: Optional[int] = Field(default=None, description="Total number of tags")


class TrainingEvent(BaseModel):
    """Training event object."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: Optional[str] = Field(default=None, description="Unique event identifier")
    training_id: Optional[str] = Field(default=None, description="Training job ID")
    event_type: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("event_type", "type"),
        description="Event type",
    )
    message: Optional[str] = Field(default=None, description="Event message")
    created_at: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("created_at", "timestamp"),
        description="Event timestamp",
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional event metadata")


class ListEventsResponse(BaseModel):
    """Response model for listing training events."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    events: Optional[List[TrainingEvent]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("events", "results", "data"),
        description="List of training events",
    )
    total: Optional[int] = Field(default=None, description="Total number of events")
    offset: Optional[int] = Field(default=None, description="Current pagination offset")
    limit: Optional[int] = Field(default=None, description="Current pagination limit")


class MetricDataPoint(BaseModel):
    """Metric data point."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    step: Optional[int] = Field(default=None, description="Training step number")
    value: Optional[float] = Field(default=None, description="Metric value")
    timestamp: Optional[str] = Field(default=None, description="Measurement timestamp")


class ListMetricsResponse(BaseModel):
    """Response model for listing training metrics."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    metric: Optional[str] = Field(default=None, description="Metric name")
    data: Optional[List[MetricDataPoint]] = Field(
        default_factory=list,
        validation_alias=AliasChoices("data", "values", "results"),
        description="Metric data points",
    )
    total: Optional[int] = Field(default=None, description="Total number of data points")
    offset: Optional[int] = Field(default=None, description="Current pagination offset")
    limit: Optional[int] = Field(default=None, description="Current pagination limit")
