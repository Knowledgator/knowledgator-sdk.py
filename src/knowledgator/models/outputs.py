from importlib.resources import files
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, constr, confloat

class ComprehendResponse(BaseModel):
    label: str = Field(..., description="Predicted label")
    score: float = Field(..., description="Confidence score for the prediction")

class EntityResponse(BaseModel):
    span: str = Field(..., description="Text span of the recognized entity")
    entity: str = Field(..., description="Recognized entity type")
    start: int = Field(..., description="Start index of the entity in the text")
    end: int = Field(..., description="End index of the entity in the text")
    score: float = Field(..., description="Confidence score for the entity recognition")

class RelationResponse(BaseModel):
    source: EntityResponse = Field(..., description="Source entity of the relation")
    relation: str = Field(..., description="Type of the relation")
    target: EntityResponse = Field(..., description="Target entity of the relation")
    score: float = Field(..., description="Confidence score for the relation")

class TextSpan(BaseModel):
    text: constr(min_length=1) = Field(..., description="Text span")
    start: int = Field(..., description="Start index of the text span")
    end: int = Field(..., description="End index of the text span")
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Score cut-off for the text span"
    )

class ComprehendResults(BaseModel):
    results: List[ComprehendResponse] = Field(
        ..., description="List of classification results"
    )
    input_text: str = Field(..., description="Original input text")
    labels: List[str] = Field(
        ..., description="Labels used for classification", min_items=1
    )
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Score cut-off"
    )
    status: str = Field(..., description="Status of the classification request")


class ExtractBaseResults(BaseModel):
    results: List[EntityResponse] = Field(
        ..., description="List of extracted entities"
    )
    input_text: str = Field(..., description="Original input text")
    labels: List[str] = Field(
        ..., description="Labels used for classification", min_items=1
    )
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Score cut-off"
    )
    status: str = Field(..., description="Status of the classification request")

class ExtractRelationsResults(BaseModel):
    results: List[RelationResponse] = Field(
        ..., description="List of extracted relations"
    )
    input_text: str = Field(..., description="Original input text")
    entities: List[str] = Field(
        ..., description="Entities used for relation extraction", min_items=1
    )
    relations: List[str] = Field(
        ..., description="Relations used for extraction", min_items=1
    )
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Score cut-off"
    )
    status: str = Field(..., description="Status of the relation extraction request")


class ExtractSummaryResults(BaseModel):
    summary: str = Field(..., description="Extracted summary of the input text")
    input_text: str = Field(..., description="Original input text")
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Score cut-off for the summary"
    )
    status: str = Field(..., description="Status of the summary extraction request")


# Training models
class ObjectMeta(BaseModel):
    """Base object metadata."""
    id: str = Field(..., description="Object ID")
    name: str = Field(..., description="Object name")


class TrainingDatasets(BaseModel):
    """Training and validation dataset information."""
    num_of_examples: int = Field(..., description="Number of training examples")
    training: ObjectMeta = Field(..., description="Training dataset metadata")
    validation: ObjectMeta = Field(..., description="Validation dataset metadata")


class TrainingObject(BaseModel):
    """Training job details."""
    id: str = Field(..., description="Training job ID")
    name: str = Field(..., description="Training job name")
    type: str = Field(..., description="Model type (e.g., comprehend-it, extract-it)")
    model: ObjectMeta = Field(..., description="Fine-tuned model metadata")
    parent_model: ObjectMeta = Field(..., description="Base model metadata")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    num_of_epochs: int = Field(..., description="Number of training epochs")
    datasets: TrainingDatasets = Field(..., description="Dataset information")
    loss: float = Field(..., description="Training loss value")
    status: str = Field(..., description="Training status (active, completed, stopped, failed)")


# Training list / events / metrics / evaluations
class TrainingsListResponse(BaseModel):
    data: List[TrainingObject]
    total: int


class TrainingEvent(BaseModel):
    timestamp: str = Field(..., description="Event timestamp")
    level: Optional[str] = Field(None, description="Event level (info, warning, error)")
    message: str = Field(..., description="Event message body")


class TrainingEventsListResponse(BaseModel):
    data: List[TrainingEvent]
    total: int


class TrainingMetricPoint(BaseModel):
    step: int = Field(..., description="Training step or epoch")
    value: float = Field(..., description="Metric value at this step")


class TrainingMetricResponse(BaseModel):
    metric: str = Field(..., description="Metric name")
    points: List[TrainingMetricPoint] = Field(..., description="Time series / steps for metric")


class EvaluationObject(BaseModel):
    dataset_id: str = Field(..., description="Evaluation dataset id")
    metrics: Dict[str, float] = Field(..., description="Evaluation metrics (e.g., accuracy, f1)")
    created_at: Optional[str] = Field(None, description="When evaluation was performed")


class EvaluationsListResponse(BaseModel):
    data: List[EvaluationObject]
    total: int


# Fine-tuning Models API output models
class ModelObject(BaseModel):
    """Base model object for training."""
    id: str
    name: str
    type: str
    description: Optional[str] = None
    short_description: Optional[str] = None
    created_at: str
    updated_at: str
    status: Optional[str] = None  


class ModelTypeObject(BaseModel):
    """Model type information."""
    type: str = Field(..., description="Model type identifier")
    name: str = Field(..., description="Display name for the model type")
    description: Optional[str] = Field(None, description="Model type description")


class ModelsListResponse(BaseModel):
    """Response for list models endpoint."""
    data: List[ModelObject]
    total: int 


class ModelTypesListResponse(BaseModel):
    data: List[str]
    total: int

class FileObject(BaseModel):
    """File object details."""
    id: str = Field(..., description="File ID")
    size: int = Field(..., description="File size in bytes")
    created_by: str = Field(..., description="Created by user ID")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    application_id: str = Field(..., description="Application ID associated with the file")


# Datasets API output models
class DatasetObject(BaseModel):
    """Dataset object details."""
    id: str = Field(..., description="Dataset ID")
    name: str = Field(..., description="Dataset name")
    type: str = Field(..., description="Dataset type (e.g., comprehend-it, extract-it)")
    format: str = Field(..., description="Dataset format (e.g., json, csv)")
    short_description: Optional[str] = Field(None, description="Dataset description")
    created_by: str = Field(..., description="Created by user ID")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    examples_count: Optional[int] = Field(None, description="Number of examples in dataset")
    status: Optional[str] = Field(None, description="Dataset status")
    files: List[FileObject] = Field(..., description="List of files in the dataset")

class DatasetsListResponse(BaseModel):
    """Response for list datasets endpoint."""
    data: List[DatasetObject]
    total: int