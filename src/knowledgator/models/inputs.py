from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, constr, confloat


class ClassificationExample(BaseModel):
    text: str
    label: str

class TextLabels(BaseModel):
    text: constr(min_length=1) = Field(
        ..., description="Input text"
    )
    labels: List[str] = Field(
        ..., description="Labels for classification/recogntiion", min_items=1
    )
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Score cut-off"
    )

class ComprehendRequest(BaseModel):
    text: str
    labels: List[str]
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Score cut-off"
    )
    examples: Optional[List[ClassificationExample]] = None   # few-shot hints


class ExtractBaseRequest(BaseModel):
    text: str
    labels: List[str]
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Score cut-off"
    )
    prompt: Optional[str] = None



class ExtractRelationsRequest(BaseModel):
    text: str
    relations: List[str]
    entities: List[str]
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Score cut-off"
    )
    rules: Optional[List[Dict[str, Any]]] = None
    prompt: Optional[str] = None

class ExtractSummaryRequest(BaseModel):
    text: str


# Training models
class TrainingArgs(BaseModel):
    """Training hyperparameters."""
    seed: int = Field(..., description="Random seed for training")
    batch: Optional[int] = Field(None, description="Batch size")
    learning_rate_mult: Optional[int] = Field(None, description="Learning rate multiplier")
    num_of_epochs: Optional[int] = Field(None, description="Number of training epochs")


class TrainingMetadata(BaseModel):
    """Training dataset and arguments."""
    dataset_id: str = Field(..., min_length=3, max_length=64, pattern="^[a-zA-Z0-9-]+$", description="Training dataset ID")
    args: TrainingArgs = Field(..., description="Training hyperparameters")


class ValidationMetadata(BaseModel):
    """Validation dataset configuration."""
    dataset_id: Optional[str] = Field(None, min_length=3, max_length=64, pattern="^[a-zA-Z0-9-]+$", description="Validation dataset ID")


class TrainingCreateRequest(BaseModel):
    """Request to start a fine-tuning training job."""
    name: Optional[str] = Field(None, min_length=3, max_length=64, pattern="^[a-zA-Z0-9- .()]+$", description="Training job name")
    model_name: Optional[str] = Field(None, min_length=3, max_length=64, pattern="^[a-zA-Z0-9- .()]+$", description="Name for the fine-tuned model")
    parent_model_id: str = Field(..., min_length=3, max_length=64, pattern="^[a-zA-Z0-9-]+$", description="Base model ID to fine-tune")
    training: TrainingMetadata = Field(..., description="Training configuration")
    validation: Optional[ValidationMetadata] = Field(None, description="Validation configuration")
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Score cut-off"
    )
    prompt: Optional[str] = None


# Fine-tuning Models API input models
class ModelCreateRequest(BaseModel):
    """Request to create a base model for training."""
    name: str = Field(..., min_length=3, max_length=64, pattern="^[a-zA-Z0-9- .()]+$", description="Model name")
    type: str = Field(..., description="Model type (e.g., comprehend-it, extract-it)")
    description: Optional[str] = Field(None, description="Model description")


# Datasets API input models
class DatasetCreateRequest(BaseModel):
    """Request to create a new dataset."""
    name: str = Field(..., min_length=3, max_length=64, pattern="^[a-zA-Z0-9- .()]+$", description="Dataset name")
    type: str = Field(..., description="Dataset type (e.g., comprehend-it, extract-it)")
    description: Optional[str] = Field(None, description="Dataset description")


class DatasetUpdateRequest(BaseModel):
    """Request to update a dataset."""
    name: Optional[str] = Field(None, min_length=3, max_length=64, pattern="^[a-zA-Z0-9- .()]+$", description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")