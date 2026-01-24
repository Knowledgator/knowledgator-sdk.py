from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, constr, confloat

from .common import FileMetadata


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
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=0.5, description="Score cut-off"
    )
    prompt: Optional[str] = None


class InstructRequest(BaseModel):
    """Request model for Instruct-it API endpoint."""
    text: str = Field(..., description="The main instruction/prompt for the model")
    template: Optional[str] = Field(
        default=None,
        description="JSON string representing a structure the model should fill or emulate"
    )
    files: Optional[List[FileMetadata]] = Field(
        default=None,
        description="References to pre-uploaded or otherwise accessible files"
    )
    output_type: Optional[str] = Field(
        default=None,
        description="High-level hint for the desired output style (json, string, list, markdown, etc.)"
    )
