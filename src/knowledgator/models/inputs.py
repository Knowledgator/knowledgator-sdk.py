from typing import Any, Dict, List, Optional

from pydantic import Field, BaseModel, ConfigDict, AliasChoices, constr, confloat

from .common import FileMetadata


class ClassificationExample(BaseModel):
    text: str
    label: str


class TextLabels(BaseModel):
    text: constr(min_length=1) = Field(..., description="Input text")
    labels: List[str] = Field(..., description="Labels for classification/recogntiion", min_length=1)
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(default=0.5, description="Score cut-off")



class ExtractBaseRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    labels: List[str] = Field(
        ...,
        alias="entities",
        serialization_alias="entities",
        validation_alias=AliasChoices("entities", "labels"),
        description="Entities/labels for extraction",
    )
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(default=0.5, description="Score cut-off")
    prompt: Optional[str] = None


class ExtractRelationsRequest(BaseModel):
    text: str
    relations: List[str]
    entities: List[str]
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(default=0.5, description="Score cut-off")
    rules: Optional[List[Dict[str, Any]]] = None
    prompt: Optional[str] = None


class ExtractSummaryRequest(BaseModel):
    text: str
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(default=0.5, description="Score cut-off")
    prompt: Optional[str] = None


class InstructRequest(BaseModel):
    """Request model for Instruct-it API endpoint."""

    text: str = Field(..., description="The main instruction/prompt for the model")
    template: Optional[str] = Field(
        default=None, description="JSON string representing a structure the model should fill or emulate"
    )
    files: Optional[List[FileMetadata]] = Field(
        default=None, description="References to pre-uploaded or otherwise accessible files"
    )
    output_type: Optional[str] = Field(
        default=None, description="High-level hint for the desired output style (json, string, list, markdown, etc.)"
    )
