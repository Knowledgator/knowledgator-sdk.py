import json
from typing import Any, List, Optional

from pydantic import Field, BaseModel, constr, confloat


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
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(default=0.5, description="Score cut-off for the text span")


class ComprehendResults(BaseModel):
    results: List[ComprehendResponse] = Field(..., description="List of classification results")
    input_text: str = Field(..., description="Original input text")
    labels: List[str] = Field(..., description="Labels used for classification", min_length=1)
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(default=0.5, description="Score cut-off")
    status: str = Field(..., description="Status of the classification request")


class ExtractBaseResults(BaseModel):
    results: List[EntityResponse] = Field(..., description="List of extracted entities")
    input_text: str = Field(..., description="Original input text")
    labels: List[str] = Field(..., description="Labels used for classification", min_length=1)
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(default=0.5, description="Score cut-off")
    status: str = Field(..., description="Status of the classification request")


class ExtractRelationsResults(BaseModel):
    results: List[RelationResponse] = Field(..., description="List of extracted relations")
    input_text: str = Field(..., description="Original input text")
    entities: List[str] = Field(..., description="Entities used for relation extraction", min_length=1)
    relations: List[str] = Field(..., description="Relations used for extraction", min_length=1)
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(default=0.5, description="Score cut-off")
    status: str = Field(..., description="Status of the relation extraction request")


class ExtractSummaryResults(BaseModel):
    summary: str = Field(..., description="Extracted summary of the input text")
    input_text: str = Field(..., description="Original input text")
    threshold: Optional[confloat(ge=0.0, le=1.0)] = Field(default=0.5, description="Score cut-off for the summary")
    status: str = Field(..., description="Status of the summary extraction request")


class InstructResults(BaseModel):
    outputs: Any = Field(..., description="Any JSON-serializable value: object, list, string, or nested structure")
    parsed: bool = Field(
        default=True, description="True if outputs was successfully parsed as JSON, False if raw string"
    )

    @classmethod
    def from_model_response(cls, raw_response: str) -> "InstructResults":
        """
        Create an InstructResults from raw model output.

        Attempts to parse as JSON. If parsing fails, returns the raw string
        with parsed=False.
        """
        try:
            parsed_output = json.loads(raw_response)
            return cls(outputs=parsed_output, parsed=True)
        except json.JSONDecodeError:
            return cls(outputs=raw_response, parsed=False)
