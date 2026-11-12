from __future__ import annotations

import json
from typing import Any, Dict, List, Union, Optional

from pydantic import Field, BaseModel, ConfigDict, constr, confloat, model_validator

# --- New Unified Inference Task Outputs ---


class BaseTaskOutput(BaseModel):
    """Base model for individual task prediction output."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class ClassifyTaskOutput(BaseTaskOutput):
    """Output for 'classify' task."""

    type: str = "classify"
    value: Optional[Union[str, List[str]]] = None
    scores: Optional[Dict[str, float]] = None
    threshold: Optional[float] = None
    adapter_id: Optional[str] = None


class BooleanTaskOutput(BaseTaskOutput):
    """Output for 'boolean' task."""

    type: str = "boolean"
    value: bool
    truth_score: Optional[float] = None
    scores: Optional[Dict[str, float]] = None
    threshold: Optional[float] = None
    adapter_id: Optional[str] = None


class ScoreTaskOutput(BaseTaskOutput):
    """Output for 'score' task."""

    type: str = "score"
    value: str
    scores: Optional[Dict[str, float]] = None
    threshold: Optional[float] = None
    adapter_id: Optional[str] = None


class SpanEntity(BaseModel):
    """Extracted text span entity."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    text: str
    start: int
    end: int
    label: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            d = dict(data)
            if "text" not in d and "span" in d:
                d["text"] = d["span"]
            elif "span" not in d and "text" in d:
                d["span"] = d["text"]
            if "label" not in d and "entity" in d:
                d["label"] = d["entity"]
            elif "entity" not in d and "label" in d:
                d["entity"] = d["label"]
            return d
        return data

    @property
    def span(self) -> str:
        return self.text

    @property
    def entity(self) -> Optional[str]:
        return self.label

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class SpanTaskOutput(BaseTaskOutput):
    """Output for 'span' task (named entity recognition / summarization)."""

    type: str = "span"
    value: List[SpanEntity] = Field(default_factory=list)
    threshold: Optional[float] = None


class RelationEntity(BaseModel):
    """Extracted relationship pair."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    head: SpanEntity
    relation: Optional[str] = None
    label: Optional[str] = None
    tail: SpanEntity
    score: Optional[float] = None

    def model_post_init(self, __context: Any) -> None:
        if self.label is None and self.relation is not None:
            self.label = self.relation
        elif self.relation is None and self.label is not None:
            self.relation = self.label

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class RelationTaskOutput(BaseTaskOutput):
    """Output for 'relation' task."""

    type: str = "relation"
    value: List[RelationEntity] = Field(default_factory=list)
    threshold: Optional[float] = None


def parse_task_output(task_data: Any, task_type: Optional[str] = None) -> Any:
    """Parse raw task output dictionary into typed task model if applicable."""
    if not isinstance(task_data, dict):
        return task_data

    t = task_data.get("type", task_type)
    if t == "classify":
        return ClassifyTaskOutput.model_validate(task_data)
    if t == "boolean":
        return BooleanTaskOutput.model_validate(task_data)
    if t == "score":
        return ScoreTaskOutput.model_validate(task_data)
    if t == "span":
        return SpanTaskOutput.model_validate(task_data)
    if t == "relation":
        return RelationTaskOutput.model_validate(task_data)
    return BaseTaskOutput.model_validate(task_data)


# --- Legacy Output Models (Preserved for compatibility) ---


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


# --- Response Containers ---


class ComprehendResults(BaseModel):
    """Result container for Comprehend API.

    Supports both new task-keyed outputs and legacy output lists.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    results: Any = Field(default_factory=dict, description="Task results or legacy classification outputs")
    outputs: Optional[Dict[str, Any]] = Field(default=None, description="Task results dictionary")
    input_text: Optional[str] = Field(default=None, description="Original input text")
    labels: Optional[List[str]] = Field(default=None, description="Labels used for classification")
    threshold: Optional[float] = Field(default=None, description="Score cut-off")
    status: str = Field(default="ok", description="Status of the classification request")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Usage metadata")
    raw: Optional[Dict[str, Any]] = Field(default=None, description="Raw response payload")

    def __init__(self, *args: Any, **data: Any):
        if args and isinstance(args[0], dict) and not data:
            raw = args[0]
            from knowledgator.utils.dictionaries_op import get_inference_tasks

            tasks_raw = get_inference_tasks(raw)
            if isinstance(tasks_raw, dict):
                parsed = {k: parse_task_output(v) for k, v in tasks_raw.items()}
                data = {
                    "results": parsed,
                    "outputs": parsed,
                    "raw": raw,
                    "usage": raw.get("usage"),
                    "status": "ok",
                }
            else:
                raise ValueError(
                    "Unexpected response shape: expected task-keyed dictionary in 'outputs'/'results'"
                )
        super().__init__(**data)

    def model_post_init(self, __context: Any) -> None:
        if self.outputs is None and isinstance(self.results, dict):
            self.outputs = self.results
        elif self.outputs is not None and (not self.results or self.results == {}):
            self.results = self.outputs

    def __getitem__(self, key: str) -> Any:
        if isinstance(self.results, dict) and key in self.results:
            return self.results[key]
        if self.outputs and key in self.outputs:
            return self.outputs[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        if isinstance(self.results, dict):
            return key in self.results
        if self.outputs:
            return key in self.outputs
        return False

    def __getattr__(self, name: str) -> Any:
        if isinstance(self.results, dict) and name in self.results:
            return self.results[name]
        if self.outputs and name in self.outputs:
            return self.outputs[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class ExtractResults(BaseModel):
    """Result container for Extract API.

    Supports task-keyed outputs (entities, relations, summary, etc.).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    results: Any = Field(default_factory=dict, description="Task results dictionary")
    outputs: Optional[Dict[str, Any]] = Field(default=None, description="Task results dictionary")
    input_text: Optional[str] = Field(default=None, description="Original input text")
    threshold: Optional[float] = Field(default=None, description="Score cut-off")
    status: str = Field(default="ok", description="Status of the extraction request")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Usage metadata")
    raw: Optional[Dict[str, Any]] = Field(default=None, description="Raw response payload")

    def __init__(self, *args: Any, **data: Any):
        if args and isinstance(args[0], dict) and not data:
            raw = args[0]
            from knowledgator.utils.dictionaries_op import get_inference_tasks

            tasks_raw = get_inference_tasks(raw)
            if isinstance(tasks_raw, dict):
                parsed = {k: parse_task_output(v) for k, v in tasks_raw.items()}
                data = {
                    "results": parsed,
                    "outputs": parsed,
                    "raw": raw,
                    "usage": raw.get("usage"),
                    "status": "ok",
                }
            else:
                raise ValueError(
                    "Unexpected response shape: expected task-keyed dictionary in 'outputs'/'results'"
                )
        super().__init__(**data)

    def model_post_init(self, __context: Any) -> None:
        if self.outputs is None and isinstance(self.results, dict):
            self.outputs = self.results
        elif self.outputs is not None and (not self.results or self.results == {}):
            self.results = self.outputs

    def __getitem__(self, key: str) -> Any:
        if isinstance(self.results, dict) and key in self.results:
            return self.results[key]
        if self.outputs and key in self.outputs:
            return self.outputs[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        if isinstance(self.results, dict):
            return key in self.results
        if self.outputs:
            return key in self.outputs
        return False

    def __getattr__(self, name: str) -> Any:
        if isinstance(self.results, dict) and name in self.results:
            return self.results[name]
        if self.outputs and name in self.outputs:
            return self.outputs[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


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
    def from_model_response(cls, raw_response: str) -> InstructResults:
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
