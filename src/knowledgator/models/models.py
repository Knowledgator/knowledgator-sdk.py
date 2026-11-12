from __future__ import annotations
from typing import Generic, TypeVar, Literal, Any, Annotated

import jsonschema
from pydantic import (
    Field, field_validator, model_validator, BaseModel, RootModel, JsonValue, PrivateAttr
)
from pydantic_core import PydanticCustomError


TaskType = TypeVar("TaskType", bound=Literal[
    "classify", "boolean", "score", "span", "relation",
    "json", "embedding", "rank", "box", "mask", "count", "time_span"
])


class BaseSchema(BaseModel):
    model_config = {
        "extra": "forbid"
    }


class LabelDefinition(BaseSchema):
    title: str | None = None
    description: str | None = None
    metadata: dict[str, JsonValue] | None = None


class LabelSpec(RootModel[Any]):
    model_config = {"defer_build": True}
    root: str | list[str] | dict[str, str | LabelDefinition | None]
    _ids: list[str] = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        if isinstance(self.root, str):
            self._ids = [self.root]
        elif isinstance(self.root, list):
            self._ids = self.root
        else:
            self._ids = list(self.root.keys())
        for label_id in self._ids:
            if not label_id or not label_id.strip():
                raise PydanticCustomError("empty_label", "Labels must be non-empty strings")

    def __len__(self):
        return len(self.root)

    @property
    def ids(self):
        return self._ids


Span = Annotated[str, Field(min_length=1)]

MediaType = TypeVar("MediaType", bound=Literal["image", "audio", "layout"])
class MediaNode(BaseSchema, Generic[MediaType]):
    type: MediaType


class AnalyzeImageNode(MediaNode[Literal["image"]]):
    type: Literal["image"]
    data: str | None
    mime_type: str | None 
    file_id: str | None
    url: str | None


class AnalyzeAudioNode(MediaNode[Literal["audio"]]):
    type: Literal["audio"]
    data: str | None 
    mime_type: str | None 
    file_id: str | None
    url: str | None

Box = tuple[float, float, float, float]
class LayoutWord(BaseSchema):
    text: str 
    box: Box


class AnalyzeLayoutPage(BaseSchema):
  width: float | None
  height: float | None # required when coordinates = "absolute"
  words: list[LayoutWord] | list[str]
  boxes: list[Box]


class AnalyzeLayoutNode(MediaNode[Literal["layout"]]):
    type: Literal["layout"]
    coordinates: Literal["absolute", "normalized_1000", "relative"] = "absolute"
    pages: list[AnalyzeLayoutPage] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_absolute_coords(self):
        if self.coordinates == "absolute":
            for i, page in enumerate(self.pages):
                if page.height is None:
                    raise PydanticCustomError("invalid_label", f"pages[{i}].height is required when coordinates='absolute'")
        return self


AnalyzeMediaNode = AnalyzeLayoutNode | AnalyzeImageNode | AnalyzeAudioNode


class AnalyzeInput(RootModel[Any]):
    model_config = {"defer_build": True}
    root: str | int | float | bool | list['AnalyzeInput'] | dict[str, 'AnalyzeInput'] | AnalyzeMediaNode | None


class AnalyzeCandidate(BaseSchema):
    id: str = Field(min_length=1)
    input: AnalyzeInput


class AnalyzeTask(BaseSchema, Generic[TaskType]):
    type: TaskType
    instruction: str | None = None

    model_config = {"populate_by_name": True}


class AnalyzeTaskClassify(AnalyzeTask[Literal["classify"]]):
    type: Literal["classify"]
    threshold: float | None = Field(None, ge=0, le=1)
    labels: LabelSpec = Field(min_length=1)
    multi_label: bool | None = None


class AnalyzeTaskBoolean(AnalyzeTask[Literal["boolean"]]):
    type: Literal["boolean"]
    threshold: float | None = Field(None, ge=0, le=1)
    labels: LabelSpec | None = None

    @field_validator("labels")
    @classmethod
    def validate_boolean_labels(cls, v: LabelSpec | None) -> LabelSpec | None:
        if v:
            for label in v.ids:
                if label not in ("true", "false"):
                    raise PydanticCustomError("invalid_label", f"'boolean' labels may only define 'true' and 'false', got {label!r}")
        return v


class AnalyzeTaskScore(AnalyzeTask[Literal["score"]]):
    type: Literal["score"]
    threshold: float | None = Field(None, ge=0, le=1)
    labels: LabelSpec = Field(min_length=2)
    order: list[str] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_order_matches_labels(self):
        u_order = set(self.order)
        if u_order != set(self.labels.ids) or len(u_order) != len(self.order):
            raise PydanticCustomError("invalid_order", "'order' must list every label ID exactly once")
        return self


class AnalyzeTaskSpan(AnalyzeTask[Literal["span"]]):
    type: Literal["span"]
    threshold: float | None = Field(None, ge=0, le=1)
    labels: LabelSpec | None = Field(None, min_length=1)
    flat: bool | None = True


class AnalyzeTaskRelation(AnalyzeTask[Literal["relation"]]):
    type: Literal["relation"]
    threshold: float | None = Field(None, ge=0, le=1)
    labels: LabelSpec = Field(min_length=1)
    flat: bool | None = True
    entity_labels: LabelSpec | None = Field(None, min_length=1)


class AnalyzeTaskJson(AnalyzeTask[Literal["json"]]):
    type: Literal["json"]
    labels: LabelSpec | None = None
    schema_: dict[str, Any] | None = Field(None, alias="schema")

    @model_validator(mode="after")
    def validate_labels_or_schema(self):
        has_labels = self.labels is not None
        has_schema = self.schema_ is not None
        if has_labels and has_schema:
            raise PydanticCustomError("invalid_task", "A 'json' task cannot have both 'labels' and 'schema'")
        if not has_labels and not has_schema:
            raise PydanticCustomError("invalid_task", "A 'json' task needs either 'labels' or 'schema'")
        return self

    @field_validator("schema_", mode="before")
    @classmethod
    def validate_schema(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        try:
            if value is not None:
                jsonschema.Draft202012Validator.check_schema(value)
            return value
        except:
            raise PydanticCustomError("invalid_schema", "JSON schema is expected")



class AnalyzeTaskEmbedding(AnalyzeTask[Literal["embedding"]]):
    type: Literal["embedding"]
    normalize: bool = True


class AnalyzeTaskRank(AnalyzeTask[Literal["rank"]]):
    type: Literal["rank"]
    candidates: list[AnalyzeCandidate] = Field(min_length=1)
    top_k: int | None = Field(None, ge=1)

    @field_validator("candidates")
    @classmethod
    def validate_unique_candidate_ids(cls, v: list[AnalyzeCandidate]) -> list[AnalyzeCandidate]:
        seen: set[str] = set()
        for c in v:
            if c.id in seen:
                raise PydanticCustomError("invalid_candidate", f"Duplicate candidate ID {c.id!r}")
            seen.add(c.id)
        return v


class AnalyzeTaskBox(AnalyzeTask[Literal["box"]]):
    type: Literal["box"]
    threshold: float | None = Field(None, ge=0, le=1)
    labels: LabelSpec | None = None


class AnalyzeTaskMask(AnalyzeTask[Literal["mask"]]):
    type: Literal["mask"]
    threshold: float | None = Field(None, ge=0, le=1)
    labels: LabelSpec | None = None
    mode: Literal["semantic", "instance", "panoptic"] | None = None


class AnalyzeTaskCount(AnalyzeTask[Literal["count"]]):
    type: Literal["count"]
    labels: LabelSpec | None = None


class AnalyzeTaskTimeSpan(AnalyzeTask[Literal["time_span"]]):
    type: Literal["time_span"]
    threshold: float | None = Field(None, ge=0, le=1)
    labels: LabelSpec | None = None
    mode: Literal["events", "diarization"] | None = None


class AnalyzeOptions(BaseSchema):
    include_label_definitions: bool = False
    include_usage: bool = False
    include_evidence: bool = False
    include_layout_text: bool = False
    reference_date: str | None = None
    adapter_id: str | None = None


AnalyzeTaskType = TypeVar(
    "AnalyzeTaskType",
    bound=(
        AnalyzeTaskClassify | AnalyzeTaskBoolean | AnalyzeTaskScore |
        AnalyzeTaskSpan | AnalyzeTaskRelation | AnalyzeTaskJson |
        AnalyzeTaskEmbedding | AnalyzeTaskRank |
        AnalyzeTaskBox | AnalyzeTaskMask | AnalyzeTaskCount | AnalyzeTaskTimeSpan
    ),
)


class AnalyzeRequest(BaseSchema, Generic[AnalyzeTaskType]):
    # model: str | None = None
    input: AnalyzeInput
    tasks: dict[str, AnalyzeTaskType] = Field(min_length=1)
    options: AnalyzeOptions | None = None

    @field_validator("input")
    @classmethod
    def validate_input_not_empty(cls, v: AnalyzeInput) -> AnalyzeInput:
        if v.root != 0 and not v.root:
            raise PydanticCustomError("invalid_input", "input cannot be empty")
        return v


class BatchItem(BaseSchema, Generic[AnalyzeTaskType]):
    id: str = Field(min_length=1)
    input: AnalyzeInput
    tasks: dict[str, AnalyzeTaskType] = Field(min_length=1)

    @field_validator("input")
    @classmethod
    def validate_input_not_empty(cls, v: AnalyzeInput) -> AnalyzeInput:
        if v.root != 0 and not v.root:
            raise PydanticCustomError("invalid_input", "input cannot be empty")
        return v


class BatchRequest(BaseSchema, Generic[AnalyzeTaskType]):
    items: list[BatchItem[AnalyzeTaskType]] = Field(min_length=1)
    # model: str | None = None
    options: AnalyzeOptions | None = None


ComprehendTasks = Annotated[AnalyzeTaskClassify | AnalyzeTaskBoolean | AnalyzeTaskScore, Field(discriminator="type")]
class ComprehendRequest(AnalyzeRequest[ComprehendTasks]):
    ...


class ComprehendBatchRequest(BatchRequest[ComprehendTasks]):
    ...


ExtractTasks = Annotated[AnalyzeTaskSpan | AnalyzeTaskRelation, Field(discriminator="type")]
class ExtractRequest(AnalyzeRequest[ExtractTasks]):
    ...


class ExtractBatchRequest(BatchRequest[ExtractTasks]):
    ...

