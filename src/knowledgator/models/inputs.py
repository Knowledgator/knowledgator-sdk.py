from typing import List, Optional, Dict
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

