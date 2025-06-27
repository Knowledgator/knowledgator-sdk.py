from typing import List, Optional, Dict
from pydantic import BaseModel, Field, constr, confloat

class ComprehendResponse(BaseModel):
    label: str = Field(..., description="Predicted label")
    score: float = Field(..., description="Confidence score for the prediction")

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