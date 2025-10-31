"""Pydantic models for request and response validation."""
from .inputs import (
    ComprehendRequest,
    ExtractBaseRequest,
    ExtractRelationsRequest,
    ExtractSummaryRequest,
    TrainingCreateRequest,
    TrainingArgs,
    TrainingMetadata,
    ValidationMetadata,
)
from .outputs import (
    ComprehendResponse,
    ComprehendResults,
    EntityResponse,
    RelationResponse,
    ExtractBaseResults,
    ExtractRelationsResults,
    ExtractSummaryResults,
    TrainingObject,
    TrainingDatasets,
    ObjectMeta,
)

__all__ = [
    # Input models
    "ComprehendRequest",
    "ExtractBaseRequest",
    "ExtractRelationsRequest",
    "ExtractSummaryRequest",
    "TrainingCreateRequest",
    "TrainingArgs",
    "TrainingMetadata",
    "ValidationMetadata",
    # Output models
    "ComprehendResponse",
    "ComprehendResults",
    "EntityResponse",
    "RelationResponse",
    "ExtractBaseResults",
    "ExtractRelationsResults",
    "ExtractSummaryResults",
    "TrainingObject",
    "TrainingDatasets",
    "ObjectMeta",
]
