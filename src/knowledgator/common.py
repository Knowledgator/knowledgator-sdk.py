"""Common types and enums shared across the SDK."""

from enum import Enum


class ServiceType(str, Enum):
    """Service types supported by the Knowledgator API.

    Used across models, datasets, and fine-tuning operations.
    """

    NER = "ner"
    TEXT_CLASSIFICATION = "text-classification"
    RELATION_EXTRACTION = "relation-extraction"
