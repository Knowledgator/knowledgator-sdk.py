"""Common types and enums for fine-tuning."""

from enum import Enum


class ServiceType(str, Enum):
    """Fine-tuning service types."""

    NER = "ner"
    TEXT_CLASSIFICATION = "text-classification"
    RELATION_EXTRACTION = "relation-extraction"


class TrainingStatus(str, Enum):
    """Training completion statuses."""

    ACTIVE = "active"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"
