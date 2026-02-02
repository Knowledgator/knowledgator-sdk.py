"""Common types and enums for datasets."""

from enum import Enum


class ServiceType(str, Enum):
    """Dataset service types."""

    NER = "ner"
    TEXT_CLASSIFICATION = "text-classification"
    RELATION_EXTRACTION = "relation-extraction"


class DatasetFormat(str, Enum):
    """Dataset file formats."""

    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"


class DatasetStatus(str, Enum):
    """Dataset processing statuses."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
