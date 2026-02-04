"""Common types and enums for datasets."""

from enum import Enum

from ..common import ServiceType

__all__ = ["ServiceType", "DatasetFormat", "DatasetStatus"]


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
