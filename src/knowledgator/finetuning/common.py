"""Common types and enums for fine-tuning."""

from enum import Enum

from ..common import ServiceType

__all__ = ["ServiceType", "TrainingStatus"]


class TrainingStatus(str, Enum):
    """Training completion statuses."""

    ACTIVE = "active"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"
