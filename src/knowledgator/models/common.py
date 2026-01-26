from typing import Optional

from pydantic import Field, BaseModel


class FileMetadata(BaseModel):
    """Metadata for a file reference, optionally with content.

    Attributes:
        name: File name, ID, or reference path.
        content: Optional file content as bytes. If None, the file is treated
                 as a reference only (e.g., for the model to acknowledge abstractly).
    """

    name: str = Field(..., description="File name, ID, or reference path")
    content: Optional[bytes] = Field(
        default=None,
        description="Optional file content as bytes",
        exclude=True,  # Exclude binary content from JSON serialization
    )
