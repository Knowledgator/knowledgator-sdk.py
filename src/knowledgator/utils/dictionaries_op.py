from __future__ import annotations

from typing import Any, Dict


def remove_none_from_dict(d: dict | None) -> dict:
    """Remove keys with None values from a dictionary."""
    if d is None:
        return {}
    return {k: v for k, v in d.items() if v is not None}


def remove_omit_from_dict(dict: Dict[Any, Any], omit: Any):
    if omit is None:
        return dict
    new_dict = {}
    for k, v in dict.items():
        if v != omit:
            new_dict[k] = v
    return new_dict


def get_model_id_header(model_id: str | None) -> dict[str, str]:
    """Get the model ID header if model_id is provided, otherwise return empty dict."""
    return {"KnowledgatorAPI-Model-ID": model_id} if model_id else {}
