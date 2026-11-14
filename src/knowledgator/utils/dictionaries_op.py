from __future__ import annotations

from typing import Any, Dict


def remove_none_from_dict(dict: Dict[Any, Any]):
    """Remove keys with None values from a dictionary."""
    new_dict = {}
    for k, v in dict.items():
        if v is not None:
            new_dict[k] = v
    return new_dict


def remove_omit_from_dict(dict: Dict[Any, Any], omit: Any):
    if omit is None:
        return dict
    new_dict = {}
    for k, v in dict.items():
        if v != omit:
            new_dict[k] = v
    return new_dict


def get_model_id_header(model_id: Any = None) -> Dict[str, str]:
    """Get the model ID header if model_id is provided, otherwise return empty dict."""
    if model_id is not None:
        return {
            "KnowledgatorAPI-Model-ID": str(model_id),
            "X-Model-Id": str(model_id),
        }
    return {}


def get_inference_tasks(raw: Dict[str, Any]) -> Any:
    """Extract tasks dictionary from response payload (supporting 'outputs' or raw dict)."""
    if "outputs" in raw:
        return raw["outputs"]
    return raw

