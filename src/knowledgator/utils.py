from typing import Dict, Any, Optional

def remove_none_from_dict(dict: Dict[Any, Any]):
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

