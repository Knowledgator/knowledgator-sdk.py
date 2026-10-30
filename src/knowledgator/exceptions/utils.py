from .base import APIError, ClientError, ServerError
from . import client_errors, server_errors


def _get_status_map():
    mapping = {}
    for base_cls in (ClientError, ServerError):
        for sub_cls in base_cls.__subclasses__():
            code = getattr(sub_cls, "code", None)
            if code is not None:
                mapping[code] = sub_cls
    return mapping


def from_http_response(resp) -> APIError:
    """
    Turn an `httpx.Response` into the corresponding APIError subclass.

    Accepts anything w/ .status_code, .json(), .text, .headers.
    """
    status_map = _get_status_map()
    cls = status_map.get(resp.status_code)
    if cls is None:
        # Fallback class keeps hierarchy correct
        cls = ClientError if 400 <= resp.status_code < 500 else ServerError

    try:
        payload = resp.json()
    except ValueError:
        payload = {"raw": resp.text}

    return cls(
        f"{resp.status_code} {resp.reason_phrase}",
        status_code=resp.status_code,
        payload=payload,
        request_id=resp.headers.get("x-request-id"),
    )
