from .base import _STATUS_MAP, APIError, ClientError, ServerError


def from_http_response(resp) -> APIError:
    """
    Turn an `httpx.Response` into the corresponding APIError subclass.

    Accepts anything w/ .status_code, .json(), .text, .headers.
    """
    cls = _STATUS_MAP.get(resp.status_code)
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
