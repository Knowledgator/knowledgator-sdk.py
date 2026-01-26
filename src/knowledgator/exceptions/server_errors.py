from .base import ServerError


# concrete 5xx
class InternalServerError(ServerError):
    code = 500


class BadGateway(ServerError):
    code = 502


class ServiceUnavailable(ServerError):
    code = 503


class GatewayTimeout(ServerError):
    code = 504
