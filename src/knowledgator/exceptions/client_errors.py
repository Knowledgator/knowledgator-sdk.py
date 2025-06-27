from .base import ClientError

# concrete 4xx
class BadRequest(ClientError):             code = 400
class Unauthorized(ClientError):           code = 401
class Forbidden(ClientError):              code = 403
class NotFound(ClientError):               code = 404
class Conflict(ClientError):               code = 409
class UnprocessableEntity(ClientError):    code = 422
class RateLimitExceeded(ClientError):      code = 429
