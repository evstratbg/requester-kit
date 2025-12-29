from collections.abc import Mapping, Sequence
from typing import IO, Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T_co = TypeVar("T_co", bound=BaseModel, covariant=True)

retry_status_codes = Literal[
    400,  # Bad Request
    401,  # Unauthorized
    402,  # Payment Required
    403,  # Forbidden
    404,  # Not Found
    405,  # Method Not Allowed
    406,  # Not Acceptable
    407,  # Proxy Authentication Required
    408,  # Request Timeout
    409,  # Conflict
    410,  # Gone
    411,  # Length Required
    412,  # Precondition Failed
    413,  # Payload Too Large
    414,  # URI Too Long
    415,  # Unsupported Media Type
    416,  # Range Not Satisfiable
    417,  # Expectation Failed
    418,  # I'm a Teapot (April Fools' Day joke)
    421,  # Misdirected Request
    422,  # Unprocessable Entity (WebDAV)
    423,  # Locked (WebDAV)
    424,  # Failed Dependency (WebDAV)
    425,  # Too Early
    426,  # Upgrade Required
    428,  # Precondition Required
    429,  # Too Many Requests
    431,  # Request Header Fields Too Large
    451,  # Unavailable For Legal Reasons
]

RequestHeaders = dict[str, Any]
RequestParams = dict[str, Any]
RequestData = dict[str, Any]
FileContent = IO[bytes] | bytes | str
FileTypes = (
    FileContent
    | tuple[str | None, FileContent]
    | tuple[str | None, FileContent, str | None]
    | tuple[str | None, FileContent, str | None, Mapping[str, str]]
)
RequestFiles = Mapping[str, FileTypes] | Sequence[tuple[str, FileTypes]]
RequestJson = dict[str, Any] | list[dict[str, Any]]
RequestContent = str | bytes
RequestCookies = dict[str, Any]
RequestAuth = tuple[str, str]

PositiveInt = Annotated[int, Field(strict=True, ge=0)]
PositiveFloat = Annotated[float, Field(strict=True, ge=0)]


class RequesterKitResponse[T_co](BaseModel):
    status_code: int | None = None
    is_ok: bool
    parsed_data: T_co | None = None
    raw_data: bytes = b""


class BaseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetryerSettings(BaseSettings):
    retries: PositiveInt = 0
    delay: PositiveFloat = 0.5
    increment: PositiveFloat = 0.1
    custom_status_codes: set[retry_status_codes] = set()


class LoggerSettings(BaseSettings):
    log_error_for_4xx: bool = True
    log_error_for_5xx: bool = True
