from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import IO, Annotated, Any, Generic, Literal, Optional, TypeVar, Union

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
FileContent = Union[IO[bytes], bytes, str]
FileTypes = Union[
    FileContent,
    tuple[Optional[str], FileContent],
    tuple[Optional[str], FileContent, Optional[str]],
    tuple[Optional[str], FileContent, Optional[str], Mapping[str, str]],
]
RequestFiles = Union[Mapping[str, FileTypes], Sequence[tuple[str, FileTypes]]]
RequestJson = Union[dict[str, Any], list[dict[str, Any]]]
RequestContent = Union[str, bytes]
RequestCookies = dict[str, Any]
RequestAuth = tuple[str, str]

PositiveInt = Annotated[int, Field(strict=True, ge=0)]
PositiveFloat = Annotated[float, Field(strict=True, ge=0)]


class RequesterKitResponse(BaseModel, Generic[T_co]):
    status_code: Optional[int] = None
    is_ok: bool
    parsed_data: Optional[T_co] = None
    raw_data: bytes = b""


class BaseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrySettings(BaseSettings):
    retries: PositiveInt = 0
    delay: PositiveFloat = 0.5
    increment: PositiveFloat = 0.1
    custom_status_codes: set[retry_status_codes] = set()


class LoggerSettings(BaseSettings):
    log_error_for_4xx: bool = True
    log_error_for_5xx: bool = True
