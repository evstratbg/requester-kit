from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, cast

if TYPE_CHECKING:
    from httpx import Response
    from prometheus_client import Counter, Histogram

    from requester_kit.client import BaseRequesterKit


@dataclass(frozen=True)
class PrometheusMetricsSettings:
    target: str
    count_payload_bytes: bool = False
    count_response_bytes: bool = False


_PROM_METRICS_SETTINGS: ContextVar[Optional[PrometheusMetricsSettings]] = ContextVar(
    "requester_kit_prometheus_metrics_settings",
    default=None,
)
_AsyncMethodT = TypeVar("_AsyncMethodT", bound=Callable[..., Any])
_PROM_HISTOGRAMS: dict[str, Histogram] = {}
_PROM_COUNTERS: dict[str, Counter] = {}
_PROM_REQUEST_DURATION_NAME = "requester_kit_request_duration_seconds"
_PROM_REQUEST_ERRORS_NAME = "requester_kit_request_errors_total"
_PROM_REQUEST_SIZE_NAME = "requester_kit_request_payload_bytes"
_PROM_RESPONSE_SIZE_NAME = "requester_kit_response_bytes"


def prometheus_metrics(
    name: Optional[str] = None,
    *,
    count_payload_bytes: bool = False,
    count_response_bytes: bool = False,
) -> Callable[[_AsyncMethodT], _AsyncMethodT]:
    def decorator(func: _AsyncMethodT) -> _AsyncMethodT:
        @wraps(func)
        async def wrapper(self: BaseRequesterKit, *args: Any, **kwargs: Any) -> Any:
            settings = PrometheusMetricsSettings(
                target=build_metric_target(
                    requester=self,
                    method_name=func.__name__,
                    metric_name=name,
                ),
                count_payload_bytes=count_payload_bytes,
                count_response_bytes=count_response_bytes,
            )
            token = _PROM_METRICS_SETTINGS.set(settings)
            try:
                return await func(self, *args, **kwargs)
            finally:
                _PROM_METRICS_SETTINGS.reset(token)

        return cast("_AsyncMethodT", wrapper)

    return decorator


def build_metric_target(
    requester: BaseRequesterKit,
    method_name: str,
    metric_name: Optional[str] = None,
) -> str:
    metric_method_name = metric_name or method_name
    return f"{type(requester).__name__}.{metric_method_name}"


def get_prometheus_metrics_settings() -> Optional[PrometheusMetricsSettings]:
    return _PROM_METRICS_SETTINGS.get()


def get_prometheus_histogram(name: str) -> Histogram:
    try:
        histogram = import_module("prometheus_client").Histogram
    except ImportError as exc:
        raise RuntimeError("prometheus_client is required when using @prometheus_metrics") from exc

    if name not in _PROM_HISTOGRAMS:
        _PROM_HISTOGRAMS[name] = histogram(
            name,
            "HTTP request duration in seconds",
            labelnames=("method", "status_code", "status_class", "attempt"),
        )
    return _PROM_HISTOGRAMS[name]


def get_prometheus_counter(name: str) -> Counter:
    try:
        counter = import_module("prometheus_client").Counter
    except ImportError as exc:
        raise RuntimeError("prometheus_client is required when using @prometheus_metrics") from exc

    if name not in _PROM_COUNTERS:
        _PROM_COUNTERS[name] = counter(
            name,
            "Total number of HTTP request errors",
            labelnames=("method", "status_code", "error_type", "attempt"),
        )
    return _PROM_COUNTERS[name]


def get_prometheus_size_histogram(name: str) -> Histogram:
    return get_prometheus_histogram(name)


def record_prometheus_request_start(
    attempt: str,
    request_content: bytes,
) -> Optional[PrometheusMetricsSettings]:
    settings = get_prometheus_metrics_settings()
    if settings is None:
        return None

    if settings.count_payload_bytes:
        get_prometheus_size_histogram(_PROM_REQUEST_SIZE_NAME).labels(
            method=settings.target,
            status_code="request",
            status_class="request",
            attempt=attempt,
        ).observe(len(request_content))
    return settings


def record_prometheus_response(
    *,
    settings: PrometheusMetricsSettings,
    response: Response,
    duration: float,
    attempt: str,
) -> None:
    get_prometheus_histogram(_PROM_REQUEST_DURATION_NAME).labels(
        method=settings.target,
        status_code=str(response.status_code),
        status_class=f"{response.status_code // 100}xx",
        attempt=attempt,
    ).observe(duration)

    if settings.count_response_bytes:
        get_prometheus_size_histogram(_PROM_RESPONSE_SIZE_NAME).labels(
            method=settings.target,
            status_code=str(response.status_code),
            status_class=f"{response.status_code // 100}xx",
            attempt=attempt,
        ).observe(len(response.content or b""))


def record_prometheus_http_error(
    *,
    settings: PrometheusMetricsSettings,
    duration: float,
    attempt: str,
) -> None:
    get_prometheus_histogram(_PROM_REQUEST_DURATION_NAME).labels(
        method=settings.target,
        status_code="exception",
        status_class="error",
        attempt=attempt,
    ).observe(duration)
    get_prometheus_counter(_PROM_REQUEST_ERRORS_NAME).labels(
        method=settings.target,
        status_code="exception",
        error_type="http_error",
        attempt=attempt,
    ).inc()


def record_prometheus_http_status_error(
    *,
    settings: PrometheusMetricsSettings,
    status_code: int,
    attempt: str,
) -> None:
    get_prometheus_counter(_PROM_REQUEST_ERRORS_NAME).labels(
        method=settings.target,
        status_code=str(status_code),
        error_type="http_status",
        attempt=attempt,
    ).inc()
