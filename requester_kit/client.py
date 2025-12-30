from __future__ import annotations

import inspect
import logging
import time
from http import HTTPStatus
from importlib import import_module
from json import JSONDecodeError
from typing import TYPE_CHECKING, Optional, cast

from httpx import AsyncClient, AsyncHTTPTransport, HTTPError, Request, Response
from pydantic import ValidationError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_incrementing

from requester_kit.types import LoggerSettings, RequesterKitResponse, RetrySettings, T_co

if TYPE_CHECKING:
    from prometheus_client import Counter, Histogram

    from requester_kit import types

_PROM_HISTOGRAMS: dict[str, Histogram] = {}
_PROM_COUNTERS: dict[str, Counter] = {}
_PROM_REQUEST_DURATION_NAME = "requester_kit_request_duration_seconds"
_PROM_REQUEST_ERRORS_NAME = "requester_kit_request_errors_total"
_PROM_REQUEST_SIZE_NAME = "requester_kit_request_payload_bytes"
_PROM_RESPONSE_SIZE_NAME = "requester_kit_response_bytes"


def _get_prometheus_histogram(name: str) -> Histogram:
    try:
        histogram = import_module("prometheus_client").Histogram
    except ImportError as exc:
        raise RuntimeError("prometheus_client is required when enable_prometheus_metrics=True") from exc

    if name not in _PROM_HISTOGRAMS:
        _PROM_HISTOGRAMS[name] = histogram(
            name,
            "HTTP request duration in seconds",
            labelnames=("method", "status_code", "status_class", "attempt"),
        )
    return _PROM_HISTOGRAMS[name]


def _get_prometheus_counter(name: str) -> Counter:
    try:
        counter = import_module("prometheus_client").Counter
    except ImportError as exc:
        raise RuntimeError("prometheus_client is required when enable_prometheus_metrics=True") from exc

    if name not in _PROM_COUNTERS:
        _PROM_COUNTERS[name] = counter(
            name,
            "Total number of HTTP request errors",
            labelnames=("method", "status_code", "error_type", "attempt"),
        )
    return _PROM_COUNTERS[name]


def _get_prometheus_size_histogram(name: str) -> Histogram:
    return _get_prometheus_histogram(name)


class RequesterKitRequestError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BaseRequesterKit:
    def __init__(
        self,
        base_url: str = "",
        auth: Optional[types.RequestAuth] = None,
        params: Optional[types.RequestParams] = None,
        headers: Optional[types.RequestHeaders] = None,
        cookies: Optional[types.RequestCookies] = None,
        timeout: Optional[float] = None,
        retryer_settings: Optional[RetrySettings] = None,
        logger_settings: Optional[LoggerSettings] = None,
        *,
        enable_prometheus_metrics: bool = False,
    ) -> None:
        self._retryer_settings = retryer_settings or RetrySettings()
        self._logger_settings = logger_settings or LoggerSettings()
        self._logger = logging.getLogger(type(self).__name__)
        self._enable_prometheus_metrics = enable_prometheus_metrics
        self._client = AsyncClient(
            base_url=base_url,
            headers=headers,
            cookies=cookies,
            auth=auth,
            params=params,
            timeout=timeout,
            transport=AsyncHTTPTransport(retries=self._retryer_settings.retries),
        )
        self._retryer = AsyncRetrying(
            stop=stop_after_attempt(self._retryer_settings.retries + 1),
            wait=wait_incrementing(start=self._retryer_settings.delay, increment=self._retryer_settings.increment),
            retry=retry_if_exception(self._need_to_retry),
            reraise=True,
        )

    async def get(
        self,
        url: str,
        response_model: Optional[type[T_co]] = None,
        headers: Optional[types.RequestHeaders] = None,
        params: Optional[types.RequestParams] = None,
    ) -> RequesterKitResponse[T_co]:
        return await self._make_request(
            method="GET",
            url=url,
            headers=headers,
            params=params,
            response_model=response_model,
        )

    async def post(
        self,
        url: str,
        response_model: Optional[type[T_co]] = None,
        headers: Optional[types.RequestHeaders] = None,
        json: Optional[types.RequestJson] = None,
        data: Optional[types.RequestData] = None,
        content: Optional[types.RequestContent] = None,
        files: Optional[types.RequestFiles] = None,
        params: Optional[types.RequestParams] = None,
    ) -> RequesterKitResponse[T_co]:
        return await self._make_request(
            method="POST",
            url=url,
            headers=headers,
            json=json,
            data=data,
            content=content,
            files=files,
            params=params,
            response_model=response_model,
        )

    async def put(
        self,
        url: str,
        response_model: Optional[type[T_co]] = None,
        headers: Optional[types.RequestHeaders] = None,
        json: Optional[types.RequestJson] = None,
        data: Optional[types.RequestData] = None,
        content: Optional[types.RequestContent] = None,
        files: Optional[types.RequestFiles] = None,
        params: Optional[types.RequestParams] = None,
    ) -> RequesterKitResponse[T_co]:
        return await self._make_request(
            method="PUT",
            url=url,
            headers=headers,
            json=json,
            data=data,
            content=content,
            files=files,
            params=params,
            response_model=response_model,
        )

    async def patch(
        self,
        url: str,
        response_model: Optional[type[T_co]] = None,
        headers: Optional[types.RequestHeaders] = None,
        json: Optional[types.RequestJson] = None,
        data: Optional[types.RequestData] = None,
        content: Optional[types.RequestContent] = None,
        files: Optional[types.RequestFiles] = None,
        params: Optional[types.RequestParams] = None,
    ) -> RequesterKitResponse[T_co]:
        return await self._make_request(
            method="PATCH",
            url=url,
            headers=headers,
            json=json,
            data=data,
            content=content,
            files=files,
            params=params,
            response_model=response_model,
        )

    async def head(
        self,
        url: str,
        response_model: Optional[type[T_co]] = None,
        headers: Optional[types.RequestHeaders] = None,
        params: Optional[types.RequestParams] = None,
    ) -> RequesterKitResponse[T_co]:
        return await self._make_request(
            method="HEAD",
            url=url,
            headers=headers,
            params=params,
            response_model=response_model,
        )

    async def delete(
        self,
        url: str,
        response_model: Optional[type[T_co]] = None,
        headers: Optional[types.RequestHeaders] = None,
        params: Optional[types.RequestParams] = None,
    ) -> RequesterKitResponse[T_co]:
        return await self._make_request(
            method="DELETE",
            url=url,
            headers=headers,
            params=params,
            response_model=response_model,
        )

    def _need_to_retry(self, exc: BaseException) -> bool:
        if not isinstance(exc, RequesterKitRequestError):
            self._logger.error("Received unexpected exception: %s", exc)
            return False
        if not exc.status_code:
            return True
        return (
            exc.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR
            or exc.status_code in self._retryer_settings.custom_status_codes
        )

    async def _make_request(
        self,
        method: str,
        url: str,
        response_model: Optional[type[T_co]] = None,
        headers: Optional[types.RequestHeaders] = None,
        json: Optional[types.RequestJson] = None,
        data: Optional[types.RequestData] = None,
        content: Optional[types.RequestContent] = None,
        files: Optional[types.RequestFiles] = None,
        params: Optional[types.RequestParams] = None,
    ) -> RequesterKitResponse[T_co]:
        request = self._client.build_request(
            method=method,
            url=url,
            headers=headers,
            json=json,
            data=data,
            files=files,
            params=params,
            content=content,
        )
        try:
            async for attempt in self._retryer:
                with attempt:
                    response = await self._send_request(request, attempt.retry_state.attempt_number)
        except RequesterKitRequestError as exc:
            return RequesterKitResponse(
                status_code=exc.status_code,
                is_ok=False,
            )

        if not response_model:
            return RequesterKitResponse(
                status_code=response.status_code,
                is_ok=True,
                raw_data=response.content,
            )

        try:
            return RequesterKitResponse(
                status_code=response.status_code,
                is_ok=True,
                parsed_data=response_model.model_validate(response.json()),
                raw_data=response.content,
            )
        except (ValidationError, JSONDecodeError) as exc:
            self._logger.error("Unexpected response with error: %s", exc)
            return RequesterKitResponse(
                status_code=response.status_code,
                is_ok=False,
                raw_data=response.content,
            )

    async def _send_request(self, request: Request, attempt_number: int = 1) -> Response:
        self._log_request(request)

        start_time = time.perf_counter()
        metric = None
        error_counter = None
        request_size_metric = None
        response_size_metric = None
        if self._enable_prometheus_metrics:
            metric = _get_prometheus_histogram(_PROM_REQUEST_DURATION_NAME)
            error_counter = _get_prometheus_counter(_PROM_REQUEST_ERRORS_NAME)
            request_size_metric = _get_prometheus_size_histogram(_PROM_REQUEST_SIZE_NAME)
            response_size_metric = _get_prometheus_size_histogram(_PROM_RESPONSE_SIZE_NAME)
            metric_label = self._resolve_metric_label(request)
            attempt_label = str(attempt_number)
            request_size_metric.labels(
                method=metric_label,
                status_code="request",
                status_class="request",
                attempt=attempt_label,
            ).observe(len(request.content or b""))

        try:
            response = await self._client.send(
                request,
                auth=self._client.auth,
            )
        except HTTPError as exc:
            duration = time.perf_counter() - start_time
            if metric is not None:
                metric.labels(
                    method=metric_label,
                    status_code="exception",
                    status_class="error",
                    attempt=attempt_label,
                ).observe(duration)
            if error_counter is not None:
                error_counter.labels(
                    method=metric_label,
                    status_code="exception",
                    error_type="http_error",
                    attempt=attempt_label,
                ).inc()
            raise RequesterKitRequestError(str(exc)) from exc

        duration = time.perf_counter() - start_time
        if metric is not None:
            metric.labels(
                method=metric_label,
                status_code=str(response.status_code),
                status_class=f"{response.status_code // 100}xx",
                attempt=attempt_label,
            ).observe(duration)
            response_size_metric = cast("Histogram", response_size_metric)
            response_size_metric.labels(
                method=metric_label,
                status_code=str(response.status_code),
                status_class=f"{response.status_code // 100}xx",
                attempt=attempt_label,
            ).observe(len(response.content or b""))

        self._log_response(response, duration, str(request.url))

        if response.status_code >= HTTPStatus.BAD_REQUEST:
            if error_counter is not None:
                error_counter.labels(
                    method=metric_label,
                    status_code=str(response.status_code),
                    error_type="http_status",
                    attempt=attempt_label,
                ).inc()
            raise RequesterKitRequestError("Bad response", response.status_code)

        return response

    def _resolve_metric_label(self, request: Request) -> str:
        frame = inspect.currentframe()
        if frame is None:
            return f"{self.__class__.__name__}.{request.method.lower()}"
        try:
            frame = frame.f_back
            while frame:
                frame_self = frame.f_locals.get("self")
                if isinstance(frame_self, BaseRequesterKit):
                    method_name = frame.f_code.co_name
                    base_method = getattr(BaseRequesterKit, method_name, None)
                    base_code = getattr(base_method, "__code__", None) if base_method else None
                    if base_code is frame.f_code:
                        frame = frame.f_back
                        continue
                    return f"{type(frame_self).__name__}.{method_name}"
                frame = frame.f_back
        finally:
            del frame
        return f"{self.__class__.__name__}.{request.method.lower()}"

    def _log_request(self, request: Request) -> None:
        self._logger.info("Sending %s request to %s", request.method, request.url)

    def _log_response(
        self,
        response: Response,
        total_time: float,
        request_url: str,
    ) -> None:
        msg = f"Response from ({request_url}) with status_code {response.status_code}"
        extra = {
            "status_code": response.status_code,
            "url": request_url,
            "total_time": total_time,
        }

        if response.status_code < HTTPStatus.BAD_REQUEST:
            self._logger.info(msg, extra=extra)
            return

        if (response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR and self._logger_settings.log_error_for_5xx) or (
            response.status_code < HTTPStatus.INTERNAL_SERVER_ERROR and self._logger_settings.log_error_for_4xx
        ):
            extra["body"] = response.content.decode()
            self._logger.warning(msg, extra=extra)
            return
