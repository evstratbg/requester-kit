import logging
import inspect
import time
from http import HTTPMethod, HTTPStatus
from json import JSONDecodeError
from typing import TYPE_CHECKING

from httpx import AsyncClient, AsyncHTTPTransport, HTTPError, Request, Response
from pydantic import ValidationError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_incrementing

from requester_kit import types
from requester_kit.types import LoggerSettings, RequesterKitResponse, RetryerSettings, T_co

if TYPE_CHECKING:
    from prometheus_client import Histogram

_PROM_HISTOGRAMS: dict[str, "Histogram"] = {}
_PROM_REQUEST_DURATION_NAME = "requester_kit_request_duration_seconds"


def _get_prometheus_histogram(name: str) -> "Histogram":
    try:
        from prometheus_client import Histogram
    except ImportError as exc:
        raise RuntimeError("prometheus_client is required when enable_prometheus_metrics=True") from exc

    if name not in _PROM_HISTOGRAMS:
        _PROM_HISTOGRAMS[name] = Histogram(
            name,
            "HTTP request duration in seconds",
            labelnames=("method",),
        )
    return _PROM_HISTOGRAMS[name]


class RequesterKitRequestError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RequesterKit:
    def __init__(
        self,
        base_url: str = "",
        auth: types.RequestAuth | None = None,
        params: types.RequestParams | None = None,
        headers: types.RequestHeaders | None = None,
        cookies: types.RequestCookies | None = None,
        timeout: float | None = None,
        retryer_settings: RetryerSettings | None = None,
        logger_settings: LoggerSettings | None = None,
        enable_prometheus_metrics: bool = False,
    ) -> None:
        self._retryer_settings = retryer_settings or RetryerSettings()
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
        response_model: type[T_co] | None = None,
        headers: types.RequestHeaders | None = None,
        params: types.RequestParams | None = None,
    ) -> RequesterKitResponse[T_co]:
        return await self._make_request(
            method=HTTPMethod.GET,
            url=url,
            headers=headers,
            params=params,
            response_model=response_model,
        )

    async def post(
        self,
        url: str,
        response_model: type[T_co] | None = None,
        headers: types.RequestHeaders | None = None,
        json: types.RequestJson | None = None,
        data: types.RequestData | None = None,
        content: types.RequestContent | None = None,
        files: types.RequestFiles | None = None,
        params: types.RequestParams | None = None,
    ) -> RequesterKitResponse[T_co]:
        return await self._make_request(
            method=HTTPMethod.POST,
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
        response_model: type[T_co] | None = None,
        headers: types.RequestHeaders | None = None,
        json: types.RequestJson | None = None,
        data: types.RequestData | None = None,
        content: types.RequestContent | None = None,
        files: types.RequestFiles | None = None,
        params: types.RequestParams | None = None,
    ) -> RequesterKitResponse[T_co]:
        return await self._make_request(
            method=HTTPMethod.PUT,
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
        response_model: type[T_co] | None = None,
        headers: types.RequestHeaders | None = None,
        json: types.RequestJson | None = None,
        data: types.RequestData | None = None,
        content: types.RequestContent | None = None,
        files: types.RequestFiles | None = None,
        params: types.RequestParams | None = None,
    ) -> RequesterKitResponse[T_co]:
        return await self._make_request(
            method=HTTPMethod.PATCH,
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
        response_model: type[T_co] | None = None,
        headers: types.RequestHeaders | None = None,
        params: types.RequestParams | None = None,
    ) -> RequesterKitResponse[T_co]:
        return await self._make_request(
            method=HTTPMethod.HEAD,
            url=url,
            headers=headers,
            params=params,
            response_model=response_model,
        )

    async def delete(
        self,
        url: str,
        response_model: type[T_co] | None = None,
        headers: types.RequestHeaders | None = None,
        params: types.RequestParams | None = None,
    ) -> RequesterKitResponse[T_co]:
        return await self._make_request(
            method=HTTPMethod.DELETE,
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
        method: HTTPMethod,
        url: str,
        response_model: type[T_co] | None = None,
        headers: types.RequestHeaders | None = None,
        json: types.RequestJson | None = None,
        data: types.RequestData | None = None,
        content: types.RequestContent | None = None,
        files: types.RequestFiles | None = None,
        params: types.RequestParams | None = None,
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
                    response = await self._send_request(request)
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

    async def _send_request(self, request: Request) -> Response:
        self._log_request(request)

        start_time = time.perf_counter()
        metric = None
        if self._enable_prometheus_metrics:
            metric = _get_prometheus_histogram(_PROM_REQUEST_DURATION_NAME)
            metric_label = self._resolve_metric_label(request)

        try:
            response = await self._client.send(
                request,
                auth=self._client.auth,
            )
        except HTTPError as exc:
            duration = time.perf_counter() - start_time
            if metric is not None:
                metric.labels(method=metric_label).observe(duration)
            raise RequesterKitRequestError(str(exc)) from exc

        duration = time.perf_counter() - start_time
        if metric is not None:
            metric.labels(method=metric_label).observe(duration)

        self._log_response(response, duration, str(request.url))

        if response.status_code >= HTTPStatus.BAD_REQUEST:
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
                if frame_self is not None and type(frame_self) is not RequesterKit:
                    return f"{type(frame_self).__name__}.{frame.f_code.co_name}"
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
