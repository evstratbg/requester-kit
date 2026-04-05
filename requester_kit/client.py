from __future__ import annotations

import logging
import time
from http import HTTPStatus
from http.cookies import SimpleCookie
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, Optional

from httpx import AsyncClient, AsyncHTTPTransport, HTTPError, Request, Response
from pydantic import ValidationError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_incrementing

from requester_kit.metrics import (
    get_prometheus_metrics_settings,
    record_prometheus_http_error,
    record_prometheus_http_status_error,
    record_prometheus_request_start,
    record_prometheus_response,
)
from requester_kit.types import LoggerSettings, RequesterKitResponse, RetrySettings, T_co

if TYPE_CHECKING:
    from requester_kit import types


class RequesterKitRequestError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BaseRequesterKit:
    def __init__(  # noqa: PLR0913
        self,
        base_url: str = "",
        auth: Optional[types.RequestAuth] = None,
        params: Optional[types.RequestParams] = None,
        headers: Optional[types.RequestHeaders] = None,
        cookies: Optional[types.RequestCookies] = None,
        timeout: Optional[float] = None,
        verify: Optional[types.RequestVerify] = None,
        retryer_settings: Optional[RetrySettings] = None,
        logger_settings: Optional[LoggerSettings] = None,
        logger: Any | None = None,
    ) -> None:
        self._retryer_settings = retryer_settings or RetrySettings()
        self._logger_settings = logger_settings or LoggerSettings()

        self._logger = logger or logging.getLogger(type(self).__name__)
        self._verify = verify
        transport_verify: types.RequestVerify = True if verify is None else verify
        self._client = AsyncClient(
            base_url=base_url,
            headers=headers,
            cookies=cookies,
            auth=auth,
            params=params,
            timeout=timeout,
            transport=AsyncHTTPTransport(retries=self._retryer_settings.retries, verify=transport_verify),
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
        response: Optional[Response] = None
        try:
            async for attempt in self._retryer:  # pragma: no branch
                with attempt:
                    response = await self._send_request(
                        request=request,
                        attempt_number=attempt.retry_state.attempt_number,
                    )
        except RequesterKitRequestError as exc:
            return RequesterKitResponse(
                status_code=exc.status_code,
                is_ok=False,
                error_msg=str(exc),
            )
        if response is None:
            error_msg = "Request failed without response object"
            self._logger.error(error_msg)
            return RequesterKitResponse(
                status_code=None,
                is_ok=False,
                error_msg=error_msg,
            )

        if not response_model:
            return RequesterKitResponse(
                status_code=response.status_code,
                is_ok=True,
                raw_data=response.content,
                headers=self._extract_response_headers(response),
                cookies=self._extract_response_cookies(response),
            )

        try:
            return RequesterKitResponse(
                status_code=response.status_code,
                is_ok=True,
                parsed_data=response_model.model_validate(response.json()),
                raw_data=response.content,
                headers=self._extract_response_headers(response),
                cookies=self._extract_response_cookies(response),
            )
        except (ValidationError, JSONDecodeError) as exc:
            self._logger.error("Unexpected response with error: %s", exc)
            return RequesterKitResponse(
                status_code=response.status_code,
                is_ok=False,
                error_msg=str(exc),
                raw_data=response.content,
                headers=self._extract_response_headers(response),
                cookies=self._extract_response_cookies(response),
            )

    async def _send_request(
        self,
        request: Request,
        attempt_number: int = 1,
    ) -> Response:
        start_time = time.perf_counter()
        metric_label = self._resolve_metric_target(request)
        self._log_request(request, metric_label)
        attempt_label = str(attempt_number)
        metrics_settings = record_prometheus_request_start(
            attempt=attempt_label,
            request_content=request.content if hasattr(request, "_content") else b"",
        )

        response = await self._send_http_request(
            client=self._client,
            request=request,
            start_time=start_time,
            attempt_label=attempt_label,
        )

        duration = time.perf_counter() - start_time
        if metrics_settings is not None:
            record_prometheus_response(
                settings=metrics_settings,
                response=response,
                duration=duration,
                attempt=attempt_label,
            )

        self._log_response(response, duration, str(request.url), metric_label)

        if response.status_code >= HTTPStatus.BAD_REQUEST:
            if metrics_settings is not None:
                record_prometheus_http_status_error(
                    settings=metrics_settings,
                    status_code=response.status_code,
                    attempt=attempt_label,
                )
            raise RequesterKitRequestError("Bad response", response.status_code)

        return response

    async def _send_http_request(
        self,
        client: AsyncClient,
        request: Request,
        start_time: float,
        attempt_label: str,
    ) -> Response:
        try:
            return await client.send(
                request,
                auth=self._client.auth,
            )
        except HTTPError as exc:
            duration = time.perf_counter() - start_time
            metrics_settings = get_prometheus_metrics_settings()
            if metrics_settings is not None:
                record_prometheus_http_error(
                    settings=metrics_settings,
                    duration=duration,
                    attempt=attempt_label,
                )
            raise RequesterKitRequestError(str(exc)) from exc

    def _resolve_metric_target(self, request: Request) -> str:
        metrics_settings = get_prometheus_metrics_settings()
        if metrics_settings is not None:
            return metrics_settings.target
        return f"{self.__class__.__name__}.{request.method.lower()}"

    def _log_request(self, request: Request, request_target: str) -> None:
        self._logger.info(
            "request to %s started",
            request_target,
            extra={
                "event_name": "http.request.started",
                "method": request.method,
                "url": str(request.url),
                "request_target": request_target,
            },
        )

    def _log_response(
        self,
        response: Response,
        total_time: float,
        request_url: str,
        request_target: str,
    ) -> None:
        payload = {
            "event_name": "http.request.finished",
            "method": response.request.method if response.request is not None else None,
            "url": request_url,
            "status_code": response.status_code,
            "request_target": request_target,
            "total_time_seconds": round(total_time, 3),
        }

        if response.status_code < HTTPStatus.BAD_REQUEST:
            self._logger.info("request to %s ok", request_target, extra=payload)
            return

        should_log_error = (
            response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR and self._logger_settings.log_error_for_5xx
        ) or (response.status_code < HTTPStatus.INTERNAL_SERVER_ERROR and self._logger_settings.log_error_for_4xx)
        if not should_log_error:
            return

        try:
            payload["response_body"] = response.text
        except AttributeError:
            payload["response_body"] = "<unavailable>"

        self._logger.warning(
            "request to %s completed with status %s",
            request_target,
            response.status_code,
            extra=payload,
        )

    def _extract_response_headers(self, response: Response) -> dict[str, str]:
        return dict(response.headers.items())

    def _extract_response_cookies(self, response: Response) -> dict[str, str]:
        try:
            return dict(response.cookies.items())
        except RuntimeError:
            set_cookie_headers = response.headers.get_list("set-cookie")
            if not set_cookie_headers:
                return {}
            cookies = {}
            for set_cookie_header in set_cookie_headers:
                parsed = SimpleCookie()
                parsed.load(set_cookie_header)
                for key, value in parsed.items():
                    cookies[key] = value.value
            return cookies
