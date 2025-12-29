import io
import time
from collections.abc import Awaitable, Callable
from http import HTTPMethod

import httpx
import pytest
from pydantic import BaseModel, ValidationError
from pytest_mock import MockerFixture

from requester_kit.client import RequesterKit
from requester_kit.types import RetryerSettings
from tests.conftest import MockHTTPX


class HelloWorldModel(BaseModel):
    hello: str


method_parametrize = pytest.mark.parametrize(
    ("method", "method_name"),
    [
        (RequesterKit.get, HTTPMethod.GET),
        (RequesterKit.post, HTTPMethod.POST),
        (RequesterKit.put, HTTPMethod.PUT),
        (RequesterKit.delete, HTTPMethod.DELETE),
        (RequesterKit.patch, HTTPMethod.PATCH),
        (RequesterKit.head, HTTPMethod.HEAD),
    ],
)


@method_parametrize
async def test__base_async_requester__403_and_no_log_error_for_4xx__should_print_no_error_log(
    async_requester: RequesterKit,
    mock_httpx: MockHTTPX,
    method: Callable[..., Awaitable[httpx.Response]],
    method_name: HTTPMethod,
    caplog: pytest.LogCaptureFixture,
):
    async_requester._logger_settings.log_error_for_4xx = False
    mock_httpx(403)
    response = await method(async_requester, "http://localhost/hewwo")
    assert response.status_code == 403

    assert caplog.record_tuples == [
        ("RequesterKit", 20, f"Sending {method_name} request to http://localhost/hewwo"),
    ]


@method_parametrize
async def test__base_async_requester__500_and_no_log_error_for_5xx__should_print_no_error_log(
    async_requester: RequesterKit,
    mock_httpx: MockHTTPX,
    method: Callable[..., Awaitable[httpx.Response]],
    method_name: HTTPMethod,
    caplog: pytest.LogCaptureFixture,
):
    async_requester._logger_settings.log_error_for_5xx = False
    mock_httpx(500)
    response = await method(async_requester, "http://localhost/hewwo")
    assert response.status_code == 500

    assert caplog.record_tuples == [
        ("RequesterKit", 20, f"Sending {method_name} request to http://localhost/hewwo"),
    ]


@method_parametrize
async def test__base_async_requester__403_and_500__should_print_error_log(
    async_requester: RequesterKit,
    mock_httpx: MockHTTPX,
    method: Callable[..., Awaitable[httpx.Response]],
    method_name: HTTPMethod,
    caplog: pytest.LogCaptureFixture,
):
    mock_httpx(500)
    response = await method(async_requester, "http://localhost/hewwo")
    assert response.status_code == 500

    mock_httpx(403)
    response = await method(async_requester, "http://localhost/hewwo")
    assert response.status_code == 403

    assert caplog.record_tuples == [
        ("RequesterKit", 20, f"Sending {method_name} request to http://localhost/hewwo"),
        (
            "RequesterKit",
            30,
            "Response from (http://localhost/hewwo) with status_code 500",
        ),
        ("RequesterKit", 20, f"Sending {method_name} request to http://localhost/hewwo"),
        (
            "RequesterKit",
            30,
            "Response from (http://localhost/hewwo) with status_code 403",
        ),
    ]


@method_parametrize
async def test__base_async_requester__all_http_methods(
    async_requester: RequesterKit,
    method: Callable[[RequesterKit, str], Awaitable[httpx.Response]],
    method_name: HTTPMethod,
):
    response = await async_requester._send_request(
        httpx.Request(
            method=method_name,
            url="http://localhost/hewwo",
        )
    )
    assert response.status_code == 200
    assert response.request.method == method.__name__.upper() == method_name


async def test__base_async_requester__send_params(async_requester: RequesterKit):
    response = await async_requester._send_request(
        httpx.Request(method=HTTPMethod.GET, url="http://localhost/hewwo", params={"hello": "world"})
    )
    assert response.status_code == 200
    assert response.request.url == "http://localhost/hewwo?hello=world"


async def test__base_async_requester__send_files(async_requester: RequesterKit):
    response = await async_requester._send_request(
        httpx.Request(
            method=HTTPMethod.POST,
            url="http://localhost/hewwo",
            files={"hello": io.BytesIO(b"darkness my old friend")},
        )
    )
    assert response.status_code == 200

    assert b"Content-Type: application/octet-stream" in response.request.read()
    assert b'Content-Disposition: form-data; name="hello"; filename="upload"' in response.request.read()
    assert b"darkness my old friend" in response.request.read()


@pytest.mark.parametrize(
    "method",
    [
        HTTPMethod.PUT,
        HTTPMethod.POST,
        HTTPMethod.PATCH,
    ],
)
async def test__base_async_requester__send_files_as_tuple(
    async_requester: RequesterKit,
    method: HTTPMethod,
):
    response = await async_requester._send_request(
        httpx.Request(
            method=method,
            url="http://localhost/hewwo",
            files=(("hello", io.BytesIO(b"darkness my old friend")),),
        )
    )
    assert response.status_code == 200

    assert b"Content-Type: application/octet-stream" in response.request.read()
    assert b'Content-Disposition: form-data; name="hello"; filename="upload"' in response.request.read()
    assert b"darkness my old friend" in response.request.read()


async def test__base_async_requester__send_data_as_dict(
    async_requester: RequesterKit,
):
    response = await async_requester._send_request(
        httpx.Request(
            method=HTTPMethod.POST,
            url="http://localhost/hewwo",
            data={"hello": "world"},
        )
    )
    assert response.status_code == 200
    assert response.request.content == b"hello=world"


@pytest.mark.parametrize("data", [b"hewwo", "hewwo"])
async def test__base_async_requester__send_data_as_str(async_requester: RequesterKit, data: str | bytes):
    response = await async_requester._send_request(
        httpx.Request(
            method=HTTPMethod.POST,
            url="http://localhost/hewwo",
            content=data,
        )
    )
    assert response.status_code == 200
    assert response.request.content == b"hewwo"


async def test__base_async_requester__send_login_password_info(mock_httpx: MockHTTPX, mocker: MockerFixture):
    mocked_send = mock_httpx(200)
    requester = RequesterKit(auth=("MyLogin", "MyPassword"))
    response = await requester._send_request(httpx.Request(method=HTTPMethod.GET, url="http://localhost/hewwo"))
    mocked_send.assert_awaited_once_with(response.request, auth=requester._client.auth)


async def test__base_async_requester__prometheus_metrics__observes_duration(
    mock_httpx: MockHTTPX,
    mocker: MockerFixture,
):
    mock_httpx(200)
    metric = mocker.Mock()
    metric_child = mocker.Mock()
    metric.labels.return_value = metric_child
    get_metric = mocker.patch("requester_kit.client._get_prometheus_histogram", return_value=metric)
    mocker.patch("requester_kit.client.time.perf_counter", side_effect=[0.0, 1.0])
    mocker.patch(
        "requester_kit.client.RequesterKit._resolve_metric_label",
        return_value="UserInfo.get_user_info",
    )

    requester = RequesterKit(enable_prometheus_metrics=True)
    await requester.get("http://localhost/hewwo")

    get_metric.assert_called_once_with("requester_kit_request_duration_seconds")
    metric.labels.assert_called_once_with(method="UserInfo.get_user_info")
    metric_child.observe.assert_called_once_with(1.0)


async def test__base_async_requester__prometheus_metrics__label_from_subclass_method(
    mock_httpx: MockHTTPX,
    mocker: MockerFixture,
):
    mock_httpx(200)
    metric = mocker.Mock()
    metric_child = mocker.Mock()
    metric.labels.return_value = metric_child
    mocker.patch("requester_kit.client._get_prometheus_histogram", return_value=metric)
    mocker.patch("requester_kit.client.time.perf_counter", side_effect=[0.0, 1.0])

    class UserInfo(RequesterKit):
        async def get_user_info(self):
            return await self.get("http://localhost/hewwo")

    requester = UserInfo(enable_prometheus_metrics=True)
    await requester.get_user_info()

    metric.labels.assert_called_once_with(method="UserInfo.get_user_info")
    metric_child.observe.assert_called_once_with(1.0)


def test__get_prometheus_histogram__caches_and_adds_label():
    from requester_kit import client as client_module

    client_module._PROM_HISTOGRAMS.clear()
    histogram = client_module._get_prometheus_histogram("requester_kit_request_duration_seconds_test")
    cached = client_module._get_prometheus_histogram("requester_kit_request_duration_seconds_test")

    assert histogram is cached
    assert "method" in histogram._labelnames


def test__get_prometheus_histogram__missing_dependency_raises(mocker: MockerFixture):
    from requester_kit import client as client_module

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "prometheus_client":
            raise ImportError("no prometheus")
        return real_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=fake_import)

    with pytest.raises(RuntimeError, match="prometheus_client is required"):
        client_module._get_prometheus_histogram("requester_kit_request_duration_seconds_missing")


async def test__base_async_requester__prometheus_metrics__label_fallback_when_no_frame(
    mock_httpx: MockHTTPX,
    mocker: MockerFixture,
):
    mock_httpx(200)
    metric = mocker.Mock()
    metric_child = mocker.Mock()
    metric.labels.return_value = metric_child
    mocker.patch("requester_kit.client._get_prometheus_histogram", return_value=metric)
    mocker.patch("requester_kit.client.time.perf_counter", side_effect=[0.0, 1.0])
    mocker.patch("requester_kit.client.inspect.currentframe", return_value=None)

    requester = RequesterKit(enable_prometheus_metrics=True)
    await requester.get("http://localhost/hewwo")

    metric.labels.assert_called_once_with(method="RequesterKit.get")
    metric_child.observe.assert_called_once_with(1.0)


async def test__base_async_requester__prometheus_metrics__label_fallback_when_no_external_method(
    mock_httpx: MockHTTPX,
    mocker: MockerFixture,
):
    mock_httpx(200)
    metric = mocker.Mock()
    metric_child = mocker.Mock()
    metric.labels.return_value = metric_child
    mocker.patch("requester_kit.client._get_prometheus_histogram", return_value=metric)
    mocker.patch("requester_kit.client.time.perf_counter", side_effect=[0.0, 1.0])

    requester = RequesterKit(enable_prometheus_metrics=True)
    await requester.get("http://localhost/hewwo")

    metric.labels.assert_called_once_with(method="RequesterKit.get")
    metric_child.observe.assert_called_once_with(1.0)


async def test__base_async_requester__exception_during_send__turned_into_500(
    async_requester: RequesterKit,
    mocker: MockerFixture,
):
    mocker.patch.object(httpx.AsyncClient, "send", side_effect=httpx.HTTPError("Such error"))
    response = await async_requester.get("http://localhost/hewwo")
    assert not response.status_code


async def test__base_async_requester__exception_during_build_request__raised(
    async_requester: RequesterKit,
):
    with pytest.raises(
        TypeError,
        match="Invalid type for url.  Expected str or httpx.URL, got <class 'bytes'>: b'not an url'",
    ):
        await async_requester.get(b"not an url")  # pyright: ignore[reportGeneralTypeIssues]


@pytest.mark.parametrize("status_code", [400, 404, 429])
async def test__base_async_requester_no_retries__on_4xx_response_code__error(
    mocker: MockerFixture,
    status_code: int,
):
    side_effect = httpx.Response(
        status_code=status_code,
        content=b'{"error": {"message": "Wow. Such error"}}',
        headers={"content-type": "application/json"},
    )
    mocked_send = mocker.patch.object(httpx.AsyncClient, "send", side_effect=[side_effect] * 6)
    response = await RequesterKit(
        retryer_settings=RetryerSettings(
            retries=5,
            delay=0,
            increment=0,
        )
    ).get("http://localhost/hewwo")
    assert response.status_code == status_code
    assert mocked_send.call_count == 1  # no retries for 4xx errors


@pytest.mark.parametrize("status_code", [404, 429])
async def test__base_async_requester_retry__on_4xx_response_code_when_it_is_explicitly_set__error(
    mocker: MockerFixture,
    status_code: int,
):
    error_response = httpx.Response(
        status_code=status_code,
        content=b'{"error": {"message": "Wow. Such error"}}',
        headers={"content-type": "application/json"},
    )
    side_effect = [error_response] * 6
    mocked_send = mocker.patch.object(
        httpx.AsyncClient,
        "send",
        side_effect=side_effect,
    )
    response = await RequesterKit(
        retryer_settings=RetryerSettings(
            retries=5,
            delay=0,
            increment=0,
            custom_status_codes={429, 400, 404},
        )
    ).get("http://localhost/hewwo")
    assert response.status_code == status_code
    assert mocked_send.call_count == 6  # 1 call + 5 retries


@pytest.mark.parametrize("status_code", [200, 201])
async def test__base_async_requester_no_retry__on_2xx_response_code_even_when_it_is_explicitly_set__success(
    mocker: MockerFixture,
    status_code: int,
):
    success_response = httpx.Response(
        status_code=status_code,
        content=b'{"hello": "world"}',
        headers={"content-type": "application/json"},
    )
    mocked_send = mocker.patch.object(
        httpx.AsyncClient,
        "send",
        side_effect=[success_response],
    )
    async_requester = RequesterKit(
        retryer_settings=RetryerSettings(
            retries=5,
            delay=0,
            increment=0,
        )
    )
    async_requester._retryer_settings.custom_status_codes = {200, 201}
    response = await async_requester.get("http://localhost/hewwo", response_model=HelloWorldModel)

    assert mocked_send.call_count == 1  # no retries
    assert response.parsed_data.hello == "world"


async def test_base_async_requester_wrong_status_codes():
    with pytest.raises(ValidationError) as exc_info:
        RequesterKit(
            retryer_settings=RetryerSettings(
                retries=5,
                delay=0,
                increment=0,
                custom_status_codes={200, 201},
            )
        )

    assert exc_info.value.errors()[0]["loc"][0] == "custom_status_codes"


@pytest.mark.parametrize("status_code", [400, 404, 429])
async def test__base_async_requester_retry__on_4xx_response_code_when_it_is_explicitly_set_works_for__error(
    mocker: MockerFixture,
    status_code: int,
):
    error_response = httpx.Response(
        status_code=status_code,
        content=b'{"error": {"message": "Wow. Such error"}}',
        headers={"content-type": "application/json"},
    )
    successful_response = httpx.Response(
        status_code=201,
        content=b'{"hello": "world"}',
        headers={"content-type": "application/json"},
    )

    mocked_send = mocker.patch.object(
        httpx.AsyncClient,
        "send",
        side_effect=[error_response, error_response, successful_response],
    )
    response = await RequesterKit(
        retryer_settings=RetryerSettings(
            retries=10,
            delay=0,
            increment=0,
            custom_status_codes={429, 400, 404},
        )
    ).get("http://localhost/hewwo", response_model=HelloWorldModel)
    # side_effect is set to 2 errors and 1 success after that, so only 2 retries proceeded, so 3 calls
    assert mocked_send.call_count == 3
    assert response.parsed_data.hello == "world"


async def test__base_async_requester_retries__two_exceptions_one_ok__success(mocker: MockerFixture):
    mocked_send = mocker.patch.object(
        httpx.AsyncClient,
        "send",
        side_effect=[
            httpx.HTTPError("Such error"),
            httpx.Response(
                request=httpx.Request(method=HTTPMethod.POST, url="http://localhost/hewwo"),
                status_code=500,
                content=b'{"Such":"error"}',
            ),
            httpx.Response(
                request=httpx.Request(method=HTTPMethod.POST, url="http://localhost/hewwo"),
                status_code=201,
                content=b'{"hello": "world"}',
                headers={"content-type": "application/json"},
            ),
        ],
    )
    response = await RequesterKit(
        retryer_settings=RetryerSettings(
            retries=2,
            delay=0,
            increment=0,
        )
    ).get("http://localhost/hewwo", response_model=HelloWorldModel)
    assert response.status_code == 201
    assert response.is_ok
    assert response.parsed_data, response.parsed_data.model_dump() == {"hello": "world"}
    assert mocked_send.call_count == 3


async def test__base_async_requester_backoff_factor(mock_httpx: MockHTTPX):
    mock_httpx(500, b'{"Such":"error"}')
    start = time.time()
    response = await RequesterKit(
        retryer_settings=RetryerSettings(
            retries=3,
            delay=0.2,
            increment=0.1,
        )
    ).get("http://localhost/hewwo")
    assert response.status_code == 500
    assert not response.is_ok
    assert abs((time.time() - start) - (0.2 + 0.3 + 0.4)) < 0.06


async def test_no_data_validation(mock_httpx: MockHTTPX):
    mock_httpx(200, b'{"bla":"blabla"}')
    response = await RequesterKit(
        retryer_settings=RetryerSettings(
            retries=3,
            delay=0.2,
            increment=0.1,
        )
    ).get("http://localhost/hewwo")
    assert response.status_code == 200
    assert response.is_ok
    assert not response.parsed_data


async def test_invalid_data_response(mock_httpx: MockHTTPX):
    mock_httpx(200, b'{"bla":"blabla"}')
    response = await RequesterKit(
        retryer_settings=RetryerSettings(
            retries=3,
            delay=0.2,
            increment=0.1,
        )
    ).get("http://localhost/hewwo", response_model=HelloWorldModel)

    assert response.status_code == 200
    assert not response.is_ok


async def test_async_requester_not_retry_unexpected_error():
    assert RequesterKit()._need_to_retry(ValueError) is False
