# requester-kit

Async HTTP connector toolkit with retries, typed responses, and decorator-driven Prometheus metrics.

## Install

```bash
uv add requester-kit
```

Metrics extra:

```bash
uv add --extra metrics requester-kit
```

## Quickstart

```python
from uuid import UUID

from pydantic import BaseModel

from requester_kit import BaseRequesterKit, RequesterKitResponse, prometheus_metrics


class UserInfo(BaseModel):
    id: UUID
    name: str


class UsersAPI(BaseRequesterKit):
    @prometheus_metrics()
    async def get_user_info(self, user_id: UUID) -> RequesterKitResponse[UserInfo]:
        return await self.get(f"/users/{user_id}", response_model=UserInfo)


async def run():
    users = UsersAPI(base_url="https://api.example.com")
    response = await users.get_user_info(UUID("00000000-0000-0000-0000-000000000000"))
    if response.is_ok and response.parsed_data:
        print(response.parsed_data.name)
```

## Retries and logging

```python
from requester_kit import BaseRequesterKit
from requester_kit.types import LoggerSettings, RetrySettings

client = BaseRequesterKit(
    base_url="https://api.example.com",
    retryer_settings=RetrySettings(
        retries=3,
        delay=0.2,
        increment=0.1,
        custom_status_codes={429},
    ),
    logger_settings=LoggerSettings(
        log_error_for_4xx=False,
        log_error_for_5xx=True,
    ),
)
```

## Prometheus metrics

Use `@prometheus_metrics(...)` on external requester methods to enable Prometheus collection for that operation.
Each HTTP call records a Histogram named `requester_kit_request_duration_seconds` with labels:
`method` (for example `UsersAPI.get_user_info`), `status_code`, `status_class`, and `attempt`.
This provides request count and timing via the standard `_count` and `_sum` series.

Errors are counted in `requester_kit_request_errors_total` with labels:
`method`, `status_code`, `error_type` (`http_status` or `http_error`), and `attempt`.

Payload sizes are recorded in Histograms:
`requester_kit_request_payload_bytes` and `requester_kit_response_bytes` with the same labels as
`requester_kit_request_duration_seconds`. They are optional and are only collected when
`count_payload_bytes=True` or `count_response_bytes=True` is set on the decorator.

If `name` is omitted, the method label is always `ClassName.method_name`. You can override only the method part:

```python
from requester_kit import BaseRequesterKit, prometheus_metrics

class UsersAPI(BaseRequesterKit):
    @prometheus_metrics()
    async def get_user_info(self, user_id: str):
        return await self.get(f"/users/{user_id}")

    @prometheus_metrics("load_profile", count_response_bytes=True)
    async def get_profile(self, user_id: str):
        return await self.get(f"/profiles/{user_id}")

    @prometheus_metrics(count_payload_bytes=True)
    async def create_user(self, payload: dict):
        return await self.post("/users", json=payload)
```

If a method is not decorated, Prometheus metrics are not recorded for it.

Expose metrics in FastAPI:

```python
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

app = FastAPI()

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

Expose metrics using `prometheus-fastapi-instrumentator`:

```python
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

Why this works: `BaseRequesterKit` writes metrics to the default Prometheus registry, and both `generate_latest()`
and `prometheus-fastapi-instrumentator` expose that same registry, so your HTTP client metrics appear alongside
your app metrics on `/metrics`.

## Release

1. Update the package version in `pyproject.toml`.
2. Add a new entry to `CHANGELOG.md` with the release date and notable changes.
3. Run checks:

```bash
uv run pytest -q
uv run ruff check requester_kit tests
```

4. Commit the changes and create a git tag matching the version:

```bash
git tag v0.5.4
git push origin main --tags
```

5. Build and publish the package:

```bash
uv build
uv publish
```

If you also keep `uv.lock` in sync for releases, refresh it before tagging with `uv lock`.

## Testing

Use `pytest-mock` to stub out connector methods:

```python
from unittest import mock

from pydantic import BaseModel

from requester_kit import BaseRequesterKit
from requester_kit.types import RequesterKitResponse


class UserInfo(BaseModel):
    id: str
    name: str


class UsersAPI(BaseRequesterKit):
    async def get_user_info(self, user_id: str) -> RequesterKitResponse[UserInfo]:
        return await self.get(f"/users/{user_id}", response_model=UserInfo)


async def test_get_user_info(mocker):
    mocker.patch.object(
        UsersAPI,
        "get_user_info",
        new=mock.AsyncMock(
            return_value=RequesterKitResponse(
                status_code=200,
                is_ok=True,
                parsed_data=UserInfo(id="1", name="Ada"),
            )
        ),
    )

    client = UsersAPI(base_url="https://api.example.com")
    response = await client.get_user_info("1")

    assert response.is_ok
```
