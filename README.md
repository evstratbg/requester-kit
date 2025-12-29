# requester-kit

The requester-kit is an internal package that provides an interface for the implementation of *connectors* 
which serve for internal communications between services. 

Refer to the examples of implementation below.


## Connectors quickstart

```python
import os
from datetime import datetime
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, Field

from requester_kit.base.requester import (
    DEFAULT_TIMEOUT,
    BaseConfig,
    Connector,
    RequesterKitResponse,
)


class StreetCatCreateRequest(BaseModel):
    name: str
    age: int


class StreetCatResource(BaseModel):
    id: UUID
    created_at: datetime
    updated_at: datetime
    name: str
    age: int



class StreetCatsAPI(Connector):
    """Some internal cat management service"""

    async def create(
        self, payload: StreetCatCreateRequest, timeout: float = DEFAULT_TIMEOUT
    ) -> RequesterKitResponse[StreetCatResource]:
        return await self.client.post(self.base_url, json=payload.dict(exclude_unset=True), timeout=timeout)

    async def get(self, cat_id: UUID, timeout: float = DEFAULT_TIMEOUT) -> RequesterKitResponse[StreetCatResource]:
        return await self.client.get(f"{self.base_url}/{cat_id}", timeout=timeout)


async def run():
    # this env is set at build stage, do not set it explicitly
    os.environ["CATS_SERVICE_URL"] = "https://street-cats.com"
    street_cats_service = StreetCatsAPI()
    # or for local development. base_url is used instead environ
    street_cats_service = StreetCatsAPI(base_url="https://street-cats.com")

    r = await street_cats_service.create(StreetCatCreateRequest(name="Bob", age=1))
    print(r.dataclass())  # you receive filled StreetCatResource model here

```

> Using `response.dataclass()` is the recommended way to get connector response as it utilizes type annotations which makes for safer code and nice autocompletions. If this is not supported by connector it should be rewritten to support this ASAP

## Testing

### Using own connectors in tests

To make sure that connectors to your service work as expected, it is recommended to use them in that service's unit tests. This has an added benefit that reading and writing connector calls is much easier than writing HTTP requests. So for example, in `cats` service we would like to make a simple test that creates a cat. First, we would add the following fixture to our `conftest.py`:

```python
from .connectors import StreetCatsAPI
from requester_kit.base.requester_kit.testing import get_local_connector


@pytest.fixture
def street_cats_api(individual_client: AsyncClient) -> StreetCatsAPI:
    return get_local_connector(StreetCatsAPI, individual_client)
```

And then write our test:

```python
from .schemas import StreetCatCreateRequest
from .connectors import StreetCatsAPI


async def test_create_cat(street_cats_api: StreetCatsAPI):
    response = await street_cats_api.create(StreetCatCreateRequest(name="Millie", age=2))
    assert response.status_code == status.HTTP_201_CREATED 
    
    cat = response.dataclass()
    assert cat.name == "Millie"
    assert cat.age == 2
    
```

### Mocking connectors

But our cats are not safe because the dogs service is being called in our scenarios whenever a cat gets created. So the dogs can scare our cats which is why we need to mock any usages of the dogs service.

It is recommended to mock connectors using the `requester_kit.testing.MockConnector` fixture. It is much more concise, it automatically turns connector return values into RequesterKitResponse, and it automatically adds `response_model` to that response.

Note that:

- `mock_connector`'s default status_code for `RequesterKitResponse` is `200` and default content is `None`
- `mock_connector` returns an AsyncMock so you are free to use all of its capabilities such as [side effects](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.side_effect) and ["assert_called" checks](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.side_effect)
- `mock_connector` cannot be used after `mocker.patch.object` so it is recommended to use `mock_connector` everywhere instead

```python
from requester_kit.base.requester_kit.testing import MockConnector
from .connectors import DogsAPI
from .connectors import StreetCatsAPI
from .schemas import UploadCatRequest
from litestar import status_codes


async def test_create_cat(mock_connector: MockConnector, street_cats_api: StreetCatsAPI):
    # Note tht if these calls is repeated in many tests,
    # it is recommended to put them into a common auto-executing fixture
    mock_connector(DogsAPI.bark, "No bark", status_codes.HTTP_400_BAD_REQUEST)
    mock_connector(DogsAPI.walk, {"data": ["Peaceful walk"]})

    bark_response = await DogsAPI().bark()
    assert bark_response.status_code == status_codes.HTTP_400_BAD_REQUEST
    assert bark_response.json() == "No bark"

    response = await street_cats_api.create(UploadCatRequest(name="Millie", age=2))
    assert response.status_code == status_codes.HTTP_201_CREATED 

    cat = response.dataclass()
    assert cat.name == "Millie"
    assert cat.age == 2
```

### Preventing External HTTP Requests in Tests

When testing, you might want to prevent external HTTP requests from being executed to isolate and speed up your tests.
This can be achieved using the `requester_kit.testing.ensure_no_http_requests_executed` context manager.

## Prometheus metrics

Pass `enable_prometheus_metrics=True` to `RequesterKit` to track HTTP request duration.
Each HTTP call records a Histogram named `requester_kit_request_duration_seconds` with a `method` label like
`UserInfo.get_user_info`, which provides request count and timing via the standard `_count` and `_sum` series.
Install `prometheus_client` to enable this feature.

For example:

```python
@pytest.fixture(scope="session", autouse=True)
def _ensure_no_http_requests_executed():
    with ensure_no_http_requests_executed(test_host="test"):
        yield
```

Note the `test_host` argument. We use it to allow HTTP requests to `test` host because that's usually
the host we use for our fastapi service that is being tested.
