from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from unittest import mock

import httpx
import pytest
from pytest_fixture_classes import fixture_class

from requester_kit.client import BaseRequesterKit

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytest_mock import MockerFixture

BASE_DIR = Path(__file__).parent.parent


@fixture_class(name="mock_httpx")
class MockHTTPX:
    mocker: MockerFixture

    def __call__(
        self,
        status_code: int,
        content: bytes = b"{}",
        headers: Optional[Mapping[str, Any]] = None,
    ) -> mock.AsyncMock:
        async def mocked_send(
            request: httpx.Request,
            *,
            stream: bool = False,
            auth: None = None,
            follow_redirects: None = None,
        ) -> httpx.Response:
            return httpx.Response(
                status_code,
                request=request,
                content=content,
                headers=headers or {"content-type": "application/json"},
            )

        return self.mocker.patch.object(httpx.AsyncClient, "send", mock.AsyncMock(wraps=mocked_send))


@pytest.fixture
def async_requester(mock_httpx: MockHTTPX) -> BaseRequesterKit:
    mock_httpx(200)
    return BaseRequesterKit()
