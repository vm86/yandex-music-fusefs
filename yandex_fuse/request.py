# https://github.com/AlexxIT/YandexStation/blob/master/custom_components/yandex_station/core/yandex_session.py
from __future__ import annotations

from typing import Any

from aiohttp import ClientError, ClientSession
from yandex_music.exceptions import (  # type: ignore[import-untyped]
    BadRequestError,
    NetworkError,
    NotFoundError,
    TimedOutError,
    UnauthorizedError,
    YandexMusicError,
)
from yandex_music.utils.request_async import (  # type: ignore[import-untyped]
    USER_AGENT,
    Request,
)


class ClientRequest(Request):  # type: ignore[misc]
    def __init__(
        self,
        client_session: ClientSession,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        # Request (yandex_music, untyped) decides for itself what it takes.
        super().__init__(*args, **kwargs)
        self.__client_session = client_session

    async def _request_wrapper(  # noqa: C901
        self,
        method: str,
        url: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> bytes:
        if "headers" not in kwargs:
            kwargs["headers"] = {}

        kwargs["headers"]["User-Agent"] = USER_AGENT

        # api.music.yandex.net/rotor/station/*/feedback and .../settings3
        # only accept a JSON body; yandex_music sends form-data there
        # (data=dict), which makes the server reply 400 "condition is not met".
        data = kwargs.get("data")
        if "/rotor/station/" in url and isinstance(data, dict):
            kwargs["json"] = kwargs.pop("data")

        kwargs.pop("timeout", None)
        try:
            async with self.__client_session.request(
                method, url, **kwargs
            ) as _resp:
                resp = _resp
                content = await resp.content.read()
        except TimeoutError as e:
            raise TimedOutError from e
        except ClientError as e:
            raise NetworkError(e) from e

        if resp.ok:
            return content

        message = "Unknown error"
        try:
            parse = self._parse(content)
            if parse:
                message = parse.get_error()
        except YandexMusicError:
            message = "Unknown HTTPError"

        if resp.status in (401, 403):
            raise UnauthorizedError(message)
        if resp.status == 400:  # noqa: PLR2004
            raise BadRequestError(message)
        if resp.status == 404:  # noqa: PLR2004
            raise NotFoundError(message)
        if resp.status in (409, 413):
            raise NetworkError(message)

        if resp.status == 502:  # noqa: PLR2004
            raise NetworkError("Bad Gateway")

        raise NetworkError(f"{message} ({resp.status}): {content!r}")
