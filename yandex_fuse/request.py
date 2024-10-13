import asyncio
from typing import Any

from aiohttp import ClientError, ClientSession
from yandex_music.exceptions import (
    BadRequestError,
    NetworkError,
    NotFoundError,
    TimedOutError,
    UnauthorizedError,
    YandexMusicError,
)
from yandex_music.utils.request_async import (
    USER_AGENT,
    Request,
)


class ClientRequest(Request):
    def __init__(self, client_session: ClientSession, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__client_session = client_session

    async def _request_wrapper(self, *args: Any, **kwargs: Any) -> bytes:  # noqa: C901
        if "headers" not in kwargs:
            kwargs["headers"] = {}

        kwargs["headers"]["User-Agent"] = USER_AGENT

        kwargs.pop("timeout")
        try:
            async with self.__client_session.request(*args, **kwargs) as _resp:
                resp = _resp
                content = await resp.content.read()
        except asyncio.TimeoutError as e:
            raise TimedOutError from e
        except ClientError as e:
            raise NetworkError(e) from e

        if 200 <= resp.status <= 299:
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
        if resp.status == 400:
            raise BadRequestError(message)
        if resp.status == 404:
            raise NotFoundError(message)
        if resp.status in (409, 413):
            raise NetworkError(message)

        if resp.status == 502:
            raise NetworkError("Bad Gateway")

        raise NetworkError(f"{message} ({resp.status}): {content}")
