# https://github.com/AlexxIT/YandexStation/blob/master/custom_components/yandex_station/core/yandex_session.py
from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any, ParamSpec

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

if TYPE_CHECKING:
    from aiohttp.abc import AbstractCookieJar

P = ParamSpec("P")


class ClientRequest(Request):  # type: ignore[misc]
    def __init__(
        self,
        client_session: ClientSession,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.__client_session = client_session

    @property
    def _cookie_jar(self) -> AbstractCookieJar:
        return self.__client_session.cookie_jar

    async def _request_wrapper(  # noqa: C901
        self, method: str, url: str, **kwargs: dict[str, Any]
    ) -> bytes:
        if "headers" not in kwargs:
            kwargs["headers"] = {}

        kwargs["headers"]["User-Agent"] = USER_AGENT

        kwargs.pop("timeout", None)
        try:
            async with self.__client_session.request(
                method, url, **kwargs
            ) as _resp:
                resp = _resp
                content = await resp.content.read()
        except asyncio.TimeoutError as e:
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


class YandexClientRequest(ClientRequest):
    def __init__(
        self, client_session: ClientSession, *args: P.args, **kwargs: P.kwargs
    ) -> None:
        super().__init__(client_session, *args, **kwargs)
        self._auth_payload: dict[str, str] = {}

    async def get_qr(self) -> str:
        # step 1: csrf_token
        response_csrf_token = await self._request_wrapper(
            "GET",
            "https://passport.yandex.ru/am?app_platform=android",
        )

        re_result = re.search(
            rb'"csrf_token" value="([^"]+)"',
            response_csrf_token,
        )
        if re_result is None:
            raise RuntimeError("CSRF token not found!")
        self._auth_payload = {"csrf_token": re_result[1].decode()}

        # step 2: track_id
        response_track_id = await self._request_wrapper(
            "POST",
            "https://passport.yandex.ru/registration-validations/auth/password/submit",
            data={
                **self._auth_payload,
                "retpath": "https://passport.yandex.ru/profile",
                "with_code": 1,
            },
        )
        response_json = json.loads(response_track_id)
        if response_json["status"] != "ok":
            raise RuntimeError(f"Error login {response_json['errors']}")
        self._auth_payload = {
            "csrf_token": response_json["csrf_token"],
            "track_id": response_json["track_id"],
        }

        return f"https://passport.yandex.ru/auth/magic/code/?track_id={self._auth_payload["track_id"]}"

    async def login_qr(self) -> str | None:
        response = await self._request_wrapper(
            "POST",
            "https://passport.yandex.ru/auth/new/magic/status/",
            data=self._auth_payload,
        )
        response_json = json.loads(response)
        if not response_json:
            return None
        # resp={} if no auth yet
        if response_json["status"] != "ok":
            raise RuntimeError(f"Error login {response_json['errors']}")
        return await self.login_cookies()

    async def login_cookies(self) -> str:
        cookies = "; ".join(
            [
                f"{c.key}={c.value}"
                for c in self._cookie_jar
                if c["domain"].endswith("yandex.ru")
            ],
        )
        # https://gist.github.com/superdima05/04601c6b15d5eeb1c376535579d08a99
        response = await self._request_wrapper(
            "POST",
            "https://mobileproxy.passport.yandex.net/1/bundle/oauth/token_by_sessionid",
            data={
                "client_id": "c0ebe342af7d48fbbbfcf2d2eedb8f9e",
                "client_secret": "ad0a908f0aa341a182a37ecd75bc319e",
            },
            headers={
                "Ya-Client-Host": "passport.yandex.ru",
                "Ya-Client-Cookie": cookies,
            },
        )
        response_json = json.loads(response)
        token: str = response_json["access_token"]
        return token

    async def get_music_token(self, x_token: str) -> dict[str, str]:
        payload = {
            # Thanks to https://github.com/MarshalX/yandex-music-api/
            "client_secret": "53bc75238f0c4d08a118e51fe9203300",
            "client_id": "23cabbbdc6cd418abb4b39c32c41195d",
            "grant_type": "x-token",
            "access_token": x_token,
        }
        response = await self._request_wrapper(
            "POST",
            "https://oauth.mobile.yandex.net/1/token",
            data=payload,
        )
        response_json: dict[str, str] = json.loads(response)
        return response_json
