# ruff: noqa: S101
# ruff: noqa: ANN001
# ruff: noqa: ANN201
# ruff: noqa: ANN204
# mypy: ignore-errors

from typing import ParamSpec, TypeVar
from unittest import mock

import pytest
from pytest_mock import MockerFixture

from yandex_fuse.ya_music_fs import SQLPlaylist, SQLTrack, YaMusicFS

P = ParamSpec("P")
T = TypeVar("T")


class MockResponse:
    def __init__(self, text: str, status: int) -> None:
        self._text = text
        self.status = status

    async def text(self) -> str:
        return self._text

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self


TRACK_INFO = SQLTrack(
    name="TestTrack",
    inode=10,
    track_id=10,
    codec="acc",
    bitrate=128,
    artist="test",
    title="test",
    album="test",
    year="2024",
    genre="test",
    duration_ms=100,
    playlist_id="NOT",
    quality="hq",
    size=100,
)

PLAYLIST_INFO = SQLPlaylist(
    name="TestPlaylist",
    playlist_id="NOT",
    inode=1,
    station_id="station",
    batch_id="batch",
)


@pytest.fixture(autouse="True")
def client_session_mock():
    with mock.patch("yandex_fuse.ya_music_fs.YaMusicFS._client_session") as m:
        yield m


@mock.patch("yandex_fuse.ya_music_fs.Buffer.download", mock.AsyncMock())
@mock.patch("yandex_fuse.ya_music_fs.YaMusicFS._ya_player", mock.AsyncMock())
@pytest.mark.asyncio
class TestMusicFS:
    @pytest.fixture(scope="session")
    def ya_music_fs(self) -> YaMusicFS:
        yandex_music = YaMusicFS
        yandex_music.FILE_DB = "file::memory:?cache=shared"
        return yandex_music()

    async def test_read_track_missing_returns_none(
        self,
        ya_music_fs: YaMusicFS,
        mocker: MockerFixture,
    ) -> None:
        mocker.patch.object(
            ya_music_fs, "_get_track_by_inode", return_value=None
        )

        chunk = await ya_music_fs.read_track(999, 0, 100)
        assert chunk is None

    async def test_read_track_reuses_cached_buffer(
        self,
        ya_music_fs: YaMusicFS,
        mocker: MockerFixture,
    ) -> None:
        buffer = mock.MagicMock()
        buffer.read_from = mock.AsyncMock(return_value=b"Test")
        buffer.is_downloded = False
        buffer.is_send_feedback = True

        mocker.patch.object(
            ya_music_fs, "_get_track_by_inode", return_value=TRACK_INFO
        )
        get_buffer = mocker.patch.object(
            ya_music_fs, "_get_buffer", mock.AsyncMock(return_value=buffer)
        )
        mocker.patch.object(ya_music_fs, "_track_buffers", {})

        chunk = await ya_music_fs.read_track(TRACK_INFO.inode, 0, 100)
        assert chunk == b"Test"
        chunk = await ya_music_fs.read_track(TRACK_INFO.inode, 0, 100)
        assert chunk == b"Test"

        assert get_buffer.call_count == 1

    async def test_read_track_sends_started_feedback(
        self,
        ya_music_fs: YaMusicFS,
        mocker: MockerFixture,
    ) -> None:
        buffer = mock.MagicMock()
        buffer.read_from = mock.AsyncMock(return_value=b"Test")
        buffer.is_downloded = True
        buffer.is_send_feedback = False
        buffer.total_second.return_value = 31

        mocker.patch.object(
            ya_music_fs, "_get_track_by_inode", return_value=TRACK_INFO
        )
        mocker.patch.object(
            ya_music_fs,
            "_get_playlist_by_id",
            return_value=PLAYLIST_INFO,
        )
        mocker.patch.object(
            ya_music_fs, "_track_buffers", {TRACK_INFO.inode: buffer}
        )
        feedback_track = mocker.patch.object(
            ya_music_fs._ya_player,  # noqa: SLF001
            "feedback_track",
            mock.AsyncMock(),
        )

        await ya_music_fs.read_track(TRACK_INFO.inode, 0, 4)

        feedback_track.assert_awaited_once_with(
            TRACK_INFO.track_id,
            "trackStarted",
            PLAYLIST_INFO.station_id,
            PLAYLIST_INFO.batch_id,
            0,
        )
        assert buffer.is_send_feedback is True

    async def test_read_track_sends_finished_feedback_at_eof(
        self,
        ya_music_fs: YaMusicFS,
        mocker: MockerFixture,
    ) -> None:
        buffer = mock.MagicMock()
        buffer.read_from = mock.AsyncMock(return_value=b"Test")
        buffer.is_downloded = False
        buffer.is_send_feedback = False
        buffer.total_second.return_value = 5

        mocker.patch.object(
            ya_music_fs, "_get_track_by_inode", return_value=TRACK_INFO
        )
        mocker.patch.object(
            ya_music_fs,
            "_get_playlist_by_id",
            return_value=PLAYLIST_INFO,
        )
        mocker.patch.object(
            ya_music_fs, "_track_buffers", {TRACK_INFO.inode: buffer}
        )
        feedback_track = mocker.patch.object(
            ya_music_fs._ya_player,  # noqa: SLF001
            "feedback_track",
            mock.AsyncMock(),
        )

        offset = TRACK_INFO.size - 4
        await ya_music_fs.read_track(TRACK_INFO.inode, offset, 4)

        feedback_track.assert_awaited_once_with(
            TRACK_INFO.track_id,
            "trackFinished",
            PLAYLIST_INFO.station_id,
            PLAYLIST_INFO.batch_id,
            5,
        )
        assert buffer.is_send_feedback is True

    async def test_needs_auth_without_token(
        self,
        ya_music_fs: YaMusicFS,
        mocker: MockerFixture,
    ) -> None:
        mocker.patch.object(
            ya_music_fs._ya_player,  # noqa: SLF001
            "is_init",
            False,
        )

        assert ya_music_fs.needs_auth is True

    async def test_needs_auth_with_token(
        self,
        ya_music_fs: YaMusicFS,
        mocker: MockerFixture,
    ) -> None:
        mocker.patch.object(
            ya_music_fs._ya_player,  # noqa: SLF001
            "is_init",
            True,
        )

        assert ya_music_fs.needs_auth is False

    async def test_auth_page_contains_url_and_code(
        self,
        ya_music_fs: YaMusicFS,
        mocker: MockerFixture,
    ) -> None:
        auth_url = "https://passport.yandex.ru/activate"
        mocker.patch.object(
            ya_music_fs._ya_player,  # noqa: SLF001
            "auth_url",
            auth_url,
        )
        mocker.patch.object(
            ya_music_fs._ya_player,  # noqa: SLF001
            "user_code",
            "ABCD-1234",
        )

        html = ya_music_fs.auth_page()

        assert auth_url.encode() in html
        assert b"ABCD-1234" in html

    async def test_auth_page_without_code_yet(
        self,
        ya_music_fs: YaMusicFS,
        mocker: MockerFixture,
    ) -> None:
        mocker.patch.object(
            ya_music_fs._ya_player,  # noqa: SLF001
            "auth_url",
            None,
        )
        mocker.patch.object(
            ya_music_fs._ya_player,  # noqa: SLF001
            "user_code",
            None,
        )

        html = ya_music_fs.auth_page()

        assert b"<html>" in html
