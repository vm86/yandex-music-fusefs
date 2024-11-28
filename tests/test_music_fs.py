# ruff: noqa: S101
# ruff: noqa: ARG002
# ruff: noqa: ANN001
# ruff: noqa: ANN201
# ruff: noqa: ANN202
# ruff: noqa: ANN204
# mypy: ignore-errors

from typing import ParamSpec, TypeVar
from unittest import mock

import pytest
from pytest_mock import MockerFixture

from yandex_fuse.ya_music_fs import SQLTrack, StreamReader, YaMusicFS

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

    @mock.patch("yandex_fuse.ya_music_fs.YaMusicFS._get_track_by_inode")
    @mock.patch("yandex_fuse.ya_music_fs.YaMusicFS.get_or_update_direct_link")
    async def test_open(
        self,
        mock_get_or_update_direct_link: mock.Mock,
        mock_get_track_by_inode: mock.Mock,
        ya_music_fs: YaMusicFS,
    ) -> None:
        mock_get_track_by_inode.return_value = TRACK_INFO
        file_info = await ya_music_fs.open(519, 0o664, None)
        assert file_info.fh == 1

        assert mock_get_or_update_direct_link.call_count == 1

        file_info = await ya_music_fs.open(519, 0o664, None)
        assert file_info.fh == 2  # noqa: PLR2004

    async def test_read(
        self,
        ya_music_fs: YaMusicFS,
        mocker: MockerFixture,
    ) -> None:
        buffer = mock.MagicMock()

        buffer.read_from = mock.AsyncMock()
        buffer.read_from.return_value = b"Test"
        buffer.total_second.return_value = 0

        stream = StreamReader(
            buffer=buffer,
            track=TRACK_INFO,
            is_send_feedback=False,
        )
        mocker.patch.object(ya_music_fs, "_fd_map_stream", {10: stream})
        chunk = await ya_music_fs.read(10, 100, 100)
        assert chunk == b"Test"

    @mock.patch("yandex_fuse.virt_fs.VirtFS._get_file_stat_by_inode")
    async def test_release(
        self,
        mock_get_file_stat_by_inode: mock.Mock(),
        ya_music_fs: YaMusicFS,
        mocker: MockerFixture,
    ) -> None:
        stream = StreamReader(
            buffer=mock.MagicMock(),
            track=TRACK_INFO,
            is_send_feedback=False,
        )
        mocker.patch.object(ya_music_fs, "_fd_map_inode", {10: 519})
        mocker.patch.object(ya_music_fs, "_fd_map_stream", {10: stream})

        await ya_music_fs.release(10)

        # TODO(vm86): check via mock
        assert ya_music_fs._fd_map_inode == {}  # noqa: SLF001
        assert ya_music_fs._fd_map_stream == {}  # noqa: SLF001
