from __future__ import annotations

import errno
import logging
import os
from asyncio import (
    FIRST_COMPLETED,
    CancelledError,
    Event,
    Task,
    create_task,
    sleep,
    wait,
    wait_for,
)
from asyncio import (
    TimeoutError as AsyncTimeoutError,
)
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from aiohttp import (
    ClientError,
    ClientSession,
    ClientTimeout,
    SocketTimeoutError,
)
from pyfuse3 import (
    FileHandleT,
    FileInfo,
    FlagT,
    FUSEError,
    InodeT,
    RequestContext,
)
from yandex_music.exceptions import (  # type: ignore[import-untyped]
    BadRequestError,
    NetworkError,
    NotFoundError,
    TimedOutError,
    UnauthorizedError,
)

from yandex_fuse.virt_fs import VirtFS, fail_is_exit
from yandex_fuse.ya_player import ExtendTrack, TrackTag, YandexMusicPlayer

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Coroutine,
    )

log = logging.getLogger(__name__)

LIMIT_TASKS = 5
LIMIT_ONYOURWAVE = 200

PLAYLIST_ID2NAME = {
    "user:onyourwave": "Моя волна",
    "likes": "Мне нравится",
}


P = ParamSpec("P")
T = TypeVar("T")


def retry_request(
    func: Callable[..., Coroutine[Any, Any, T]], count: int = 3
) -> Callable[..., Coroutine[Any, Any, T | None]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
        for retry in range(1, count + 1):
            log.debug("Retry %d/%d", retry, count)
            try:
                return await func(*args, **kwargs)
            except (BadRequestError, NotFoundError):
                raise
            except (
                NetworkError,
                TimedOutError,
                UnauthorizedError,
                ClientError,
                SocketTimeoutError,
            ) as err:
                log.error("Error request: %r %r/%r", err, args[1:], kwargs)  # noqa:TRY400
                await sleep(0.5)

        return None

    return wrapper


class Buffer:
    CHUNK_SIZE = 128

    def __init__(
        self,
        direct_link: str,
        client_session: ClientSession,
        track: dict[str, Any],
    ) -> None:
        self.__direct_link = direct_link
        self.__client_session = client_session
        self.__bytes = BytesIO()
        self.__data = bytearray()
        self.__track = track
        self.__total_read = 0
        self.__tag = TrackTag.from_json(track)
        self.__download_task: Task[Any] | None = create_task(
            self.download(), name="download"
        )
        self.__tagging = False
        self.__ready_read = Event()

    def __del__(self) -> None:
        self.__ready_read.set()
        if self.__download_task is not None:
            self.__download_task.cancel()

    @property
    def is_downloded(self) -> bool:
        return self.__download_task is None

    def write(self, buffer: bytes) -> int:
        self.__ready_read.set()
        return self.__bytes.write(buffer)

    @property
    def offset(self) -> int:
        if not self.__tagging:
            return 0
        return len(self.__bytes.getbuffer())

    async def read_from(self, offset: int, size: int) -> bytes:
        if self.offset < offset + size:
            while (offset + size - self.offset) > 0:
                if self.is_downloded:
                    break
                self.__ready_read.clear()
                with suppress(AsyncTimeoutError):
                    await wait_for(self.__ready_read.wait(), timeout=15)
        self.__total_read += size
        return bytes(self.__bytes.getbuffer()[offset : offset + size])

    def total_second(self) -> int:
        s2p = self.__tag.duration_ms / self.__track["size"]
        second_read = s2p * self.__total_read
        return int(min(second_read // 1000, self.__tag.duration_ms // 1000))

    @retry_request
    async def download(self) -> None:
        try:
            buffer = BytesIO()
            async with self.__client_session.request(
                "GET",
                self.__direct_link,
                headers={"range": f"bytes={self.offset}-"},
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.content.iter_chunked(self.CHUNK_SIZE):
                    if not self.__tagging:
                        buffer.write(chunk)
                        new_buffer = self.__tag.to_bytes(buffer)
                        if new_buffer is None:
                            continue
                        self.__bytes.seek(0)
                        self.write(new_buffer)
                        self.__tagging = True
                    else:
                        self.write(chunk)
        except CancelledError:
            raise
        except Exception:
            log.exception("Error downloading ..")
            raise
        else:
            log.debug("Track %s downloaded", self.__track["name"])
            if self.__download_task is not None:
                download_task = self.__download_task
                self.__download_task = None
                download_task.cancel()


@dataclass
class StreamReader:
    buffer: Buffer
    track: dict[str, Any]
    is_send_feedback: bool = False


class YaMusicFS(VirtFS):
    CHUNK_SIZE = 128

    def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__client_session: ClientSession | None = None
        self.__ya_player: YandexMusicPlayer | None = None
        self._fd_map_stream: dict[int, StreamReader] = {}
        self._station_id_map_inode: dict[int, int] = {}
        self._tracks = 0

    async def start(self) -> None:
        self.__client_session = ClientSession(
            raise_for_status=False,
            timeout=ClientTimeout(sock_read=3, sock_connect=5),
        )
        self.__ya_player = YandexMusicPlayer(
            Path.home().joinpath(".config/yandex-fuse.json"),
            self.__client_session,
        )
        self.__task = create_task(self.__fsm(), name="fsm")

    @property
    def _client_session(self) -> ClientSession:
        if self.__client_session is None:
            raise RuntimeError("ClientSession is not init!")
        return self.__client_session

    @property
    def _ya_player(self) -> YandexMusicPlayer:
        if self.__ya_player is None:
            raise RuntimeError("ClientSession is not init!")
        return self.__ya_player

    @retry_request
    async def _get_or_update_direct_link(
        self,
        track_id: str,
        codec: str,
        bitrate_in_kbps: int,
    ) -> str | None:
        with self._get_direct_link(track_id) as (direct_link, cursor):
            if direct_link is not None:
                async with self._client_session.request(
                    "HEAD",
                    direct_link,
                ) as resp:
                    if resp.ok:
                        return direct_link

            new_direct_link = await self._ya_player.get_download_link(
                track_id,
                codec,
                bitrate_in_kbps,
            )
            if new_direct_link is None:
                log.warning("Track %s is not be downloaded!", track_id)
                return None
            self._insert_direct_link(cursor, track_id, new_direct_link)
            return new_direct_link

    @retry_request
    async def _update_track(self, track: ExtendTrack, dir_inode: int) -> None:
        if not track.available:
            log.warning(
                "Track %s is not available for listening.",
                track.save_name,
            )
            return

        for exists_track in self._get_links_track_by_id(track.id):
            if exists_track["parent_inode"] == dir_inode:
                return
            log.debug(
                "Link track %s / %s - %s",
                track.id,
                track.save_name,
                track.real_id,
            )
            self._link_track_inode(exists_track["id"], dir_inode)
            return

        log.debug(
            "Create track %s / %s - %s",
            track.id,
            track.save_name,
            track.real_id,
        )
        if (
            direct_link := await self._get_or_update_direct_link(
                track.track_id,
                track.codec,
                track.bitrate_in_kbps,
            )
        ) is None:
            log.warning("Track %s is not be downloaded!", track.save_name)
            return
        async with self._client_session.request(
            "GET",
            direct_link,
        ) as resp:
            track.size = int(resp.headers["Content-Length"])

            byte = BytesIO()
            async for chunk in resp.content.iter_chunked(1024):
                byte.write(chunk)
                old_size = len(byte.getbuffer())
                new_buffer = track.tag.to_bytes(byte)
                if new_buffer is None:
                    continue
                tag_size = len(new_buffer) - old_size
                track.size += tag_size

                break

        if track.save_name.endswith("unknown"):
            log.warning("Track %s create skip.", track.save_name)
            return

        tag = track.tag.to_dict()
        self._create_track(
            (
                track.save_name,
                track.id,
                track.codec,
                track.bitrate_in_kbps,
                track.size,
                tag["artist"],
                tag["title"],
                tag["album"],
                tag["year"],
                tag["genre"],
                tag["duration_ms"],
            ),
            dir_inode,
        )

    def _unlink_track(self, track_id: str, dir_inode: int) -> None:
        for exists_track in self._get_links_track_by_id(track_id):
            if exists_track["parent_inode"] != dir_inode:
                return
            log.debug(
                "Unlink track %s, inode: %d, dir_inode: %d",
                track_id,
                exists_track["inode"],
                dir_inode,
            )
            self.remove(dir_inode, exists_track["inode"])
            self._invalidate_inode(exists_track["inode"])

    @staticmethod
    async def _background_tasks(
        tasks: set[Task[Any]],
    ) -> tuple[set[Task[Any]], bool]:
        error = False
        finished, unfinished = await wait(tasks, return_when=FIRST_COMPLETED)
        for x in finished:
            if (err := x.exception()) is not None:
                log.error("Error task %s: %r", x.get_name(), err)
                error = True
            x.cancel()
        return unfinished, error

    async def __update_like_playlists(self) -> None:
        playlist_id = "likes"
        playlist_info = self._get_playlist_by_id(playlist_id)
        if playlist_info is None:
            self._create_plyalist(PLAYLIST_ID2NAME[playlist_id], playlist_id)
            playlist_info = self._get_playlist_by_id(playlist_id)

        if playlist_info is None:
            raise RuntimeError("Playlist info is empty!")

        dir_inode = playlist_info["inode_id"]
        revision = playlist_info["revision"] or 0

        # BUG: if if_modified_since_revision > 0
        #   Traceback: AttributeError: 'str' object has no attribute 'get'
        users_likes_tracks = await self._ya_player.users_likes_tracks(
            if_modified_since_revision=revision - 1,
        )
        if users_likes_tracks is None:
            log.warning("Like playlist track list empty")
            return

        if revision == users_likes_tracks.revision:
            log.debug("Playlist revision %d, no changes.", revision)
            return

        log.info("Totol like track: %d", len(users_likes_tracks.tracks))

        like_tracks_ids = {track.id for track in users_likes_tracks.tracks}
        loaded_tracks = self._get_tracks()
        tracks = await users_likes_tracks.fetch_tracks()

        new_tracks_ids = like_tracks_ids - loaded_tracks.keys()
        unlink_tracks_ids = loaded_tracks.keys() - like_tracks_ids

        log.debug(
            "New track like playlist: %d. Remove track: %d",
            len(new_tracks_ids),
            len(unlink_tracks_ids),
        )

        tracks = (
            await self._ya_player.tracks(list(new_tracks_ids))
            if new_tracks_ids
            else []
        )

        tasks = set()
        error_update = False

        async for track in self._ya_player.load_tracks(
            tracks,
            exclude_track_ids=set(loaded_tracks.keys()),
        ):
            tasks.add(
                create_task(
                    self._update_track(track, dir_inode),
                    name=f"create-track-{track.save_name}",
                ),
            )
            if len(tasks) > LIMIT_TASKS:
                tasks, error = await self._background_tasks(tasks)
                error_update = error or error_update
        while tasks:
            tasks, error = await self._background_tasks(tasks)
            error_update = error or error_update

        for track_id in unlink_tracks_ids:
            self._unlink_track(track_id, dir_inode)

        if not error_update:
            self._update_plyalist(playlist_id, users_likes_tracks.revision)
        else:
            log.warning("Playlist is partially updated!")

        self._invalidate_inode(dir_inode)
        log.info(
            "Loaded track in like playlist %d.",
            users_likes_tracks.revision,
        )

    async def __update_onyourwave_tracks(self) -> None:
        playlist_id = "user:onyourwave"
        playlist_info = self._get_playlist_by_id(playlist_id)
        if playlist_info is None:
            self._create_plyalist(PLAYLIST_ID2NAME[playlist_id], playlist_id)
            playlist_info = self._get_playlist_by_id(playlist_id)

        if playlist_info is None:
            raise RuntimeError("Playlist info is empty!")

        dir_inode = playlist_info["inode_id"]

        if len(self._get_tracks_by_parent_inode(dir_inode)) >= LIMIT_ONYOURWAVE:
            log.debug("Music directory is full.")
            return

        tasks = set()
        error_update = False
        async for track in self._ya_player.next_tracks("user:onyourwave"):
            tasks.add(
                create_task(
                    self._update_track(track, dir_inode),
                    name=f"create-track-{track.save_name}",
                ),
            )
            if len(tasks) > LIMIT_TASKS:
                tasks, error = await self._background_tasks(tasks)
                error_update = error or error_update

        while tasks:
            tasks, error = await self._background_tasks(tasks)
            error_update = error or error_update

        if not error_update:
            _, batch_id = self._ya_player.get_last_station_info()
            self._update_plyalist(playlist_id, 0, batch_id)
        log.info("Playlist onyourwave is updated.")
        self._invalidate_inode(dir_inode)

    async def __cleanup_track(self) -> None:
        loaded_tracks = self._get_tracks()
        ya_tracks = await self._ya_player.tracks(track_ids=loaded_tracks.keys())
        for ya_track in ya_tracks:
            if ya_track.available:
                continue
            track = loaded_tracks[ya_track.track_id]
            for info_track in self._get_links_track_by_id(ya_track.track_id):
                log.warning(
                    "Track %s is not available for listening. Cleanup inode %d",
                    track["name"],
                    info_track["inode"],
                )
                self.remove(info_track["parent_inode"], info_track["inode"])
                self._invalidate_inode(info_track["inode"])
        log.debug("Cleanup finished.")

    async def __fsm(self) -> None:
        while True:
            try:
                if not self._ya_player.is_init:
                    await sleep(5)
                    continue
                await self.__update_like_playlists()
                await self.__update_onyourwave_tracks()
                await self.__cleanup_track()

                await sleep(300)
            except CancelledError:
                raise
            except Exception:
                log.exception("Error sync:")
                await sleep(60)

    async def _get_buffer(self, track: dict[str, Any]) -> Buffer | None:
        if (
            direct_link := await self._get_or_update_direct_link(
                track["track_id"],
                track["codec"],
                track["bitrate"],
            )
        ) is None:
            log.warning("Track %s is not be downloaded!", track["name"])
            return None
        return Buffer(direct_link, self._client_session, track)

    @fail_is_exit
    async def open(
        self,
        inode: InodeT,
        flags: FlagT,
        ctx: RequestContext,
    ) -> FileInfo:
        track = self._get_track_by_inode(inode)

        if track is None:
            raise FUSEError(errno.ENOENT)

        if not self._ya_player.is_init:
            raise FUSEError(errno.EPERM)

        buffer = await self._get_buffer(track)
        if buffer is None:
            raise FUSEError(errno.EPERM)

        file_info = await super().open(inode, flags, ctx)
        if flags & os.O_RDWR or flags & os.O_WRONLY:
            raise FUSEError(errno.EPERM)

        log.debug("Open stream %s -> %d", track["name"], file_info.fh)
        self._fd_map_stream[file_info.fh] = StreamReader(
            track=track,
            buffer=buffer,
        )

        return file_info

    @fail_is_exit
    async def read(self, fd: FileHandleT, offset: int, size: int) -> bytes:
        stream_reader = self._fd_map_stream[fd]

        chunk = await stream_reader.buffer.read_from(offset, size)

        if len(chunk) > size:
            log.warning(
                "Chunk is corrupt. Invalid size chunk %d > %d",
                len(chunk),
                size,
            )
            raise FUSEError(errno.EPIPE)

        try:
            if (
                not stream_reader.is_send_feedback
                and stream_reader.buffer.is_downloded
                and stream_reader.buffer.total_second() > 30  # noqa: PLR2004
            ):
                playlist = self._get_playlist_by_inode(
                    stream_reader.track["parent_inode"],
                )

                if playlist is not None and playlist["batch_id"] is not None:
                    await self._ya_player.feedback_track(
                        stream_reader.track["track_id"],
                        "trackStarted",
                        playlist["batch_id"],
                        0,
                    )
                    stream_reader.is_send_feedback = True
        except CancelledError:
            raise
        except Exception:
            log.exception("Error send feedback:")

        return chunk

    @fail_is_exit
    async def release(self, fd: FileHandleT) -> None:
        await super().release(fd)

        stream_reader = self._fd_map_stream.pop(fd, None)
        if stream_reader is None:
            log.warning("FD %d is none.", fd)
            return
        log.debug("Release stream %d > %s", fd, stream_reader.track["name"])
        parent_inode = stream_reader.track["parent_inode"]
        playlist = self._get_playlist_by_inode(parent_inode)

        if playlist is None:
            raise RuntimeError("Playlist info is empty!")

        if playlist["batch_id"] is None:
            return

        total_second = 10
        if stream_reader.buffer is not None:
            total_second = stream_reader.buffer.total_second()
        if total_second == 0:
            return

        try:
            await self._ya_player.feedback_track(
                stream_reader.track["track_id"],
                "trackFinished",
                playlist["batch_id"],
                total_second,
            )
        except CancelledError:
            raise
        except Exception:
            log.exception("Error send feedback:")

    def xattrs(self, inode: InodeT) -> dict[str, Any]:
        track_info = self._get_track_by_inode(inode) or {}
        return {
            "inode": inode,
            "nlookup": len(self._nlookup),
            "fd_map_inode": len(self._fd_map_inode),
            "queue_invalidate_inode": self.queue_later_invalidate_inode,
            "track_info": track_info,
            "stream": {
                fd: {
                    "name": stream.track["name"],
                    "size": stream.track["size"],
                    "codec": stream.track["codec"],
                    "bitrate": stream.track["bitrate"],
                    "play_second": stream.buffer.total_second()
                    if stream.buffer is not None
                    else 0,
                }
                for fd, stream in self._fd_map_stream.items()
            },
        }
