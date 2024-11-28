from __future__ import annotations

import errno
import logging
import os
import stat
import time
from asyncio import (
    FIRST_COMPLETED,
    CancelledError,
    Event,
    InvalidStateError,
    Task,
    create_task,
    sleep,
    wait,
    wait_for,
)
from asyncio import (
    TimeoutError as AsyncTimeoutError,
)
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from aiohttp import (
    ClientError,
    ClientSession,
    ClientTimeout,
    ConnectionTimeoutError,
    SocketTimeoutError,
)
from pyfuse3 import (
    FileHandleT,
    FileInfo,
    FlagT,
    FUSEError,
    InodeT,
    RequestContext,
    XAttrNameT,
)
from yandex_music.exceptions import (  # type: ignore[import-untyped]
    BadRequestError,
    NetworkError,
    NotFoundError,
    TimedOutError,
    UnauthorizedError,
)

from yandex_fuse.virt_fs import SQLRow, VirtFS, fail_is_exit
from yandex_fuse.ya_player import ExtendTrack, TrackTag, YandexMusicPlayer

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Coroutine,
        Iterator,
    )
    from sqlite3 import Cursor

log = logging.getLogger(__name__)

LIMIT_TASKS = 10
LIMIT_ONYOURWAVE = 150

PLAYLIST_ID2NAME = {"likes": "Мне нравится", "user:onyourwave": "Моя волна"}


P = ParamSpec("P")
T = TypeVar("T")


def retry_request(
    func: Callable[..., Coroutine[Any, Any, T]], count: int = 3
) -> Callable[..., Coroutine[Any, Any, T | None]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
        for retry in range(1, count + 1):
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
                await sleep(0.75 * retry)
            log.debug("Retry %d/%d - %r", retry + 1, count, func)
        raise RuntimeError("Retry limit exceeded.")

    return wrapper


class Buffer:
    CHUNK_SIZE = 128

    def __init__(
        self,
        direct_link: str,
        client_session: ClientSession,
        track: SQLTrack,
        fuse_music: YaMusicFS,
    ) -> None:
        self.__direct_link = direct_link
        self.__client_session = client_session
        self.__bytes = BytesIO()
        self.__data = bytearray()
        self.__track = track
        self.__total_read = 0
        self.__tag = TrackTag.from_json(track.__dict__)
        self.__fuse_music = fuse_music
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
                with suppress(InvalidStateError):
                    if self.__download_task is not None:
                        self.__download_task.result()

                self.__ready_read.clear()
                try:
                    await wait_for(self.__ready_read.wait(), timeout=5)
                except AsyncTimeoutError:
                    log.warning("Slow downloading %s", self.__track.name)

        self.__total_read += size
        return bytes(self.__bytes.getbuffer()[offset : offset + size])

    def total_second(self) -> int:
        s2p = self.__tag.duration_ms / self.__track.size
        second_read = s2p * self.__total_read
        return int(min(second_read // 1000, self.__tag.duration_ms // 1000))

    def _write_tag(self, new_buffer: bytes, real_size: int) -> None:
        self.__bytes.seek(0)
        self.write(new_buffer)
        self.__tagging = True

        if real_size + self.__tag.size != self.__track.size:
            log.warning(
                "Missmath tagging. "
                "Track size: %d, tag size: %d "
                "Content-Length: %d",
                self.__track.size,
                self.__tag.size,
                real_size,
            )

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
                    if self.__tagging:
                        self.write(chunk)
                        continue

                    buffer.write(chunk)
                    new_buffer = self.__tag.to_bytes(buffer, self.__track.codec)
                    if new_buffer is None:
                        continue
                    self._write_tag(
                        new_buffer, int(resp.headers["Content-Length"])
                    )

        except CancelledError:
            raise
        except ConnectionTimeoutError:
            direct_link = await self.__fuse_music.get_or_update_direct_link(
                self.__track.track_id,
                self.__track.codec,
                self.__track.bitrate,
            )
            if direct_link is None:
                raise RuntimeError("Error get direct link.") from None
            self.__direct_link = direct_link
            raise
        except Exception:
            log.exception("Error downloading ..")
            raise
        else:
            log.debug("Track %s downloaded", self.__track.name)
            self.__ready_read.set()
            if self.__download_task is None:
                return
            download_task = self.__download_task
            self.__download_task = None
            download_task.cancel()


@dataclass
class SQLTrack(SQLRow):
    __tablename__ = "tracks"

    name: bytes
    track_id: str
    playlist_id: str
    codec: str
    bitrate: int
    quality: str
    size: int
    artist: str
    title: str
    album: str
    year: str
    genre: str
    duration_ms: int

    inode: InodeT
    id: int | None = None


@dataclass
class SQLPlaylist(SQLRow):
    __tablename__ = "playlists"

    name: str
    playlist_id: str
    inode: InodeT
    revision: int = 0
    batch_id: str = ""
    station_id: str = ""
    id: int | None = None


@dataclass
class SQLDirectLink(SQLRow):
    __tablename__ = "direct_link"

    track_id: str
    link: str
    expired: int
    id: int | None = None


@dataclass
class StreamReader:
    buffer: Buffer
    track: SQLTrack
    is_send_feedback: bool = False


class YaMusicFS(VirtFS):
    FILE_DB = str(Path.home().joinpath(".cache/yandex-fuse2.db"))

    CREATE_TABLE_QUERYS = (
        *VirtFS.CREATE_TABLE_QUERYS,
        """
        CREATE TABLE IF NOT EXISTS playlists (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            inode        INT NOT NULL
                REFERENCES inodes(id) ON DELETE CASCADE,
            playlist_id  TEXT(255) NOT NULL UNIQUE,
            station_id   TEXT(255),
            batch_id     TEXT(255),
            revision     INT DEFAULT 0,
            name         TEXT(255) NOT NULL,
            UNIQUE       (name, playlist_id, inode)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tracks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            inode        INT NOT NULL
                REFERENCES inodes(id) ON DELETE CASCADE,
            name         BLOB(256) NOT NULL,
            track_id     TEXT(255) NOT NULL UNIQUE,
            playlist_id  TEXT(255) NOT NULL
                REFERENCES playlists(playlist_id) ON DELETE RESTRICT,
            codec        BLOB(8) NOT NULL,
            bitrate      INT NOT NULL,
            quality      TEXT(20) NOT NULL,
            size         INT NOT NULL,
            artist       TEXT(255),
            title        TEXT(255),
            album        TEXT(255),
            year         TEXT(255),
            genre        TEXT(255),
            duration_ms  INT,
            UNIQUE       (name, track_id, inode)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS direct_link (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id   TEXT(255) NOT NULL UNIQUE,
            link       BLOB(256) NOT NULL,
            expired    INT NOT NULL
        )
        """,
    )

    CHUNK_SIZE = 128

    def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__client_session: ClientSession | None = None
        self.__ya_player: YandexMusicPlayer | None = None
        self._fd_map_stream: dict[int, StreamReader] = {}

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
            raise RuntimeError(
                "YandexMusicPlayer is not init! %r", self.__ya_player
            )
        return self.__ya_player

    @contextmanager
    def _get_direct_link(
        self, track_id: str
    ) -> Iterator[tuple[str | None, Cursor]]:
        with self._db_cursor() as cur:
            cur.execute(
                "SELECT * FROM direct_link WHERE track_id=?",
                (track_id,),
            )
            row = cur.fetchone()

            if row is None:
                yield (None, cur)
                return
            expired = int(time.time() * 1e9)
            if row["expired"] > expired:
                log.debug(
                    "Track %s, lifetime %r",
                    track_id,
                    (row["expired"] - expired) / 1e9,
                )
                yield (row["link"], cur)
                return
            log.debug("Direct link %s direct link has expired", track_id)
            yield (None, cur)

    @retry_request
    async def get_or_update_direct_link(
        self,
        track_id: str,
        codec: str,
        bitrate_in_kbps: int,
    ) -> str | None:
        with self._get_direct_link(track_id) as (direct_link, cursor):
            if direct_link is not None:
                try:
                    async with self._client_session.request(
                        "HEAD",
                        direct_link,
                    ) as resp:
                        if resp.ok:
                            return direct_link
                except ClientError as err:
                    log.error("Fail get direct link: %r", err)  # noqa: TRY400

            new_direct_links = await self._ya_player.get_download_links(
                track_id,
                codec,
                bitrate_in_kbps,
            )

            if new_direct_links is None:
                return None

            for new_direct_link in new_direct_links:
                log.debug("Check direct link %s", new_direct_link)
                try:
                    async with self._client_session.request(
                        "HEAD",
                        new_direct_link,
                    ) as resp:
                        if not resp.ok:
                            continue
                except ClientError as err:
                    log.error("Fail get direct link: %r", err)  # noqa: TRY400
                    continue

                expired = int((time.time() + 8600) * 1e9)
                cursor.execute(
                    """
                    INSERT INTO direct_link
                    (track_id, link, expired)
                    VALUES(?, ?, ?)
                    ON CONFLICT(track_id)
                    DO UPDATE SET link=excluded.link,expired=excluded.expired
                    """,
                    (track_id, new_direct_link, expired),
                )
                log.debug(
                    "Direct link: %s, track: %s", new_direct_link, track_id
                )
                return new_direct_link
        return None

    def _get_playlist_by_id(self, playlist_id: str) -> SQLPlaylist | None:
        return SQLPlaylist.from_row(
            self._get_dict_row(
                """
            SELECT
                *
            FROM playlists
            WHERE
                playlists.playlist_id = ?
            """,
                (playlist_id,),
            )
        )

    def _create_plyalist(self, playlist: SQLPlaylist) -> SQLPlaylist:
        with self._db_cursor() as cur:
            inode = self._create(
                parent_inode=self.ROOT_INODE,
                name=playlist.name.encode(),
                size=0,
                mode=(stat.S_IFDIR | 0o755),
                target=b"",
                db_cursor=cur,
            )
            playlist.inode = inode

            cur.execute(*playlist.insert())

        return playlist

    def _update_plyalist(
        self,
        uid: str,
        revision: int,
        batch_id: str | None = None,
    ) -> None:
        with self._db_cursor() as cur:
            cur.execute(
                """
                UPDATE playlists SET revision=?,batch_id=? WHERE playlist_id=?;
            """,
                (
                    revision,
                    batch_id,
                    uid,
                ),
            )

    def _get_track_by_id(self, track_id: str) -> SQLTrack | None:
        result = self._get_dict_row(
            """
            SELECT
                *
            FROM tracks
            WHERE track_id=?
            """,
            (track_id,),
        )
        if result is None:
            return None
        return SQLTrack(**result)

    def _get_track_by_inode(self, inode: int) -> SQLTrack | None:
        return SQLTrack.from_row(
            self._get_dict_row(
                """
            SELECT
                *
            FROM tracks
            WHERE inode=?
            """,
                (inode,),
            )
        )

    def _create_track(self, track: SQLTrack, parent_inode: int) -> None:
        with self._db_cursor() as cur:
            inode = self._create(
                parent_inode=InodeT(parent_inode),
                name=track.name,
                size=track.size,
                mode=(stat.S_IFREG | 0o644),
                target=b"",
                db_cursor=cur,
            )
            log.debug(
                "Create track %s, inode: %d, parent inode: %d",
                track.name,
                inode,
                parent_inode,
            )
            track.inode = inode
            cur.execute(*track.insert())

    def _symlink_track(
        self, exist_track: SQLTrack, track: ExtendTrack, dir_inode: InodeT
    ) -> None:
        playlist_info = self._get_playlist_by_id(exist_track.playlist_id)
        if playlist_info is None:
            raise RuntimeError("Error get playlist info")

        target = str(
            Path("..")
            .joinpath(playlist_info.name)
            .joinpath(exist_track.name.decode())
        ).encode()

        track_link = self._get_inode_by_name(
            InodeT(dir_inode), track.save_name.encode()
        )
        if track_link is not None:
            return

        with self._db_cursor() as cur:
            inode = self._create(
                parent_inode=dir_inode,
                name=track.save_name.encode(),
                size=4096,
                mode=(stat.S_IFLNK | 0o777),
                target=target,
                db_cursor=cur,
            )
        log.debug(
            "Symlink track %s -> %d / %s - %d",
            track.track_id,
            inode,
            track.save_name,
            dir_inode,
        )

    async def _update_track(
        self,
        track: ExtendTrack,
        playlist_id: str,
        dir_inode: InodeT,
        *,
        uniq: bool = False,
    ) -> None:
        if (
            exist_track := self._get_track_by_id(track.track_id)
        ) is not None and exist_track.playlist_id != playlist_id:
            if uniq:
                return
            self._symlink_track(exist_track, track, dir_inode)
            return

        if (
            direct_link := await self.get_or_update_direct_link(
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
                new_buffer = track.tag.to_bytes(byte, track.codec)
                if new_buffer is None:
                    continue
                track.size += track.tag.size
                break

        if track.save_name.endswith("unknown"):
            log.warning("Track %s create skip.", track.save_name)
            return

        if (
            inode_by_name := self._get_inode_by_name(
                dir_inode, track.save_name.encode()
            )
        ) is not None:
            exist_track = self._get_track_by_inode(inode_by_name)
            if exist_track is None:
                raise RuntimeError(
                    f"""
                    Missing track {track.save_name} / {track.track_id}
                    by inode {inode_by_name}"""
                )
            log.debug(
                "Update track %s, track id %s -> %s, inode: %d",
                track.save_name,
                exist_track.track_id,
                track.track_id,
                inode_by_name,
            )
            with self._db_cursor() as cur:
                cur.execute(
                    """
                UPDATE tracks SET track_id=?,size=? WHERE id=?;
                """,
                    (track.track_id, track.size, exist_track.id),
                )
                cur.execute(
                    """
                UPDATE inodes SET size=? WHERE id=?;
                """,
                    (track.size, exist_track.inode),
                )
            return

        log.debug(
            "Create track %s / %s - %d",
            track.track_id,
            track.save_name,
            dir_inode,
        )

        tag = track.tag.to_dict()
        self._create_track(
            SQLTrack(
                name=track.save_name.encode(),
                track_id=track.track_id,
                playlist_id=playlist_id,
                codec=track.codec,
                bitrate=track.bitrate_in_kbps,
                quality=track.quality,
                size=track.size,
                artist=tag["artist"],
                title=tag["title"],
                album=tag["album"],
                year=tag["year"],
                genre=tag["genre"],
                duration_ms=tag["duration_ms"],
                inode=InodeT(-1),
            ),
            dir_inode,
        )

    def _unlink_track(self, track_id: str, dir_inode: InodeT) -> None:
        track = self._get_track_by_id(track_id)
        if track is None:
            return
        with self._db_cursor() as cur:
            if self.remove(dir_inode, track.inode):
                cur.execute(
                    "DELETE FROM tracks WHERE inode=? AND track_id=?",
                    (track.inode, track.track_id),
                )
        self._invalidate_inode(track.inode)

    def _get_tracks(
        self, playlist_id: str | None = None
    ) -> dict[str, SQLTrack]:
        result = {}
        with self._db_cursor() as cur:
            cur.execute(
                """
                SELECT
                    *
                FROM tracks
                ORDER BY track_id
                """,
            )

            for row in cur.fetchall():
                if playlist_id and row["playlist_id"] != playlist_id:
                    continue
                result[row["track_id"]] = SQLTrack(**row)
        return result

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

        playlist_info = self._get_playlist_by_id(
            playlist_id
        ) or self._create_plyalist(
            SQLPlaylist(
                name=PLAYLIST_ID2NAME[playlist_id],
                playlist_id=playlist_id,
                station_id="",
                batch_id="",
                revision=0,
                inode=InodeT(-1),
            ),
        )

        dir_inode = playlist_info.inode
        revision = playlist_info.revision

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

        like_tracks_ids = {
            track.track_id for track in users_likes_tracks.tracks
        }
        loaded_tracks = self._get_tracks(playlist_id)
        tracks = await users_likes_tracks.fetch_tracks()

        new_tracks_ids = like_tracks_ids - loaded_tracks.keys()
        unlink_tracks_ids = loaded_tracks.keys() - like_tracks_ids

        log.debug(
            "New track like playlist: %d. Remove track: %d, revision: %d/%d",
            len(new_tracks_ids),
            len(unlink_tracks_ids),
            revision,
            users_likes_tracks.revision,
        )

        tracks = (
            await self._ya_player.tracks(list(new_tracks_ids))
            if new_tracks_ids
            else []
        )

        tasks = set()
        error_update = False

        for track_id in unlink_tracks_ids:
            self._unlink_track(track_id, dir_inode)

        async for track in self._ya_player.load_tracks(
            tracks,
            exclude_track_ids=set(loaded_tracks.keys()),
        ):
            tasks.add(
                create_task(
                    self._update_track(track, playlist_id, dir_inode),
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
            self._update_plyalist(playlist_id, users_likes_tracks.revision)
        else:
            log.warning("Playlist is partially updated!")

        self._invalidate_inode(dir_inode)
        log.info(
            "Loaded track in like playlist %d.",
            users_likes_tracks.revision,
        )

    async def __update_station_tracks(
        self, playlist_id: str, playlist_name: str
    ) -> None:
        playlist_info = self._get_playlist_by_id(
            playlist_id
        ) or self._create_plyalist(
            SQLPlaylist(
                name=playlist_name,
                playlist_id=playlist_id,
                station_id=playlist_id,
                batch_id="",
                revision=0,
                inode=InodeT(-1),
            )
        )

        if playlist_info is None:
            raise RuntimeError("Playlist info is empty!")

        dir_inode = playlist_info.inode

        tasks = set()
        error_update = False

        while (
            len(loaded_tracks := self._get_tracks(playlist_id))
            < LIMIT_ONYOURWAVE
        ):
            async for track in self._ya_player.next_tracks(
                playlist_id,
                count=LIMIT_ONYOURWAVE,
                exclude_track_ids=set(loaded_tracks.keys()),
            ):
                tasks.add(
                    create_task(
                        self._update_track(
                            track, playlist_id, dir_inode, uniq=True
                        ),
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

        if not loaded_tracks:
            return

        loaded_tracks_by_id = {}
        for track_id, track in loaded_tracks.items():
            track_id_without_album_id, *unused = track_id.split(":", 1)
            loaded_tracks_by_id[track_id_without_album_id] = track

        ya_tracks = await self._ya_player.tracks(track_ids=loaded_tracks.keys())
        for ya_track in ya_tracks:
            if ya_track.available:
                continue

            try:
                info_track = loaded_tracks[ya_track.track_id]
            except KeyError:
                info_track = loaded_tracks_by_id[ya_track.id]

            log.warning(
                "Track %s is not available for listening. Cleanup inode %d",
                ya_track.title,
                info_track.inode,
            )
            playlist_info = self._get_playlist_by_id(info_track.playlist_id)
            if playlist_info is None:
                raise RuntimeError("Playlist doesn't exist")
            self._unlink_track(info_track.track_id, playlist_info.inode)

        with self._db_cursor() as cur:
            cur.execute(
                """
            DELETE FROM inodes
            WHERE inodes.id IN (
                SELECT a.id
                FROM (
                    SELECT
                        inodes.id,
                        dentrys.inode
                    FROM inodes
                    LEFT JOIN dentrys ON dentrys.inode = inodes.id
                ) a
                INNER JOIN
                    (
                        SELECT
                            inodes.id,
                            dentrys.inode
                        FROM inodes
                        LEFT JOIN dentrys ON dentrys.parent_inode = inodes.id
                    ) b
                    ON a.id = b.id
                WHERE a.inode IS NULL AND b.inode IS NULL
            )
            """
            )
            cur.execute(
                """
            DELETE FROM tracks
            WHERE
                tracks.inode IN (
                    SELECT tracks.inode
                    FROM tracks
                    LEFT JOIN inodes ON tracks.inode = inodes.id
                    WHERE inodes.id IS NULL
                );
            """
            )
            cur.execute("""
            DELETE FROM direct_link
            WHERE
                id IN (
                    SELECT direct_link.id
                    FROM direct_link
                    LEFT JOIN tracks
                        ON direct_link.track_id = tracks.track_id
                    WHERE tracks.track_id IS NULL
                );
            """)
        log.debug("Cleanup finished %d/%d", len(ya_tracks), len(loaded_tracks))

    async def __fsm(self) -> None:
        while True:
            try:
                if not self._ya_player.is_init:
                    await sleep(5)
                    continue

                await self.__cleanup_track()
                await self.__update_like_playlists()

                playlist_onyourwave_id = "user:onyourwave"
                await self.__update_station_tracks(
                    playlist_onyourwave_id,
                    PLAYLIST_ID2NAME[playlist_onyourwave_id],
                )
                # playlists = await self._ya_player.users_playlists_list()
                # for playlist in playlists:
                #    print(playlist.kind, playlist.title)

                # stations = await self._ya_player.rotor_stations_list()

                # for station in stations:
                #    _station = station.station
                #    plyalist_id = f"{_station.id.type}:{_station.id.tag}"
                #    plyalist_name = _station.name
                #    print(plyalist_id, plyalist_name)
                # await self.__update_station_tracks()

                await sleep(300)
            except CancelledError:
                raise
            except Exception:
                log.exception("Error sync:")
                await sleep(60)

    async def _get_buffer(self, track: SQLTrack) -> Buffer | None:
        if (
            direct_link := await self.get_or_update_direct_link(
                track.track_id,
                track.codec,
                track.bitrate,
            )
        ) is None:
            log.warning("Track %s is not be downloaded!", track.name)
            return None
        return Buffer(direct_link, self._client_session, track, self)

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
            raise FUSEError(errno.EPIPE)

        file_info = await super().open(inode, flags, ctx)
        if flags & os.O_RDWR or flags & os.O_WRONLY:
            raise FUSEError(errno.EPERM)

        log.debug("Open stream %s -> %d", track.name, file_info.fh)
        self._fd_map_stream[file_info.fh] = StreamReader(
            track=track,
            buffer=buffer,
        )

        return file_info

    @fail_is_exit
    async def read(self, fd: FileHandleT, offset: int, size: int) -> bytes:
        stream_reader = self._fd_map_stream[fd]

        try:
            chunk = await stream_reader.buffer.read_from(offset, size)
        except RuntimeError:
            raise FUSEError(errno.EPIPE) from None

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
                playlist = self._get_playlist_by_id(
                    stream_reader.track.playlist_id
                )

                if playlist is not None and playlist.batch_id:
                    await self._ya_player.feedback_track(
                        stream_reader.track.track_id,
                        "trackStarted",
                        playlist.station_id,
                        playlist.batch_id,
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
        log.debug("Release stream %d > %s", fd, stream_reader.track.name)

        try:
            if (
                not stream_reader.is_send_feedback
                and stream_reader.buffer.is_downloded
            ):
                playlist = self._get_playlist_by_id(
                    stream_reader.track.playlist_id
                )

                if playlist is not None and playlist.batch_id:
                    await self._ya_player.feedback_track(
                        stream_reader.track.track_id,
                        "trackFinished",
                        playlist.station_id,
                        playlist.batch_id,
                        stream_reader.buffer.total_second(),
                    )
                    stream_reader.is_send_feedback = True
        except CancelledError:
            raise
        except Exception:
            log.exception("Error send feedback:")

    @fail_is_exit
    async def setxattr(
        self,
        inode: InodeT,
        name: XAttrNameT,
        value: bytes,
        ctx: RequestContext,  # noqa: ARG002
    ) -> None:
        track = self._get_track_by_inode(inode)
        if track is None:
            raise FUSEError(errno.ENOENT)

        playlist_info = self._get_playlist_by_id(track.playlist_id)
        if playlist_info is None:
            raise FUSEError(errno.ENOENT)

        if name == b".invalidate":
            self._invalidate_inode(inode)

        if name == b"update":
            if value == b".recreate":
                self._unlink_track(track.track_id, playlist_info.inode)

            ya_tracks = await self._ya_player.tracks(track_ids=[track.track_id])
            async for ya_track in self._ya_player.load_tracks(
                ya_tracks, exclude_track_ids=set()
            ):
                await self._update_track(
                    ya_track, track.playlist_id, playlist_info.inode
                )

    def xattrs(self, inode: InodeT) -> dict[str, Any]:
        return {
            "inode": inode,
            "inode_map_fd": self._inode_map_fd.get(inode),
            "stream": {
                fd: {
                    "name": stream.track.name.decode(),
                    "size": stream.track.size,
                    "codec": stream.track.codec,
                    "bitrate": stream.track.bitrate,
                    "play_second": stream.buffer.total_second(),
                }
                for fd, stream in self._fd_map_stream.items()
            },
        }
