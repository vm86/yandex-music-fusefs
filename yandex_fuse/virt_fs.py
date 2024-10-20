from __future__ import annotations

import errno
import logging
import os
import sqlite3
import stat
import sys
import time
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, suppress
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

import pyfuse3

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

log = logging.getLogger(__name__)


class FileStat(pyfuse3.EntryAttributes):
    def __init__(self) -> None:
        self.st_mode = 0
        self.st_ino = 0
        self.st_dev = 0
        self.st_nlink = 0
        self.st_uid = 0
        self.st_gid = 0
        self.st_size = 0
        self.st_atime = 0
        self.st_mtime = 0
        self.st_ctime = 0


T = TypeVar("T")


# Если возникает исключение в FUSE, то вывоз зависает.
def fail_is_exit(func: Callable) -> T:
    async def wrapped(*args: str, **kwargs: dict) -> T:
        try:
            return await func(*args, **kwargs)
        except pyfuse3.FUSEError:
            raise
        except Exception:
            log.exception("Error %r", func)
            sys.exit(42)

    return wrapped


class VirtFS(pyfuse3.Operations):
    CREATE_TABLE_QUERYS = (
        """
        CREATE TABLE inodes (
            id           INTEGER PRIMARY KEY,
            parent_inode INTEGER NOT NULL,
            type         TEXT(255),
            is_dir       TINYINT NOT NULL,
            mtime_ns     INT NOT NULL,
            atime_ns     INT NOT NULL,
            ctime_ns     INT NOT NULL,
            reference_id INTEGER,
            UNIQUE       (parent_inode, reference_id)
        )
        """,
        """
        CREATE TABLE tracks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         BLOB(256) NOT NULL,
            track_id     TEXT(255) NOT NULL UNIQUE,
            codec        BLOB(8) NOT NULL,
            bitrate      INT NOT NULL,
            size         INT NOT NULL,
            artist       TEXT(255),
            title        TEXT(255),
            album        TEXT(255),
            year         TEXT(255),
            genre        TEXT(255),
            duration_ms  INT,
            UNIQUE       (name, track_id)
        )
        """,
        """
        CREATE TABLE playlists (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            playlist_id  TEXT(255) NOT NULL UNIQUE,
            station_id   TEXT(255),
            batch_id     TEXT(255),
            revision     INT DEFAULT 0,
            name         TEXT(255) NOT NULL
        )
        """,
        """
        CREATE TABLE direct_link (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id   TEXT(255) NOT NULL REFERENCES tracks(track_id) UNIQUE,
            link       BLOB(256) NOT NULL,
            expired    INT NOT NULL
        )
        """,
    )

    ROOT_INODE = pyfuse3.ROOT_INODE

    def __init__(self) -> None:
        super().__init__()
        file_db = Path.home().joinpath(".cache/yandex-fuse.sqlite3")
        is_new = not file_db.exists()
        self._db = sqlite3.connect(file_db, uri=True)

        self._db.row_factory = sqlite3.Row

        self._fd_map_inode = defaultdict(set)

        self._nlookup = defaultdict(int)

        self.__fd_token_map_offset_read = defaultdict(list)

        if is_new:
            self.__init_table()

    def __init_table(self) -> None:
        with self.__db_cursor() as cur:
            log.debug("Init database.")

            for create_table_query in self.CREATE_TABLE_QUERYS:
                cur.execute(create_table_query)

            # Insert root directory
            now_ns = int(time.time() * 1e9)
            cur.execute(
                "INSERT INTO inodes"
                "(id, parent_inode, type, is_dir, mtime_ns, atime_ns, ctime_ns)"
                "VALUES (?,?,?,?,?,?,?)",
                (
                    self.ROOT_INODE,
                    0,
                    "ROOT",
                    1,
                    now_ns,
                    now_ns,
                    now_ns,
                ),
            )

    @contextmanager
    def __db_cursor(self, *, connect: bool = False) -> sqlite3.Cursor:
        if connect:
            yield self._db
        else:
            with self._db:
                yield self._db.cursor()

    def _get_row(self, *a: str, **kw: dict) -> dict:
        with self.__db_cursor() as cur:
            cur.execute(*a, **kw)
            result = cur.fetchone()
            if result:
                if len(result) == 1:
                    return result[0]
                return dict(result)
            return None

    def _get_fd(self) -> int:
        for i in range(2048):
            if i in self._fd_map_inode:
                continue

            return i
        raise pyfuse3.FUSEError(errno.ENOENT)

    def _create_plyalist(self, name: str, uid: str) -> int:
        with self.__db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO playlists (name, playlist_id)
                VALUES (?, ?)
            """,
                (
                    name,
                    uid,
                ),
            )
            row_id = cur.lastrowid
            now_ns = int(time.time() * 1e9)

            cur.execute(
                """
                INSERT INTO inodes
                (
                    type,
                    parent_inode,
                    is_dir,
                    mtime_ns,
                    atime_ns,
                    ctime_ns,
                    reference_id
                )
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    "PLAYLIST",
                    self.ROOT_INODE,
                    1,
                    now_ns,
                    now_ns,
                    now_ns,
                    row_id,
                ),
            )
            return cur.lastrowid

    def _update_plyalist(
        self,
        uid: str,
        revision: int,
        batch_id: str | None = None,
    ) -> None:
        now_ns = int(time.time() * 1e9)

        with self.__db_cursor() as cur:
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

            cur.execute(
                """
                UPDATE inodes
                SET mtime_ns=?
                WHERE inodes.reference_id=(
                    SELECT id FROM playlists WHERE playlist_id=?
                )
                AND inodes.type = 'PLAYLIST';
            """,
                (
                    now_ns,
                    uid,
                ),
            )

    def _get_playlist_by_id(self, uid: str) -> dict:
        return self._get_row(
            """
            SELECT
                inodes.id AS inode_id,
                playlists.station_id AS station_id,
                playlists.batch_id AS batch_id,
                playlists.revision AS revision,
                playlists.name AS name
            FROM playlists
            INNER JOIN inodes ON inodes.reference_id = playlists.id
            WHERE
                playlists.playlist_id = ?
                AND inodes.type = 'PLAYLIST';
            """,
            (uid,),
        )

    def _get_playlist_by_inode(self, inode: int) -> dict:
        return self._get_row(
            """
            SELECT
                inodes.id AS inode_id,
                playlists.station_id AS station_id,
                playlists.batch_id AS batch_id,
                playlists.revision AS revision,
                playlists.name AS name
            FROM playlists
            INNER JOIN inodes ON inodes.reference_id = playlists.id
            WHERE
                inodes.id = ?
                AND inodes.type = 'PLAYLIST';
            """,
            (inode,),
        )

    def _get_link_track_by_id(self, track_id: str) -> dict | None:
        # BUG: must return list. reference_id -> many track
        return self._get_row(
            """
            SELECT
                inodes.id AS inode,
                inodes.parent_inode as parent_inode,
                tracks.id AS id
            FROM tracks
            LEFT JOIN inodes ON inodes.reference_id = tracks.id
            WHERE tracks.track_id = ?
                AND inodes.type = 'TRACK';
            """,
            (track_id,),
        )

    def _get_track_by_inode(self, inode: int) -> dict:
        return self._get_row(
            """
            SELECT
                *
            FROM tracks
            LEFT JOIN inodes ON inodes.reference_id = tracks.id
            WHERE inodes.id=?
                AND inodes.type = 'TRACK';
            """,
            (inode,),
        )

    def _get_tracks_by_parent_inode(self, parent_inode: int) -> list[dict]:
        with self.__db_cursor() as cur:
            cur.execute(
                """
            SELECT
                *
            FROM tracks
            LEFT JOIN inodes ON inodes.reference_id = tracks.id
            WHERE inodes.parent_inode=?
                AND inodes.type = 'TRACK';
            """,
                (parent_inode,),
            )
            return list(cur.fetchall())

    def _create_track(self, track: dict, parent_inode: int = -1) -> None:
        with self.__db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO tracks
                (
                    name,
                    track_id,
                    codec,
                    bitrate,
                    size,
                    artist,
                    title,
                    album,
                    year,
                    genre,
                    duration_ms
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                track,
            )
            row_id = cur.lastrowid
            now_ns = int(time.time() * 1e9)

            cur.execute(
                """
                INSERT INTO inodes
                (
                    type,
                    parent_inode,
                    is_dir,
                    mtime_ns,
                    atime_ns,
                    ctime_ns,
                    reference_id
                )
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    "TRACK",
                    parent_inode,
                    0,
                    now_ns,
                    now_ns,
                    now_ns,
                    row_id,
                ),
            )

    def _link_track_inode(self, track_id: int, dir_inode: int) -> None:
        now_ns = int(time.time() * 1e9)
        with self.__db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO inodes
                (
                    type,
                    parent_inode,
                    is_dir,
                    mtime_ns,
                    atime_ns,
                    ctime_ns,
                    reference_id
                )
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    "TRACK",
                    dir_inode,
                    0,
                    now_ns,
                    now_ns,
                    now_ns,
                    track_id,
                ),
            )

    def _get_tracks(self) -> dict[int, Any]:
        result = {}
        with self.__db_cursor() as cur:
            cur.execute(
                """
                SELECT
                    tracks.name AS name,
                    tracks.track_id AS track_id,
                    tracks.codec AS codec,
                    tracks.bitrate AS bitrate,
                    tracks.size AS size,
                    tracks.artist AS tag_artist,
                    tracks.title AS tag_title,
                    tracks.album AS tag_album,
                    tracks.year AS tag_year,
                    tracks.genre AS tag_genre,
                    tracks.duration_ms AS tag_duration_ms
                FROM tracks
                ORDER BY track_id
                """,
            )

            for row in cur.fetchall():
                result[row["track_id"]] = dict(row)
        return result

    @contextmanager
    def _get_direct_link(
        self, track_id: str
    ) -> AbstractContextManager[str | None, sqlite3.Cursor]:
        with self.__db_cursor() as cur:
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
            cur.execute("DELETE FROM direct_link WHERE track_id=?", (track_id,))
            yield (None, cur)

    def _insert_direct_link(
        self, cur: sqlite3.Cursor, track_id: str, direct_link: str
    ) -> None:
        expired = int((time.time() + 8600) * 1e9)
        cur.execute(
            """
            INSERT INTO direct_link
            (track_id, link, expired)
            VALUES(?, ?, ?)
            ON CONFLICT(track_id) DO NOTHING
            """,
            (track_id, direct_link, expired),
        )

    def _get_inode_by_name(self, parent_inode: int, name: str) -> int:
        name = name.replace(b"\\", b"").decode()
        inode = None

        if name == ".":
            inode = parent_inode

        elif name == "..":
            inode = self._get_row(
                "SELECT parent_inode FROM inodes WHERE inode=?",
                (parent_inode,),
            )
        else:
            inode = self._get_row(
                """
                SELECT
                    inodes.id
                FROM inodes
                LEFT JOIN playlists ON playlists.id = inodes.reference_id
                LEFT JOIN tracks ON tracks.id = inodes.reference_id
                WHERE inodes.parent_inode = ?
                AND ((playlists.name = ?) OR (tracks.name = ?))
                """,
                (parent_inode, name, name),
            )
        log.debug("Inode %r by name %s", inode, name)
        return inode

    def _get_file_stat_by_inode(self, inode: int) -> FileStat:
        row = self._get_row("SELECT * FROM inodes WHERE id=?", (inode,))
        if row is None:
            raise pyfuse3.FUSEError(errno.ENOENT)

        entry = FileStat()

        entry.st_ino = inode
        entry.generation = 0
        entry.entry_timeout = 300
        entry.attr_timeout = 3000

        entry.st_mode = (
            (stat.S_IFDIR | 0o755) if row["is_dir"] else (stat.S_IFREG | 0o644)
        )
        entry.st_nlink = 1
        entry.st_uid = os.getuid()
        entry.st_gid = os.getgid()
        entry.st_rdev = 0
        entry.st_size = (
            self._get_row(
                """
            SELECT size FROM tracks
            LEFT JOIN inodes ON inodes.reference_id = tracks.id
            WHERE inodes.id=?
            """,
                (inode,),
            )
            if row["is_dir"] == 0
            else 0
        )
        entry.st_blksize = 512
        entry.st_blocks = entry.st_size // entry.st_blksize

        entry.st_atime_ns = row["atime_ns"]
        entry.st_mtime_ns = row["mtime_ns"]
        entry.st_ctime_ns = row["ctime_ns"]
        return entry

    def _invalidate_inode(self, inode: int) -> None:
        if len(self._fd_map_inode) > 0:
            return
        if inode not in self._nlookup:
            return
        if len(self._fd_map_inode) > 0:
            log.warning(
                "Invalidate inode %d skip. There are open descriptors.",
                inode,
            )
            return
        with suppress(OSError):
            pyfuse3.invalidate_inode(inode)

    def remove(self, parent_inode: int, inode: int) -> None:
        with self.__db_cursor() as cur:
            cur.execute(
                "DELETE FROM inodes WHERE id=? AND parent_inode=?",
                (inode, parent_inode),
            )

    @fail_is_exit
    async def getattr(self, inode: int, ctx=None) -> FileStat:  # noqa: ANN001, ARG002
        return self._get_file_stat_by_inode(inode)

    @fail_is_exit
    async def lookup(self, parent_inode: int, name: str, ctx=None) -> FileStat:  # noqa: ANN001, ARG002
        inode = self._get_inode_by_name(parent_inode, name)
        if inode is None:
            raise pyfuse3.FUSEError(errno.ENOENT)
        self._nlookup[inode] += 1

        return await self.getattr(inode)

    @fail_is_exit
    async def forget(self, inode_list: Sequence[tuple[int, int]]) -> None:
        for inode, count in inode_list:
            if inode not in self._nlookup:
                continue
            self._nlookup[inode] -= count
            if self._nlookup[inode] <= 0:
                self._nlookup.pop(inode, 0)

    @fail_is_exit
    async def opendir(self, inode: str, ctx=None) -> int:  # noqa: ANN001, ARG002
        fd = self._get_fd()
        self._fd_map_inode[fd].add(inode)
        return fd

    @fail_is_exit
    async def releasedir(self, fd: int) -> None:
        inodes = self._fd_map_inode.pop(fd)
        for inode in inodes:
            self.__fd_token_map_offset_read.pop(inode, None)

    @fail_is_exit
    async def statfs(self, ctx=None) -> pyfuse3.StatvfsData:  # noqa: ANN001, ARG002
        stat_ = pyfuse3.StatvfsData()

        stat_.f_bsize = 512
        stat_.f_frsize = 512

        size = self._get_row("SELECT SUM(size) FROM tracks")
        stat_.f_blocks = size // stat_.f_frsize
        stat_.f_bfree = max(size // stat_.f_frsize, 4096)
        stat_.f_bavail = stat_.f_bfree

        inodes = self._get_row("SELECT COUNT(id) FROM tracks")
        stat_.f_files = inodes
        stat_.f_ffree = 0
        stat_.f_favail = stat_.f_ffree

        return stat_

    @fail_is_exit
    async def readdir(
        self, fd: int, start_id: int, token: pyfuse3.ReaddirToken
    ) -> None:
        (dir_inode,) = self._fd_map_inode[fd]

        if dir_inode in self.__fd_token_map_offset_read:
            dir_list = self.__fd_token_map_offset_read[dir_inode]
        else:
            result = []
            names = set()
            with self.__db_cursor() as cur:
                cur.execute(
                    """
                SELECT
                    inodes.id AS ID,
                    tracks.name AS NAME
                FROM inodes
                INNER JOIN tracks ON tracks.id = inodes.reference_id
                WHERE inodes.parent_inode = ?
                    AND inodes.type = 'TRACK';
                """,
                    (dir_inode,),
                )

                for row in cur.fetchall():
                    name = row["NAME"]
                    if name in names:
                        continue
                    names.add(name)
                    result.append((name, row["id"]))
                cur.execute(
                    """
                SELECT
                    inodes.id AS ID,
                    playlists.name AS NAME
                FROM inodes
                INNER JOIN playlists ON playlists.id = inodes.reference_id
                WHERE inodes.parent_inode = ?
                    AND inodes.type = 'PLAYLIST';
                """,
                    (dir_inode,),
                )
                for row in cur.fetchall():
                    name = row["NAME"]
                    if name in names:
                        continue
                    names.add(name)
                    result.append((name, row["id"]))

            self.__fd_token_map_offset_read[dir_inode] = dir_list = [
                (".", 1),
                ("..", 1),
                *result,
            ]

        i = start_id + 1 or 0

        dir_list_slice = dir_list[i:]
        dir_list_slice.reverse()
        while len(dir_list_slice) > 0:
            name, inode = dir_list_slice.pop()
            attr = await self.getattr(inode)

            name = name.replace("/", "-").encode()
            if not pyfuse3.readdir_reply(
                token,
                name,
                attr,
                i,
            ):
                return
            i += 1
            self._nlookup[inode] += 1

    @fail_is_exit
    async def open(self, inode: int, flags: int, ctx=None) -> pyfuse3.FileInfo:  # noqa: ANN001, ARG002
        fd = self._get_fd()
        self._fd_map_inode[fd].add(inode)

        return pyfuse3.FileInfo(fh=fd)

    @fail_is_exit
    async def release(self, fd: int) -> None:
        self._fd_map_inode.pop(fd)

    @fail_is_exit
    async def unlink(self, inode_p: int, name: bytes, ctx=None) -> None:  # noqa: ANN001, ARG002
        entry = await self.lookup(inode_p, name)

        if stat.S_ISDIR(entry.st_mode):
            raise pyfuse3.FUSEError(errno.EISDIR)

        self.remove(inode_p, entry.st_ino)

    @fail_is_exit
    async def rmdir(self, inode_p: int, name: bytes, ctx=None) -> None:  # noqa: ANN001, ARG002
        entry = await self.lookup(inode_p, name)

        if not stat.S_ISDIR(entry.st_mode):
            raise pyfuse3.FUSEError(errno.ENOTDIR)

        self.remove(inode_p, entry.st_ino)

    @fail_is_exit
    async def access(self, inode: int, mode: int, ctx=None) -> bool:  # noqa: ANN001, ARG002
        return True

    def xattrs(self, inode: int) -> dict[str, Any]:
        return {
            "inode": inode,
            "nlookup": len(self._nlookup),
            "fd_map_inode": len(self._fd_map_inode),
        }

    def _to_flat_map(self, data: dict, sub_key: str = "") -> list[bytes]:
        result = []
        for key, value in data.items():
            if isinstance(value, dict):
                result.append(f"{key}:dict".encode())
                result.extend(self._to_flat_map(value, key))
            else:
                result.append(f"{sub_key}_{key}:{value}".encode())
        return result

    @fail_is_exit
    async def getxattr(self, inode: int, name: bytes, ctx=None) -> bytes:  # noqa: ANN001, ARG002
        return self.xattrs(inode).get(name.decode(), "").encode()

    @fail_is_exit
    async def listxattr(self, inode: int, ctx=None) -> Sequence[bytes]:  # noqa: ANN001, ARG002
        return self._to_flat_map(self.xattrs(inode))
