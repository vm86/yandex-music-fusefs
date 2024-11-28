from __future__ import annotations

import errno
import json
import logging
import os
import sqlite3
import stat
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from pyfuse3 import (
    ROOT_INODE,
    EntryAttributes,
    FileHandleT,
    FileInfo,
    FileNameT,
    FlagT,
    FUSEError,
    InodeT,
    ModeT,
    Operations,
    ReaddirToken,
    RequestContext,
    StatvfsData,
    XAttrNameT,
    invalidate_inode,
    readdir_reply,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Coroutine,
        Iterator,
        Sequence,
    )

log = logging.getLogger(__name__)


class FileStat(EntryAttributes):
    def __init__(self) -> None:
        self.st_mode = ModeT(0)
        self.st_ino = InodeT(0)
        self.st_dev = 0
        self.st_nlink = 0
        self.st_uid = 0
        self.st_gid = 0
        self.st_size = 0
        self.st_atime = 0
        self.st_mtime = 0
        self.st_ctime = 0


P = ParamSpec("P")
T = TypeVar("T")

ROW_DICT_TYPE = dict[str, Any] | None


# Если возникает исключение в FUSE, то вывоз зависает.
def fail_is_exit(
    func: Callable[..., Coroutine[Any, Any, T]],
) -> Callable[..., Coroutine[Any, Any, T]]:
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except FUSEError as error:
            log.debug("FUSEError %r: %s", func, str(error))
            raise
        except Exception:
            log.exception("Error %r", func)
            sys.exit(42)

    return wrapped


@dataclass
class SQLRow:
    __tablename__ = ""

    def insert(self) -> tuple[str, dict[str, Any]]:
        data = self.__dict__
        if data.get("id") is None:
            data.pop("id", None)

        columns = ", ".join(data.keys())
        placeholders = ":" + ", :".join(data.keys())
        query = f"""
        INSERT INTO {self.__tablename__}
        ({columns})
        VALUES
        ({placeholders})
        """
        return query, data

    @classmethod
    def from_row(cls: type[T], row: dict[str, Any] | None) -> T | None:
        if row is None:
            return None
        return cls(**row)


@dataclass
class Inode(SQLRow):
    __tablename__ = "inodes"

    uid: int
    gid: int
    mode: ModeT
    mtime_ns: int
    atime_ns: int
    ctime_ns: int
    target: bytes
    size: int = 0
    rdev: int = 0

    id: int | None = None


@dataclass
class Dentry(SQLRow):
    __tablename__ = "dentrys"

    name: bytes
    inode: InodeT
    parent_inode: InodeT
    rowid: int | None = None
    data: bytes = b""


class VirtFS(Operations):
    FILE_DB = "file::memory:?cache=shared"

    CREATE_TABLE_QUERYS: tuple[str, ...] = (
        """
        PRAGMA foreign_keys=ON;
        """,
        """
        CREATE TABLE IF NOT EXISTS inodes (
            id        INTEGER PRIMARY KEY,
            uid       INT NOT NULL,
            gid       INT NOT NULL,
            mode      INT NOT NULL,
            mtime_ns  INT NOT NULL,
            atime_ns  INT NOT NULL,
            ctime_ns  INT NOT NULL,
            target    BLOB(256),
            size      INT NOT NULL DEFAULT 0,
            rdev      INT NOT NULL DEFAULT 0
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dentrys (
            rowid     INTEGER PRIMARY KEY AUTOINCREMENT,
            name      BLOB(256) NOT NULL,
            inode     INT NOT NULL
                REFERENCES inodes(id) ON DELETE CASCADE,
            parent_inode INT NOT NULL
                REFERENCES inodes(id) ON DELETE RESTRICT,
            data      BLOB,
            UNIQUE (name, parent_inode)
        );
        """,
        """
        INSERT OR IGNORE INTO inodes VALUES (1,0,0,16877,0,0,0,X'',0,0);
        """,
        """
        INSERT OR IGNORE INTO dentrys VALUES(1,X'2e2e',1,1,X'');
        """,
    )

    ROOT_INODE = ROOT_INODE

    def __init__(self) -> None:
        super().__init__()

        self._db = sqlite3.connect(
            self.FILE_DB, isolation_level="IMMEDIATE", uri=True
        )
        self._db.row_factory = sqlite3.Row

        self._fd_map_inode: dict[int, InodeT] = {}
        self._nlookup: dict[InodeT, int] = defaultdict(int)

        self.__fd_token_read: dict[InodeT, list[tuple[bytes, InodeT]]] = (
            defaultdict(list)
        )
        self.__later_invalidate_inode: set[InodeT] = set()

        self._open_cur = 0
        self.__init_table()

    @contextmanager
    def _db_cursor(self) -> Iterator[sqlite3.Cursor]:
        self._open_cur += 1
        try:
            with self._db:
                yield self._db.cursor()
        finally:
            self._open_cur -= 1

    def __init_table(self) -> None:
        with self._db_cursor() as cur:
            log.debug("Init database.")

            for create_table_query in self.CREATE_TABLE_QUERYS:
                cur.execute(create_table_query)

    def _get_int_row(self, query: str, *params: tuple[str | int, ...]) -> int:
        with self._db_cursor() as cur:
            cur.execute(query, *params)
            result = cur.fetchone()
            if result:
                return int(*result)
            return 0

    def _get_dict_row(
        self, query: str, *params: tuple[Any, ...]
    ) -> ROW_DICT_TYPE:
        with self._db_cursor() as cur:
            cur.execute(query, *params)
            result = cur.fetchone()
            if result:
                return dict(result)
            return None

    def _get_list_row(
        self, query: str, *params: tuple[str | int, ...]
    ) -> list[dict[str, Any]]:
        with self._db_cursor() as cur:
            cur.execute(query, *params)
            return [dict(row) for row in cur.fetchall()]

    def _get_fd(self) -> int:
        try:
            return max(self._fd_map_inode.keys()) + 1
        except ValueError:
            return 1

    @property
    def _inode_map_fd(self) -> dict[InodeT, set[int]]:
        result = defaultdict(set)
        for fd, inode in self._fd_map_inode.items():
            result[inode].add(fd)
        return result

    def _get_file_stat_by_inode(self, inode: InodeT) -> FileStat:
        inode_row = Inode.from_row(
            self._get_dict_row("SELECT * FROM inodes WHERE id=?", (inode,))
        )

        if inode_row is None:
            raise FUSEError(errno.ENOENT)

        entry = FileStat()

        entry.st_ino = inode
        entry.generation = 0
        entry.entry_timeout = 300
        entry.attr_timeout = 3000

        entry.st_mode = inode_row.mode

        entry.st_nlink = self._get_int_row(
            "SELECT COUNT(inode) FROM dentrys WHERE inode=?", (inode,)
        )

        entry.st_uid = os.getuid()
        entry.st_gid = os.getgid()
        entry.st_rdev = 0
        entry.st_size = inode_row.size
        entry.st_blksize = 512
        entry.st_blocks = entry.st_size // entry.st_blksize

        entry.st_atime_ns = inode_row.atime_ns
        entry.st_mtime_ns = inode_row.mtime_ns
        entry.st_ctime_ns = inode_row.ctime_ns
        return entry

    def _get_inode_by_name(
        self, parent_inode: InodeT, name: bytes
    ) -> InodeT | None:
        str_name = name.replace(b"\\", b"")
        inode: InodeT | None = None

        if str_name == b".":
            return parent_inode

        if str_name == b"..":
            dentry = Dentry.from_row(
                self._get_dict_row(
                    "SELECT * FROM dentrys WHERE inode=?", (parent_inode,)
                )
            )
            if dentry is not None:
                return dentry.parent_inode
        dentry = Dentry.from_row(
            self._get_dict_row(
                "SELECT * FROM dentrys WHERE name=? AND parent_inode=?",
                (
                    name,
                    parent_inode,
                ),
            )
        )
        if dentry:
            log.debug("Inode %s by name %s", inode, str_name.decode())
            return dentry.inode
        log.debug("Inode by name %s not found.", str_name.decode())
        return None

    @property
    def queue_later_invalidate_inode(self) -> set[InodeT]:
        return self.__later_invalidate_inode

    def _invalidate_inode(self, inode: InodeT) -> None:
        self.__later_invalidate_inode.discard(inode)
        if inode not in self._nlookup:
            return
        if inode in self._inode_map_fd:
            log.warning(
                "Invalidate inode %d skip. There are open descriptors.",
                inode,
            )
            self.__later_invalidate_inode.add(inode)
            return
        with suppress(OSError):
            invalidate_inode(inode)

    def _create(
        self,
        *,
        parent_inode: InodeT,
        name: bytes,
        size: int,
        mode: int,
        target: bytes,
        db_cursor: sqlite3.Cursor,
    ) -> InodeT:
        now_ns = int(time.time() * 1e9)

        inode_object = Inode(
            mode=ModeT(mode),
            target=target,
            uid=0,
            gid=0,
            size=size,
            mtime_ns=now_ns,
            atime_ns=now_ns,
            ctime_ns=now_ns,
        )
        db_cursor.execute(*inode_object.insert())
        if db_cursor.lastrowid is None:
            raise RuntimeError("Lastrowid is none!")

        inode = InodeT(db_cursor.lastrowid)
        dentry_object = Dentry(
            parent_inode=parent_inode,
            inode=inode,
            name=name,
        )
        db_cursor.execute(*dentry_object.insert())
        if stat.S_ISDIR(inode_object.mode):
            dentry_object_dot_dot = Dentry(
                parent_inode=inode,
                inode=inode,
                name=b"..",
            )
            db_cursor.execute(*dentry_object_dot_dot.insert())

        return inode

    def remove(self, parent_inode: InodeT, inode: InodeT) -> bool:
        with self._db_cursor() as cur:
            cur.execute(
                "DELETE FROM dentrys WHERE inode=? AND parent_inode=?",
                (inode, parent_inode),
            )

            try:
                st_link = self._get_file_stat_by_inode(inode).st_nlink
            except FUSEError:
                st_link = 0

            if st_link == 0 and inode not in self._inode_map_fd:
                cur.execute("DELETE FROM inodes WHERE id=?", (inode,))
                return True
        return False

    @fail_is_exit
    async def getattr(
        self,
        inode: InodeT,
        ctx: RequestContext,  # noqa: ARG002
    ) -> EntryAttributes:
        return self._get_file_stat_by_inode(inode)

    @fail_is_exit
    async def lookup(
        self,
        parent_inode: InodeT,
        name: FileNameT,
        ctx: RequestContext,
    ) -> EntryAttributes:
        inode = self._get_inode_by_name(parent_inode, name)
        if inode is None:
            raise FUSEError(errno.ENOENT)
        self._nlookup[inode] += 1

        return await self.getattr(inode, ctx)

    @fail_is_exit
    async def forget(self, inode_list: Sequence[tuple[InodeT, int]]) -> None:
        for inode, count in inode_list:
            if inode not in self._nlookup:
                continue
            self._nlookup[inode] -= count
            if self._nlookup[inode] <= 0:
                self._nlookup.pop(inode, 0)

    @fail_is_exit
    async def opendir(self, inode: InodeT, ctx: RequestContext) -> FileHandleT:  # noqa: ARG002
        fd = self._get_fd()
        self._fd_map_inode[fd] = inode
        return FileHandleT(fd)

    @fail_is_exit
    async def releasedir(self, fd: FileHandleT) -> None:
        inode = self._fd_map_inode.pop(fd)
        self.__fd_token_read.pop(inode, None)
        if inode in self.__later_invalidate_inode:
            self._invalidate_inode(inode)

    @fail_is_exit
    async def statfs(self, ctx: RequestContext) -> StatvfsData:  # noqa: ARG002
        stat = StatvfsData()

        stat.f_bsize = 512
        stat.f_frsize = 512

        max_size = 2**40
        size = self._get_int_row("SELECT SUM(size) FROM inodes")

        stat.f_blocks = max_size // stat.f_bsize
        stat.f_bfree = (max_size - size) // stat.f_frsize
        stat.f_bavail = stat.f_bfree

        max_inodes = 2**32
        inodes = self._get_int_row("SELECT COUNT(id) FROM inodes")

        stat.f_files = max_inodes
        stat.f_ffree = max_inodes - inodes
        stat.f_favail = stat.f_ffree

        return stat

    @fail_is_exit
    async def readdir(
        self, fd: int, start_id: int, token: ReaddirToken
    ) -> None:
        dir_inode = self._fd_map_inode[fd]

        if dir_inode in self.__fd_token_read:
            dir_list = self.__fd_token_read[dir_inode]
        else:
            result = []
            names = set()
            with self._db_cursor() as cur:
                cur.execute(
                    """
                SELECT
                    *
                FROM dentrys
                WHERE parent_inode = ?
                """,
                    (dir_inode,),
                )
                for row in cur.fetchall():
                    name = row["name"]
                    if name in names:
                        continue
                    if isinstance(name, str):
                        name = name.encode()

                    names.add(name)
                    result.append((name, row["inode"]))

            self.__fd_token_read[dir_inode] = dir_list = [
                (b".", InodeT(1)),
                *result,
            ]

        i = start_id + 1 or 0

        dir_list_slice = dir_list[i - 1 :]
        dir_list_slice.reverse()
        while len(dir_list_slice) > 0:
            name, inode = dir_list_slice.pop()
            attr = self._get_file_stat_by_inode(inode)

            if not readdir_reply(
                token,
                name,
                attr,
                i,
            ):
                return
            i += 1
            self._nlookup[inode] += 1

    @fail_is_exit
    async def open(
        self,
        inode: InodeT,
        flags: FlagT,  # noqa: ARG002
        ctx: RequestContext,  # noqa: ARG002
    ) -> FileInfo:
        fd = self._get_fd()
        self._fd_map_inode[fd] = inode

        return FileInfo(fh=FileHandleT(fd))

    @fail_is_exit
    async def release(self, fd: int) -> None:
        inode = self._fd_map_inode.pop(fd)
        if self._get_file_stat_by_inode(inode).st_nlink == 0:
            with self._db_cursor() as cur:
                cur.execute("DELETE FROM inodes WHERE id=?", (inode,))
        if inode in self.__later_invalidate_inode:
            self._invalidate_inode(inode)

    @fail_is_exit
    async def unlink(
        self,
        parent_inode: InodeT,
        name: FileNameT,
        ctx: RequestContext,
    ) -> None:
        entry = await self.lookup(parent_inode, name, ctx)

        if stat.S_ISDIR(entry.st_mode):
            raise FUSEError(errno.EISDIR)

        self.remove(parent_inode, entry.st_ino)

    @fail_is_exit
    async def rmdir(
        self,
        parent_inode: InodeT,
        name: FileNameT,
        ctx: RequestContext,  # noqa: ARG002
    ) -> None:
        entry = await self.lookup(parent_inode, name)

        if not stat.S_ISDIR(entry.st_mode):
            raise FUSEError(errno.ENOTDIR)

        self.remove(parent_inode, entry.st_ino)

    @fail_is_exit
    async def access(
        self,
        inode: InodeT,  # noqa: ARG002
        mode: ModeT,  # noqa: ARG002
        ctx: RequestContext,  # noqa: ARG002
    ) -> bool:
        return True

    @fail_is_exit
    async def symlink(
        self,
        parent_inode: InodeT,
        name: bytes,
        target: bytes,
        ctx: RequestContext,
    ) -> EntryAttributes:
        target_dentry = Dentry.from_row(
            self._get_dict_row(
                "SELECT * FROM dentrys WHERE name=?",
                (target,),
            )
        )
        if target_dentry is None:
            raise FUSEError(errno.ENOENT)

        with self._db_cursor() as cur:
            inode = self._create(
                parent_inode=parent_inode,
                name=name,
                size=4096,
                mode=(stat.S_IFLNK | 0o777),
                target=target,
                db_cursor=cur,
            )
        return await self.getattr(inode, ctx)

    @fail_is_exit
    async def readlink(self, inode: InodeT, ctx: RequestContext) -> FileNameT:  # noqa: ARG002
        row = Inode.from_row(
            self._get_dict_row("SELECT * FROM inodes WHERE id=?", (inode,))
        )
        if row is None:
            raise FUSEError(errno.EINVAL)
        if not stat.S_ISLNK(row.mode):
            raise FUSEError(errno.EINVAL)
        return FileNameT(row.target)

    def xattrs(self, inode: InodeT) -> dict[str, Any]:
        return {
            "inode": inode,
            "nlookup": len(self._nlookup),
            "fd_map_inode": len(self._fd_map_inode),
        }

    def _to_flat_map(self, data: dict[str, Any]) -> Sequence[XAttrNameT]:
        result = []

        def _set_default(obj: T) -> T | list[Any]:
            if isinstance(obj, set):
                return list(obj)
            return obj

        for key, value in data.items():
            data_json = json.dumps(value, default=_set_default)
            result.append(XAttrNameT(f"{key}:{data_json}".encode()))
        return result

    @fail_is_exit
    async def getxattr(
        self,
        inode: InodeT,
        name: XAttrNameT,
        ctx: RequestContext,  # noqa: ARG002
    ) -> bytes:
        value = self.xattrs(inode).get(name.decode(), "")

        result, *unused = self._to_flat_map(value)
        return result

    @fail_is_exit
    async def listxattr(
        self,
        inode: InodeT,
        ctx: RequestContext,  # noqa: ARG002
    ) -> Sequence[XAttrNameT]:
        return self._to_flat_map(self.xattrs(inode))
