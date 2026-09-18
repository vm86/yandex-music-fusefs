from __future__ import annotations

import logging
import os
import sqlite3
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NewType

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)

InodeT = NewType("InodeT", int)
ROOT_INODE = InodeT(1)

ROW_DICT_TYPE = dict[str, Any] | None


@dataclass
class Stat:
    st_mode: int = 0
    st_ino: InodeT = field(default_factory=lambda: InodeT(0))
    st_nlink: int = 0
    st_uid: int = 0
    st_gid: int = 0
    st_size: int = 0
    st_atime_ns: int = 0
    st_mtime_ns: int = 0
    st_ctime_ns: int = 0


@dataclass
class SQLRow:
    __tablename__ = ""

    def insert(self) -> tuple[str, dict[str, Any]]:
        data = self.__dict__
        if data.get("id") is None:
            data.pop("id", None)

        columns = ", ".join(data.keys())
        placeholders = ":" + ", :".join(data.keys())
        # Table and column names come from dataclass fields, not from user
        # input -- values are passed separately, as bound placeholders.
        query = f"""
        INSERT INTO {self.__tablename__}
        ({columns})
        VALUES
        ({placeholders})
        """  # noqa: S608
        return query, data

    @classmethod
    def from_row(cls, row: dict[str, Any] | None) -> Self | None:
        if row is None:
            return None
        return cls(**row)


@dataclass
class Inode(SQLRow):
    __tablename__ = "inodes"

    uid: int
    gid: int
    mode: int
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


class VirtFS:
    """SQLite-backed inode/dentry storage -- protocol-agnostic backend.

    This used to be a pyfuse3.Operations (FUSE) subclass; now the only
    consumer is the NFSv3 server (see yandex_fuse/nfs/).
    """

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

    def _get_file_stat_by_inode(self, inode: InodeT) -> Stat | None:
        inode_row = Inode.from_row(
            self._get_dict_row("SELECT * FROM inodes WHERE id=?", (inode,))
        )

        if inode_row is None:
            return None

        entry = Stat()

        entry.st_ino = inode
        entry.st_mode = inode_row.mode

        entry.st_nlink = self._get_int_row(
            "SELECT COUNT(inode) FROM dentrys WHERE inode=?", (inode,)
        )

        entry.st_uid = os.getuid()
        entry.st_gid = os.getgid()
        entry.st_size = inode_row.size

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

    def _query_dentrys(self, dir_inode: InodeT) -> list[tuple[bytes, InodeT]]:
        result: list[tuple[bytes, InodeT]] = []
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
        return result

    def stat_by_inode(self, inode: InodeT) -> Stat | None:
        return self._get_file_stat_by_inode(inode)

    def lookup_by_name(
        self, parent_inode: InodeT, name: bytes
    ) -> InodeT | None:
        return self._get_inode_by_name(parent_inode, name)

    def dir_entries(self, inode: InodeT) -> list[tuple[bytes, InodeT]]:
        """Directory entries, including "." and "..", for NFS READDIR."""
        parent_inode = self._get_inode_by_name(inode, b"..") or inode
        return [
            (b".", inode),
            (b"..", parent_inode),
            *self._query_dentrys(inode),
        ]

    def target_of(self, inode: InodeT) -> bytes | None:
        """Symlink target by inode, or None if it's not a symlink."""
        row = Inode.from_row(
            self._get_dict_row("SELECT * FROM inodes WHERE id=?", (inode,))
        )
        if row is None or not stat.S_ISLNK(row.mode):
            return None
        return row.target

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
            mode=mode,
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
        """Remove the dentry, and the inode itself if it was the last link.

        The FUSE version kept the inode alive while a file descriptor was
        open (protecting a file being read from deletion). Stateless NFS
        reads have no notion of an open descriptor, so there's no such
        protection here: deleting a track that is currently being read
        can interrupt the read.
        """
        with self._db_cursor() as cur:
            cur.execute(
                "DELETE FROM dentrys WHERE inode=? AND parent_inode=?",
                (inode, parent_inode),
            )

            entry = self._get_file_stat_by_inode(inode)
            st_link = entry.st_nlink if entry is not None else 0

            if st_link == 0:
                cur.execute("DELETE FROM inodes WHERE id=?", (inode,))
                return True
        return False
