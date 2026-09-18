from __future__ import annotations

import stat
import time
from typing import TYPE_CHECKING, Protocol

from yandex_fuse.nfs.attrs import Attr
from yandex_fuse.nfs.constants import NF3DIR, NF3LNK, NF3REG
from yandex_fuse.virt_fs import ROOT_INODE, InodeT

if TYPE_CHECKING:
    from yandex_fuse.ya_music_fs import YaMusicFS

# A virtual file at the root, shown only while there's no auth token yet
# (see YaMusicFS.needs_auth and auth_page) -- it has no row in the inodes
# table, so its number is picked well outside the autoincrement range.
AUTH_PAGE_NAME = "index.html"
AUTH_PAGE_INODE = InodeT(2**31)
AUTH_PAGE_MODE = 0o444


class NfsBackend(Protocol):
    def getattr(self, inode: int) -> Attr | None: ...

    def lookup(self, parent_inode: int, name: str) -> int | None: ...

    def readdir(self, inode: int) -> list[tuple[str, int]]: ...

    def readlink(self, inode: int) -> bytes | None: ...

    async def read(
        self, inode: int, offset: int, size: int
    ) -> bytes | None: ...


def _attr_kind(mode: int) -> int:
    if stat.S_ISDIR(mode):
        return NF3DIR
    if stat.S_ISLNK(mode):
        return NF3LNK
    return NF3REG


class YaMusicNfsBackend:
    """NfsBackend adapter on top of the existing YaMusicFS/VirtFS."""

    def __init__(self, music_fs: YaMusicFS) -> None:
        self._fs = music_fs

    def _auth_page_attr(self) -> Attr | None:
        if not self._fs.needs_auth:
            return None
        content = self._fs.auth_page()
        return Attr(
            inode=AUTH_PAGE_INODE,
            kind=NF3REG,
            mode=AUTH_PAGE_MODE,
            size=len(content),
            mtime_ns=int(time.time() * 1e9),
        )

    def getattr(self, inode: int) -> Attr | None:
        if inode == AUTH_PAGE_INODE:
            return self._auth_page_attr()

        entry = self._fs.stat_by_inode(InodeT(inode))
        if entry is None:
            return None
        return Attr(
            inode=inode,
            kind=_attr_kind(entry.st_mode),
            mode=entry.st_mode & 0o777,
            size=entry.st_size,
            mtime_ns=entry.st_mtime_ns,
            nlink=entry.st_nlink,
        )

    def lookup(self, parent_inode: int, name: str) -> int | None:
        if (
            parent_inode == ROOT_INODE
            and name == AUTH_PAGE_NAME
            and self._fs.needs_auth
        ):
            return AUTH_PAGE_INODE

        child = self._fs.lookup_by_name(
            InodeT(parent_inode), name.encode("utf-8")
        )
        return None if child is None else int(child)

    def readdir(self, inode: int) -> list[tuple[str, int]]:
        entries = [
            (name.decode("utf-8", errors="replace"), int(child_inode))
            for name, child_inode in self._fs.dir_entries(InodeT(inode))
        ]
        if inode == ROOT_INODE and self._fs.needs_auth:
            entries.append((AUTH_PAGE_NAME, AUTH_PAGE_INODE))
        return entries

    def readlink(self, inode: int) -> bytes | None:
        if inode == AUTH_PAGE_INODE:
            return None
        return self._fs.target_of(InodeT(inode))

    async def read(self, inode: int, offset: int, size: int) -> bytes | None:
        if inode == AUTH_PAGE_INODE:
            if not self._fs.needs_auth:
                return None
            content = self._fs.auth_page()
            return content[offset : offset + size]
        return await self._fs.read_track(InodeT(inode), offset, size)
