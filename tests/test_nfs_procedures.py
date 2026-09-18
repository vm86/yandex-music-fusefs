# ruff: noqa: S101
# mypy: ignore-errors

import pytest

from yandex_fuse.nfs.attrs import Attr, decode_fh, encode_fh
from yandex_fuse.nfs.constants import (
    MOUNTPROC_MNT,
    NF3DIR,
    NF3REG,
    NFS3_OK,
    NFS3ERR_NOENT,
    NFSPROC_GETATTR,
    NFSPROC_LOOKUP,
    NFSPROC_READ,
    NFSPROC_READDIR,
)
from yandex_fuse.nfs.procedures import MountHandler, Nfsv3Handler
from yandex_fuse.nfs.rpc import RpcCall
from yandex_fuse.nfs.xdr import XdrDecoder, XdrEncoder

ROOT_INODE = 1
TRACK_INODE = 2
TRACK_CONTENT = b"data"


class FakeBackend:
    """A two-file in-memory tree: /track.mp3, for dispatcher tests."""

    def __init__(self) -> None:
        self._attrs = {
            ROOT_INODE: Attr(
                inode=ROOT_INODE,
                kind=NF3DIR,
                mode=0o755,
                size=0,
                mtime_ns=0,
                nlink=2,
            ),
            TRACK_INODE: Attr(
                inode=TRACK_INODE,
                kind=NF3REG,
                mode=0o644,
                size=len(TRACK_CONTENT),
                mtime_ns=0,
            ),
        }

    def getattr(self, inode: int) -> Attr | None:
        return self._attrs.get(inode)

    def lookup(self, parent_inode: int, name: str) -> int | None:
        if parent_inode == ROOT_INODE and name == "track.mp3":
            return TRACK_INODE
        return None

    def readdir(self, inode: int) -> list[tuple[str, int]]:
        return [("track.mp3", TRACK_INODE)] if inode == ROOT_INODE else []

    def readlink(self, inode: int) -> bytes | None:  # noqa: ARG002
        return None

    async def read(self, inode: int, offset: int, size: int) -> bytes | None:
        if inode != TRACK_INODE:
            return None
        return TRACK_CONTENT[offset : offset + size]


def _skip_fattr3(decoder: XdrDecoder) -> None:
    for _ in range(5):  # type, mode, nlink, uid, gid
        decoder.uint32()
    decoder.uint64()  # size
    decoder.uint64()  # used
    decoder.uint32()  # rdev.specdata1
    decoder.uint32()  # rdev.specdata2
    decoder.uint64()  # fsid
    decoder.uint64()  # fileid
    for _ in range(3):  # atime, mtime, ctime
        decoder.uint32()
        decoder.uint32()


def _call(proc: int, args: XdrEncoder) -> RpcCall:
    return RpcCall(
        xid=1,
        prog=100003,
        vers=3,
        proc=proc,
        args=XdrDecoder(args.to_bytes()),
    )


@pytest.mark.asyncio
async def test_getattr_known_inode() -> None:
    handler = Nfsv3Handler(FakeBackend())
    args = XdrEncoder().opaque(encode_fh(TRACK_INODE))

    reply = XdrDecoder(await handler.dispatch(_call(NFSPROC_GETATTR, args)))

    assert reply.uint32() == NFS3_OK


@pytest.mark.asyncio
async def test_getattr_missing_inode() -> None:
    handler = Nfsv3Handler(FakeBackend())
    args = XdrEncoder().opaque(encode_fh(999))

    reply = XdrDecoder(await handler.dispatch(_call(NFSPROC_GETATTR, args)))

    assert reply.uint32() == NFS3ERR_NOENT


@pytest.mark.asyncio
async def test_lookup_finds_track() -> None:
    handler = Nfsv3Handler(FakeBackend())
    args = XdrEncoder().opaque(encode_fh(ROOT_INODE)).string("track.mp3")

    reply = XdrDecoder(await handler.dispatch(_call(NFSPROC_LOOKUP, args)))

    assert reply.uint32() == NFS3_OK
    assert decode_fh(reply.opaque()) == TRACK_INODE
    assert reply.uint32() == 1  # obj_attributes present
    _skip_fattr3(reply)
    assert reply.uint32() == 0  # dir_attributes not present


@pytest.mark.asyncio
async def test_readdir_lists_track() -> None:
    handler = Nfsv3Handler(FakeBackend())
    args = (
        XdrEncoder()
        .opaque(encode_fh(ROOT_INODE))
        .uint64(0)
        .uint64(0)
        .uint32(4096)
    )

    reply = XdrDecoder(await handler.dispatch(_call(NFSPROC_READDIR, args)))

    assert reply.uint32() == NFS3_OK
    assert reply.uint32() == 0  # dir_attributes not present
    reply.uint64()  # cookieverf3, value doesn't matter
    assert reply.uint32() == 1  # value follows
    assert reply.uint64() == TRACK_INODE
    assert reply.string() == "track.mp3"
    reply.uint64()  # cookie
    assert reply.uint32() == 0  # no more entries
    assert reply.uint32() == 1  # eof


@pytest.mark.asyncio
async def test_read_returns_requested_bytes() -> None:
    handler = Nfsv3Handler(FakeBackend())
    args = XdrEncoder().opaque(encode_fh(TRACK_INODE)).uint64(0).uint32(4)

    reply = XdrDecoder(await handler.dispatch(_call(NFSPROC_READ, args)))

    assert reply.uint32() == NFS3_OK
    assert reply.uint32() == 1  # post_op_attr present
    _skip_fattr3(reply)
    assert reply.uint32() == len(TRACK_CONTENT)  # count
    assert reply.uint32() == 1  # eof
    assert reply.opaque() == TRACK_CONTENT


def test_mount_returns_root_filehandle() -> None:
    handler = MountHandler()
    args = XdrEncoder().string("/")
    call = RpcCall(
        xid=1,
        prog=100005,
        vers=3,
        proc=MOUNTPROC_MNT,
        args=XdrDecoder(args.to_bytes()),
    )

    reply = XdrDecoder(handler.dispatch(call))

    assert reply.uint32() == NFS3_OK
    assert decode_fh(reply.opaque()) == ROOT_INODE
