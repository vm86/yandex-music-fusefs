from __future__ import annotations

from typing import TYPE_CHECKING

from yandex_fuse.nfs.attrs import Attr, decode_fh, encode_fattr3, encode_fh
from yandex_fuse.nfs.constants import (
    COOKIE_VERF_SIZE,
    MOUNTPROC_MNT,
    MOUNTPROC_NULL,
    MOUNTPROC_UMNT,
    NFS3_OK,
    NFS3ERR_INVAL,
    NFS3ERR_NOENT,
    NFSPROC_ACCESS,
    NFSPROC_FSINFO,
    NFSPROC_FSSTAT,
    NFSPROC_GETATTR,
    NFSPROC_LOOKUP,
    NFSPROC_NULL,
    NFSPROC_PATHCONF,
    NFSPROC_READ,
    NFSPROC_READDIR,
    NFSPROC_READDIRPLUS,
    NFSPROC_READLINK,
)
from yandex_fuse.nfs.xdr import XdrEncoder

if TYPE_CHECKING:
    from yandex_fuse.nfs.backend import NfsBackend
    from yandex_fuse.nfs.rpc import RpcCall
    from yandex_fuse.nfs.xdr import XdrDecoder

ROOT_INODE = 1

# Approximate transfer limits, reported as conservative values.
IO_BUFFER_SIZE = 65536
NAME_MAX = 255
MAX_FILE_SIZE = 2**40
TIME_DELTA_SECONDS = 1
NFS_PROPERTIES_NO_SYMLINKS = 0x0018


class Nfsv3Handler:
    def __init__(self, backend: NfsBackend) -> None:
        self._backend = backend

    async def dispatch(self, call: RpcCall) -> bytes:
        handlers = {
            NFSPROC_NULL: self._null,
            NFSPROC_GETATTR: self._getattr,
            NFSPROC_LOOKUP: self._lookup,
            NFSPROC_ACCESS: self._access,
            NFSPROC_READLINK: self._readlink,
            NFSPROC_READ: self._read,
            NFSPROC_READDIR: self._readdir,
            NFSPROC_READDIRPLUS: self._readdirplus,
            NFSPROC_FSSTAT: self._fsstat,
            NFSPROC_FSINFO: self._fsinfo,
            NFSPROC_PATHCONF: self._pathconf,
        }
        handler = handlers.get(call.proc)
        if handler is None:
            return XdrEncoder().uint32(NFS3ERR_INVAL).to_bytes()
        return await handler(call.args)

    async def _null(self, args: XdrDecoder) -> bytes:  # noqa: ARG002
        return b""

    async def _getattr(self, args: XdrDecoder) -> bytes:
        inode = _decode_fh(args)
        attr = self._backend.getattr(inode)
        encoder = XdrEncoder()
        if attr is None:
            return encoder.uint32(NFS3ERR_NOENT).to_bytes()
        encoder.uint32(NFS3_OK)
        encode_fattr3(encoder, attr)
        return encoder.to_bytes()

    async def _lookup(self, args: XdrDecoder) -> bytes:
        parent = _decode_fh(args)
        name = args.string()
        encoder = XdrEncoder()

        child = self._backend.lookup(parent, name)
        if child is None:
            return encoder.uint32(NFS3ERR_NOENT).to_bytes()

        attr = self._backend.getattr(child)
        encoder.uint32(NFS3_OK)
        encoder.opaque(encode_fh(child))
        _encode_post_op_attr(encoder, attr)
        encoder.uint32(0)  # dir_attributes not present
        return encoder.to_bytes()

    async def _access(self, args: XdrDecoder) -> bytes:
        args.opaque()  # fh -- access is always granted, fs is read-only locally
        requested = args.uint32()
        encoder = XdrEncoder()
        encoder.uint32(NFS3_OK)
        encoder.uint32(0)  # attributes not present
        encoder.uint32(requested)
        return encoder.to_bytes()

    async def _readlink(self, args: XdrDecoder) -> bytes:
        inode = _decode_fh(args)
        target = self._backend.readlink(inode)
        encoder = XdrEncoder()
        if target is None:
            return encoder.uint32(NFS3ERR_INVAL).uint32(0).to_bytes()
        encoder.uint32(NFS3_OK)
        encoder.uint32(0)  # symlink_attributes not present
        encoder.string(target.decode("utf-8", errors="replace"))
        return encoder.to_bytes()

    async def _read(self, args: XdrDecoder) -> bytes:
        inode = _decode_fh(args)
        offset = args.uint64()
        count = args.uint32()

        data = await self._backend.read(inode, offset, count)
        encoder = XdrEncoder()
        if data is None:
            return encoder.uint32(NFS3ERR_INVAL).uint32(0).to_bytes()

        attr = self._backend.getattr(inode)
        encoder.uint32(NFS3_OK)
        _encode_post_op_attr(encoder, attr)
        encoder.uint32(len(data))
        eof = attr is not None and offset + len(data) >= attr.size
        encoder.uint32(1 if eof else 0)
        encoder.opaque(data)
        return encoder.to_bytes()

    async def _readdir(self, args: XdrDecoder) -> bytes:
        inode = _decode_fh(args)
        args.uint64()  # cookie -- paging is not implemented yet
        args.uint64()  # cookieverf
        args.uint32()  # count

        entries = self._backend.readdir(inode)

        encoder = XdrEncoder()
        encoder.uint32(NFS3_OK)
        encoder.uint32(0)  # dir_attributes not present
        encoder.raw(b"\x00" * COOKIE_VERF_SIZE)

        for i, (name, child_inode) in enumerate(entries):
            encoder.uint32(1)  # value follows
            encoder.uint64(child_inode)  # fileid
            encoder.string(name)
            encoder.uint64(i + 1)  # cookie
        encoder.uint32(0)  # no more entries
        encoder.uint32(1)  # eof
        return encoder.to_bytes()

    async def _readdirplus(self, args: XdrDecoder) -> bytes:
        inode = _decode_fh(args)
        args.uint64()  # cookie
        args.uint64()  # cookieverf
        args.uint32()  # dircount
        args.uint32()  # maxcount

        entries = self._backend.readdir(inode)

        encoder = XdrEncoder()
        encoder.uint32(NFS3_OK)
        encoder.uint32(0)  # dir_attributes not present
        encoder.raw(b"\x00" * COOKIE_VERF_SIZE)

        for i, (name, child_inode) in enumerate(entries):
            child_attr = self._backend.getattr(child_inode)
            encoder.uint32(1)  # value follows
            encoder.uint64(child_inode)  # fileid
            encoder.string(name)
            encoder.uint64(i + 1)  # cookie
            _encode_post_op_attr(encoder, child_attr)
            encoder.uint32(1)  # name_handle present
            encoder.opaque(encode_fh(child_inode))
        encoder.uint32(0)  # no more entries
        encoder.uint32(1)  # eof
        return encoder.to_bytes()

    async def _fsstat(self, args: XdrDecoder) -> bytes:
        args.opaque()
        encoder = XdrEncoder()
        encoder.uint32(NFS3_OK)
        encoder.uint32(0)  # attrs not present
        for _ in range(6):  # tbytes/fbytes/abytes/tfiles/ffiles/afiles
            encoder.uint64(MAX_FILE_SIZE)
        encoder.uint32(0)  # invarsec
        return encoder.to_bytes()

    async def _fsinfo(self, args: XdrDecoder) -> bytes:
        args.opaque()
        encoder = XdrEncoder()
        encoder.uint32(NFS3_OK)
        encoder.uint32(0)  # attrs not present
        encoder.uint32(IO_BUFFER_SIZE)  # rtmax
        encoder.uint32(IO_BUFFER_SIZE)  # rtpref
        encoder.uint32(4)  # rtmult
        encoder.uint32(IO_BUFFER_SIZE)  # wtmax
        encoder.uint32(IO_BUFFER_SIZE)  # wtpref
        encoder.uint32(4)  # wtmult
        encoder.uint32(IO_BUFFER_SIZE)  # dtpref
        encoder.uint64(MAX_FILE_SIZE)  # maxfilesize
        encoder.uint32(TIME_DELTA_SECONDS)
        encoder.uint32(0)  # time_delta.nseconds
        encoder.uint32(NFS_PROPERTIES_NO_SYMLINKS)
        return encoder.to_bytes()

    async def _pathconf(self, args: XdrDecoder) -> bytes:
        args.opaque()
        encoder = XdrEncoder()
        encoder.uint32(NFS3_OK)
        encoder.uint32(0)  # attrs not present
        encoder.uint32(NAME_MAX)  # linkmax
        encoder.uint32(NAME_MAX)  # name_max
        encoder.uint32(1)  # no_trunc
        encoder.uint32(0)  # chown_restricted
        encoder.uint32(1)  # case_insensitive
        encoder.uint32(1)  # case_preserving
        return encoder.to_bytes()


class MountHandler:
    def dispatch(self, call: RpcCall) -> bytes:
        if call.proc == MOUNTPROC_NULL:
            return b""
        if call.proc == MOUNTPROC_MNT:
            call.args.string()  # path is ignored, there's always one export
            encoder = XdrEncoder()
            encoder.uint32(NFS3_OK)
            encoder.opaque(encode_fh(ROOT_INODE))
            encoder.uint32(0)  # auth flavors list length
            return encoder.to_bytes()
        if call.proc == MOUNTPROC_UMNT:
            return b""
        return XdrEncoder().uint32(0).to_bytes()


def _decode_fh(args: XdrDecoder) -> int:
    return decode_fh(args.opaque())


def _encode_post_op_attr(encoder: XdrEncoder, attr: Attr | None) -> None:
    if attr is None:
        encoder.uint32(0)
        return
    encoder.uint32(1)
    encode_fattr3(encoder, attr)
