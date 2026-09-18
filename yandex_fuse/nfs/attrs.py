from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from yandex_fuse.nfs.constants import NF3DIR, NF3LNK

if TYPE_CHECKING:
    from yandex_fuse.nfs.xdr import XdrEncoder

NS_PER_SEC = 1_000_000_000


@dataclass
class Attr:
    inode: int
    kind: int  # NF3REG / NF3DIR / NF3LNK
    mode: int
    size: int
    mtime_ns: int
    nlink: int = 1

    @property
    def is_dir(self) -> bool:
        return self.kind == NF3DIR

    @property
    def is_symlink(self) -> bool:
        return self.kind == NF3LNK


def encode_fattr3(encoder: XdrEncoder, attr: Attr) -> None:
    seconds, nseconds = divmod(attr.mtime_ns, NS_PER_SEC)

    encoder.uint32(attr.kind)
    encoder.uint32(attr.mode)
    encoder.uint32(attr.nlink)
    encoder.uint32(0)  # uid
    encoder.uint32(0)  # gid
    encoder.uint64(attr.size)
    encoder.uint64(attr.size)  # used
    encoder.uint32(0)  # rdev.specdata1
    encoder.uint32(0)  # rdev.specdata2
    encoder.uint64(0)  # fsid
    encoder.uint64(attr.inode)  # fileid
    for _ in range(3):  # atime, mtime, ctime: nfstime3 { seconds; nseconds }
        encoder.uint32(seconds)
        encoder.uint32(nseconds)


def encode_fh(inode: int) -> bytes:
    return struct.pack(">Q", inode)


def decode_fh(data: bytes) -> int:
    return int(struct.unpack(">Q", data[:8])[0])
