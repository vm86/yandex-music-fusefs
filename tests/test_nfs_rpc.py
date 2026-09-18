# ruff: noqa: S101
# mypy: ignore-errors

import asyncio
import struct

import pytest

from yandex_fuse.nfs.rpc import build_reply, parse_call, read_record
from yandex_fuse.nfs.xdr import XdrEncoder

NFS_PROG = 100003
NFS_VERSION = 3
SAMPLE_XID = 42


def _encode_call(*, xid: int, prog: int, vers: int, proc: int) -> bytes:
    encoder = XdrEncoder()
    encoder.uint32(xid)
    encoder.uint32(0)  # MSG_CALL
    encoder.uint32(2)  # rpcvers
    encoder.uint32(prog)
    encoder.uint32(vers)
    encoder.uint32(proc)
    encoder.uint32(0).opaque(b"")  # cred flavor + body (AUTH_NONE)
    encoder.uint32(0).opaque(b"")  # verf flavor + body
    return encoder.to_bytes()


def test_parse_call_roundtrip() -> None:
    raw = _encode_call(xid=1, prog=NFS_PROG, vers=NFS_VERSION, proc=1)
    call = parse_call(raw)

    assert call.xid == 1
    assert call.prog == NFS_PROG
    assert call.vers == NFS_VERSION
    assert call.proc == 1


def test_build_reply_has_xid_and_accepted_header() -> None:
    reply = build_reply(xid=SAMPLE_XID, body=b"\x00\x00\x00\x00")

    assert struct.unpack(">I", reply[0:4])[0] == SAMPLE_XID
    assert struct.unpack(">I", reply[4:8])[0] == 1  # MSG_REPLY
    assert struct.unpack(">I", reply[8:12])[0] == 0  # MSG_ACCEPTED


@pytest.mark.asyncio
async def test_read_record_reassembles_single_fragment() -> None:
    reader = asyncio.StreamReader()
    payload = b"hello-nfs"
    marker = struct.pack(">I", 0x80000000 | len(payload))
    reader.feed_data(marker + payload)
    reader.feed_eof()

    assert await read_record(reader) == payload
