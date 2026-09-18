from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from yandex_fuse.nfs.constants import MSG_ACCEPTED, MSG_CALL, MSG_REPLY
from yandex_fuse.nfs.xdr import XdrDecoder, XdrEncoder

if TYPE_CHECKING:
    import asyncio

RECORD_LAST_FRAGMENT = 0x80000000
RECORD_LENGTH_MASK = 0x7FFFFFFF


@dataclass
class RpcCall:
    xid: int
    prog: int
    vers: int
    proc: int
    args: XdrDecoder


async def read_record(reader: asyncio.StreamReader) -> bytes:
    """Assemble an RPC message from TCP record marking (RFC 5531, §10)."""
    buffer = bytearray()
    while True:
        header = await reader.readexactly(4)
        (marker,) = struct.unpack(">I", header)
        last_fragment = bool(marker & RECORD_LAST_FRAGMENT)
        length = marker & RECORD_LENGTH_MASK
        buffer += await reader.readexactly(length)
        if last_fragment:
            return bytes(buffer)


def write_record(writer: asyncio.StreamWriter, payload: bytes) -> None:
    marker = struct.pack(">I", RECORD_LAST_FRAGMENT | len(payload))
    writer.write(marker + payload)


def parse_call(raw: bytes) -> RpcCall:
    decoder = XdrDecoder(raw)
    xid = decoder.uint32()
    msg_type = decoder.uint32()
    if msg_type != MSG_CALL:
        raise ValueError(f"Unexpected RPC message type {msg_type}")
    decoder.uint32()  # rpcvers
    prog = decoder.uint32()
    vers = decoder.uint32()
    proc = decoder.uint32()

    # AUTH_SYS credential/verifier -- not parsed, access isn't checked
    # (loopback server for personal use).
    decoder.uint32()  # cred flavor
    decoder.opaque()  # cred body
    decoder.uint32()  # verf flavor
    decoder.opaque()  # verf body

    return RpcCall(xid=xid, prog=prog, vers=vers, proc=proc, args=decoder)


def build_reply(xid: int, body: bytes) -> bytes:
    encoder = XdrEncoder()
    encoder.uint32(xid)
    encoder.uint32(MSG_REPLY)
    encoder.uint32(MSG_ACCEPTED)
    encoder.uint32(0)  # verf flavor AUTH_NONE
    encoder.uint32(0)  # verf length
    encoder.uint32(0)  # accept_stat = SUCCESS
    return encoder.to_bytes() + body
