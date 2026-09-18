from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from yandex_fuse.nfs.constants import PROG_MOUNT, PROG_NFS
from yandex_fuse.nfs.procedures import MountHandler, Nfsv3Handler
from yandex_fuse.nfs.rpc import (
    build_reply,
    parse_call,
    read_record,
    write_record,
)

if TYPE_CHECKING:
    from yandex_fuse.nfs.backend import NfsBackend

log = logging.getLogger(__name__)


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    nfs_handler: Nfsv3Handler,
    mount_handler: MountHandler,
) -> None:
    peer = writer.get_extra_info("peername")
    log.debug("NFS client connected: %r", peer)
    try:
        while True:
            try:
                raw = await read_record(reader)
            except asyncio.IncompleteReadError:
                return

            call = parse_call(raw)
            if call.prog == PROG_MOUNT:
                body = mount_handler.dispatch(call)
            elif call.prog == PROG_NFS:
                body = await nfs_handler.dispatch(call)
            else:
                log.warning("Unknown RPC program %d", call.prog)
                body = b""

            write_record(writer, build_reply(call.xid, body))
            await writer.drain()
    finally:
        writer.close()
        log.debug("NFS client disconnected: %r", peer)


async def serve(
    backend: NfsBackend,
    host: str = "127.0.0.1",
    port: int = 2049,
    *,
    ready: asyncio.Event | None = None,
) -> None:
    """Start the NFSv3 server (MOUNT + NFS programs on the same port)."""
    nfs_handler = Nfsv3Handler(backend)
    mount_handler = MountHandler()

    async def on_client(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await _handle_client(reader, writer, nfs_handler, mount_handler)

    server = await asyncio.start_server(on_client, host, port)
    log.info("NFSv3 server listening on %s:%d", host, port)
    if ready is not None:
        ready.set()
    async with server:
        await server.serve_forever()
