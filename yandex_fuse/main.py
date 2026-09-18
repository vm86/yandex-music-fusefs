from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import platform
import signal
import socket
from argparse import ArgumentParser, Namespace
from contextlib import suppress
from pathlib import Path

from yandex_fuse.nfs.backend import YaMusicNfsBackend
from yandex_fuse.nfs.server import serve
from yandex_fuse.ya_music_fs import YaMusicFS

faulthandler.enable()
log = logging.getLogger(__name__)

DEFAULT_PORT = 2049

# noresvport: the server doesn't check that the connection came from a
# privileged port, so mounting doesn't require root. nolock: the server
# doesn't implement NLM (rpc.lockd/statd), so network file locking must
# be explicitly disabled, or the Linux client will try to reach it.
_MOUNT_OPTIONS: dict[str, str] = {
    "Darwin": "vers=3,tcp,port={port},mountport={port},noresvport,"
    "actimeo=1,soft",
    "Linux": "vers=3,tcp,port={port},mountport={port},noresvport,nolock,"
    "actimeo=1,soft",
}


def _mount_command(host: str, port: int, mountpoint: Path) -> list[str]:
    system = platform.system()
    options = _MOUNT_OPTIONS.get(system)
    if options is None:
        raise RuntimeError(
            f"Automatic mount is not supported on {system}; "
            f"mount {host}:/ at {mountpoint} manually."
        )
    return [
        "mount",
        "-t",
        "nfs",
        "-o",
        options.format(port=port),
        f"{host}:/",
        str(mountpoint),
    ]


async def _mount(host: str, port: int, mountpoint: Path) -> None:
    await asyncio.to_thread(mountpoint.mkdir, parents=True, exist_ok=True)
    command = _mount_command(host, port, mountpoint)
    log.info("Mounting %s", mountpoint)
    proc = await asyncio.create_subprocess_exec(*command)
    if await proc.wait() != 0:
        raise RuntimeError(f"mount failed with code {proc.returncode}")


async def _unmount(mountpoint: Path) -> None:
    log.info("Unmounting %s", mountpoint)
    proc = await asyncio.create_subprocess_exec("umount", str(mountpoint))
    if await proc.wait() != 0:
        log.warning(
            "umount %s failed with code %d", mountpoint, proc.returncode
        )


def init_logging(*, debug: bool, systemd_run: bool) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO if not debug else logging.DEBUG)

    logging.getLogger("yandex_music").setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(threadName)s: [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if debug or systemd_run:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO if not debug else logging.DEBUG)
        root_logger.addHandler(handler)
    else:
        fh = logging.FileHandler(Path.home().joinpath(".cache/yandex_fuse.log"))
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO if not debug else logging.DEBUG)
        root_logger.addHandler(fh)


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debugging output",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        default=False,
        help="Run foreground",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="NFSv3 server bind address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="NFSv3 server port",
    )
    parser.add_argument(
        "--mountpoint",
        type=Path,
        default=None,
        help="Mount the server at this directory after start (macOS/Linux)",
    )

    return parser.parse_args()


async def _run(
    host: str,
    port: int,
    mountpoint: Path | None,
    socket_notify: socket.socket | None,
) -> None:
    ya_music_fs = YaMusicFS()
    await ya_music_fs.start()

    backend = YaMusicNfsBackend(ya_music_fs)
    ready = asyncio.Event()
    serve_task = asyncio.create_task(
        serve(backend, host=host, port=port, ready=ready)
    )

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    mounted_at: Path | None = None
    try:
        ready_wait = asyncio.create_task(ready.wait())
        await asyncio.wait(
            {ready_wait, serve_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if serve_task.done():
            ready_wait.cancel()
            serve_task.result()  # re-raise a bind failure, e.g. port in use
        if socket_notify is not None:
            socket_notify.sendall(b"READY=1")
        if mountpoint is not None:
            await _mount(host, port, mountpoint)
            mounted_at = mountpoint

        stop_wait = asyncio.create_task(stop.wait())
        await asyncio.wait(
            {serve_task, stop_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        if not stop_wait.done():
            stop_wait.cancel()
        if not serve_task.done():
            log.info("Shutting down...")
            serve_task.cancel()
        with suppress(asyncio.CancelledError):
            await serve_task
    finally:
        if mounted_at is not None:
            await _unmount(mounted_at)
        await ya_music_fs.aclose()


def main() -> None:
    options = parse_args()
    notify_addr = os.getenv("NOTIFY_SOCKET")

    init_logging(debug=options.debug, systemd_run=notify_addr is not None)

    socket_notify = None
    if notify_addr is None:
        child_pid = os.fork()
    else:
        child_pid = None
        socket_notify = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        if notify_addr[0] == "@":
            notify_addr = "\0" + notify_addr[1:]
        socket_notify.connect(notify_addr)

    if child_pid:
        if options.wait:
            os.waitpid(child_pid, 0)
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _run(
                options.host,
                options.port,
                options.mountpoint,
                socket_notify,
            )
        )
    finally:
        loop.close()


if __name__ == "__main__":
    main()
