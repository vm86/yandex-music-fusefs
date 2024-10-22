import asyncio
import faulthandler
import logging
import os
import socket
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pyfuse3
import pyfuse3_asyncio  # type: ignore[import-untyped]

from yandex_fuse.ya_music_fs import YaMusicFS

faulthandler.enable()
log = logging.getLogger(__name__)

pyfuse3_asyncio.enable()


def init_logging(*, debug: bool, systemd_run: bool) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO if not debug else logging.DEBUG)

    logging.getLogger("pyfuse3").setLevel(logging.INFO)
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
        "--debug-fuse",
        action="store_true",
        default=False,
        help="Enable FUSE debugging output",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        default=False,
        help="Run foreground",
    )
    parser.add_argument("-o", default="")
    parser.add_argument(
        "mountpoint",
        type=str,
        help="Where to mount the file system",
    )

    return parser.parse_args()


def main() -> None:
    options = parse_args()
    notify_addr = os.getenv("NOTIFY_SOCKET")

    init_logging(debug=options.debug, systemd_run=notify_addr is not None)

    fuse_options = set(pyfuse3.default_options)
    ya_music_fs = YaMusicFS()
    fuse_options.add("fsname=yandex_music")
    fuse_options.add("allow_other")

    if options.debug_fuse:
        fuse_options.add("debug")

    Path(options.mountpoint).mkdir(exist_ok=True)
    if Path(options.mountpoint).is_mount():
        raise ValueError("Is mount.")

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
    else:
        pyfuse3.init(ya_music_fs, options.mountpoint, fuse_options)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(ya_music_fs.start())
            if socket_notify is not None:
                socket_notify.sendall(b"READY=1")
            loop.run_until_complete(pyfuse3.main())
        except Exception:
            pyfuse3.close(unmount=True)
            raise
        finally:
            loop.close()

        pyfuse3.close()


if __name__ == "__main__":
    main()
