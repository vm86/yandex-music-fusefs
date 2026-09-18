# Yandex Music FuseFS

## Description

Built for personal use, to listen to Yandex Music in MPD.

It used to mount via FUSE (pyfuse3); now `yamusic-fs` is an NFSv3
server: it serves playlists and tracks over NFS on a local port, and
mounting is done with the standard `mount` command (you can run it
yourself, or let `yamusic-fs` do it via `--mountpoint`).
No system dependencies (fuse3/libfuse-dev) are required anymore -- an
NFS client ships out of the box with any modern OS.

### Installation

[Download](https://github.com/vm86/yandex-music-fusefs/releases)

``` shell
pip install yandex_fuse-*.tar.gz
```

### Getting started

#### Start the server and mount it in one command

```shell
yamusic-fs --mountpoint ~/Music/Yandex
```

The `~/Music/Yandex` directory is created if needed, and once the
server is up, NFS is mounted there automatically (the correct `mount`
command is picked per platform -- macOS/Linux). Stopping the server
(`SIGINT`/`SIGTERM`) unmounts the directory automatically.

By default the server listens on `127.0.0.1:2049`. To use a different
address/port:

```shell
yamusic-fs --host 127.0.0.1 --port 2049 --mountpoint ~/Music/Yandex
```

#### Start the server and mount it manually

```shell
yamusic-fs
mkdir -p ~/Music/Yandex
mount -t nfs -o vers=3,tcp,port=2049,mountport=2049,noresvport,actimeo=1,soft \
    127.0.0.1:/ ~/Music/Yandex
```

On Linux you additionally need the `nolock` option -- the server
doesn't implement NLM (`rpc.lockd`/`rpc.statd`):

```shell
mount -t nfs -o vers=3,tcp,port=2049,mountport=2049,noresvport,nolock,actimeo=1,soft \
    127.0.0.1:/ ~/Music/Yandex
```

#### First run

Until there's a token, the mounted directory's root contains a virtual
`index.html` -- open it in a browser: it has a link to the Yandex login
page and a code to enter there (OAuth Device Flow). Once confirmed,
`index.html` disappears, the logs show "Token saved", and syncing of
the "Liked" playlist begins. Once syncing finishes, the logs show:
"Loaded track in like playlist .."

#### Unmount

```shell
umount ~/Music/Yandex
```

#### Configuration

~/.config/yandex-fuse.json

```json
{
  "token": "",
  "quality": "hq",
  "blacklist": [],
}
```

token = Access token

quality = lossless, hq

blacklist = Genre blacklist for "My Wave" ("Моя волна")

#### Run as a systemd user service

A unit file ships inside the package:

```shell
mkdir -p ~/.config/systemd/user
cp "$(python -c 'import importlib.resources as r; print(r.files("yandex_fuse") / "contrib/yamusic-fs.service")')" \
    ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now yamusic-fs.service
```

#### Logs

```shell
journalctl --user -u yamusic-fs.service --no-pager
```

Or

```shell
cat ~/.cache/yandex_fuse.log
```

## License

LGPL-3.0-or-later, see [LICENSE](LICENSE) and [COPYING.LESSER](COPYING.LESSER).
