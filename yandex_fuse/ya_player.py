from __future__ import annotations

import base64
import json
import logging
import struct
import time
import uuid
import webbrowser
from asyncio import sleep
from asyncio.tasks import create_task
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar

from mutagen.id3 import (  # type: ignore[attr-defined]
    TALB,
    TCON,
    TIT2,
    TLEN,
    TPE1,
    TYER,
)
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from yandex_music import (  # type: ignore[import-untyped]
    ClientAsync,
    DownloadInfo,
    Track,
    YandexMusicObject,
)
from yandex_music.utils import model  # type: ignore[import-untyped]

from yandex_fuse.request import YandexClientRequest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from aiohttp import ClientSession

log = logging.getLogger(__name__)


def _list_to_str(attr: str, items: list[Any]) -> str:
    return " ".join([str(getattr(item, attr)) or "" for item in items or []])


MP4_HEADER_CHUNK_SIZE = 8
MP3_HEADER_MIN_SIZE = 16000
MP4_HEADER_MIN_SIZE = 62835


@model
class TrackTag(YandexMusicObject):  # type: ignore[misc]
    artist: str
    title: str
    album: str
    year: str
    genre: str
    duration_ms: int

    @classmethod
    def from_json(cls, data: dict[str, int | str]) -> TrackTag:
        return cls(
            **{
                key: data[key]
                for key in data
                if key in cls.__dataclass_fields__
            },
        )

    def _to_mp4_tag(self, stream: BytesIO) -> bytes | None:
        tag = {}
        tag_stream = BytesIO()

        while data := stream.read(MP4_HEADER_CHUNK_SIZE):
            if len(data) != MP4_HEADER_CHUNK_SIZE:
                break
            atom_length, atom_name = struct.unpack(">I4s", data)
            tag[atom_name] = (atom_length, stream.tell())
            if atom_name == b"mdat":
                break
            tag_stream.write(data)
            tag_stream.write(stream.read(atom_length - MP4_HEADER_CHUNK_SIZE))
        if {b"mdat", b"moov"} & tag.keys() != {b"mdat", b"moov"}:
            return None

        audiofile = MP4(fileobj=tag_stream)  # type: ignore[no-untyped-call]
        audiofile.delete(fileobj=tag_stream)
        if audiofile.tags is None:
            audiofile.add_tags()  # type: ignore[no-untyped-call]

        # https://mutagen.readthedocs.io/en/latest/api/mp4.html#mutagen.mp4.MP4Tags
        audiofile["\xa9nam"] = self.title
        audiofile["\xa9alb"] = self.album
        audiofile["\xa9ART"] = self.artist
        audiofile["\xa9day"] = self.year
        audiofile["\xa9gen"] = self.genre
        # audiofile["covr"] =
        stream.seek(0)
        audiofile.save(fileobj=tag_stream)  # type: ignore[no-untyped-call]

        stream.seek(0)
        tag_stream.seek(0)

        new_stream = BytesIO()
        while data := tag_stream.read(MP4_HEADER_CHUNK_SIZE):
            atom_length, atom_name = struct.unpack(">I4s", data)
            new_stream.write(data)
            new_stream.write(
                tag_stream.read(atom_length - MP4_HEADER_CHUNK_SIZE)
            )

        atom_length, seek = tag[b"mdat"]
        stream.seek(seek - MP4_HEADER_CHUNK_SIZE)
        new_stream.write(stream.read())

        return bytes(new_stream.getbuffer())

    def _to_mp3_tag(self, stream: BytesIO) -> bytes | None:
        new_stream = BytesIO()
        new_stream.write(stream.read())

        audiofile = MP3(fileobj=new_stream)
        audiofile.delete(fileobj=new_stream)
        if audiofile.tags is None:
            audiofile.add_tags()  # type: ignore[no-untyped-call]

        audiofile["TIT2"] = TIT2(encoding=3, text=self.title)  # type: ignore[no-untyped-call]
        audiofile["TPE1"] = TPE1(encoding=3, text=self.artist)  # type: ignore[no-untyped-call]
        audiofile["TALB"] = TALB(encoding=3, text=self.album)  # type: ignore[no-untyped-call]
        audiofile["TYE"] = TYER(encoding=3, text=self.year)  # type: ignore[no-untyped-call]
        audiofile["TCON"] = TCON(encoding=3, text=self.genre)  # type: ignore[no-untyped-call]
        audiofile["TLEN"] = TLEN(encoding=3, text=str(self.duration_ms))  # type: ignore[no-untyped-call]

        new_stream.seek(0)
        audiofile.save(fileobj=new_stream)

        return bytes(new_stream.getbuffer())

    def to_bytes(self, stream: BytesIO) -> bytes | None:
        buffer = bytearray(stream.getbuffer())

        current_offset = stream.tell()
        stream.seek(0)

        try:
            if len(buffer) <= MP3_HEADER_MIN_SIZE:
                return None
            if MP3.score("", None, buffer):  # type: ignore[no-untyped-call]
                return self._to_mp3_tag(stream)

            if len(buffer) <= MP4_HEADER_MIN_SIZE:
                return None
            if MP4.score(None, None, buffer):  # type: ignore[no-untyped-call]
                return self._to_mp4_tag(stream)

        finally:
            stream.seek(current_offset)
        return bytes(stream.getbuffer())


@model
class ExtendTrack(Track):  # type: ignore[misc]
    station_id: str = ""
    batch_id: int = 0
    size: int = 0
    codec: str = ""
    bitrate_in_kbps: int = 0
    direct_link: str = ""

    @classmethod
    def from_track(cls, track: Track) -> ExtendTrack:
        return cls(
            **{
                key: track.__dict__[key]
                for key in track.__dict__
                if key in cls.__dataclass_fields__
            },
        )

    @property
    def save_name(self) -> str:
        artists = ", ".join(self.artists_name())
        name = f"{artists} - {self.title}"
        if self.version:
            name += f" ({self.version})"

        if self.codec == "aac":
            ext_name = "m4a"
        elif self.codec == "mp3":
            ext_name = "mp3"
        else:
            ext_name = "unknown"
        return f"{name}.{ext_name}".replace("/", "-")

    async def _download_image(self) -> str:
        for size in (600, 400, 300, 200):
            cover_bytes = await self.download_cover_bytes_async(
                size=f"{size}x{size}",
            )

            if not cover_bytes:
                continue
            log.debug("Cover download %s / %d", self.title, size)

            return base64.b64encode(cover_bytes).decode("ascii")
        return ""

    @property
    def tag(self) -> TrackTag:
        return TrackTag(
            artist=", ".join(self.artists_name()),
            title=self.title or "-",
            album=_list_to_str("title", self.albums),
            year=_list_to_str("year", self.albums),
            genre=_list_to_str("genre", self.albums),
            duration_ms=self.duration_ms,
        )

    def _choose_best_dowanload_info(self) -> DownloadInfo:
        best_bitrate_in_kbps = {"aac": 0, "mp3": 0}
        track_info: dict[str, DownloadInfo | None] = {"aac": None, "mp3": None}

        best_codec = self.client.settings["best_codec"].lower()
        self.download_info: list[DownloadInfo] | None
        if self.download_info is None:
            msg = (
                f"Download info for track {self.title} empty!"
                "Call get_download_info(async) before get info."
            )
            raise ValueError(
                msg,
            )

        for info in self.download_info:
            log.debug(
                "Track %s %s/%d",
                self.title,
                info.codec,
                info.bitrate_in_kbps,
            )

            best_bitrate_in_kbps[info.codec] = max(
                info.bitrate_in_kbps,
                best_bitrate_in_kbps[info.codec],
            )

            if info.bitrate_in_kbps >= best_bitrate_in_kbps[info.codec]:
                track_info[info.codec] = info

        track_codec_info = track_info.pop(best_codec)
        if track_codec_info is None:
            _, track_codec_info = track_info.popitem()
            if track_codec_info is None:
                raise RuntimeError(f"Track info {self.title} is empty.")

            log.warning(
                "Best codec %s from track %s not available. Fallback. %s",
                best_codec,
                self.title,
                track_codec_info.codec,
            )

        log.debug(
            "Track %s choose codec %s/%d",
            self.title,
            track_codec_info.codec,
            track_codec_info.bitrate_in_kbps,
        )

        return track_codec_info

    async def update_download_info(self) -> None:
        if self.download_info is None:
            log.debug('Track "%s" update download info.', self.title)
            await self.get_download_info_async()

        download_info = self._choose_best_dowanload_info()
        self.codec = download_info.codec
        self.bitrate_in_kbps = download_info.bitrate_in_kbps
        self.download_info = [download_info]


class YandexMusicPlayer(ClientAsync):  # type: ignore[misc]
    _default_settings: ClassVar = {
        "token": None,
        "last_track": None,
        "from_id": f"music-{uuid.uuid4()}",
        "station_id": "user:onyourwave",
        "best_codec": "aac",
        "blacklist": [],
    }

    def __init__(
        self,
        settings_path: Path,
        client_session: ClientSession,
    ) -> None:
        self.__last_track: str = ""
        self.__last_station_id: tuple[str, str] = ("", "")
        self.__settings: dict[str, Any] = {}

        self.__settings_path = settings_path
        try:
            self.__settings = {
                **self._default_settings,
                **json.loads(self.__settings_path.read_text()),
            }
            self.__last_track = self.__settings["last_track"]
        except FileNotFoundError:
            self.__settings = self._default_settings
            self.save_settings()

        self.__client_session = client_session
        super().__init__(
            self.__settings["token"],
            request=YandexClientRequest(client_session),
        )

        self.__init_task = None
        if not self.__settings["token"]:
            self.__init_task = create_task(
                self._init_token(),
                name="init-token",
            )

    async def _init_token(self) -> None:
        qr_link = await self._request.get_qr()
        webbrowser.open_new_tab(qr_link)
        response = None
        while response is None:
            response = await self._request.login_qr()
            if response:
                break
            await sleep(5)
        token_info = await self._request.get_music_token(response)
        self.__settings["token"] = token_info["access_token"]
        self._request.set_authorization(self.__settings["token"])
        self.save_settings()
        log.info("Token saved.")
        if self.__init_task is not None:
            init_task = self.__init_task
            self.__init_task = None
            init_task.cancel()

    def save_settings(self) -> None:
        self.__settings_path.write_text(json.dumps(self.__settings))

    @property
    def settings(self) -> dict[str, Any]:
        return self.__settings

    @property
    def is_init(self) -> bool:
        return self.__settings["token"] is not None

    @property
    def _from_id(self) -> str:
        result: str = self.__settings["from_id"]
        return result

    async def load_tracks(
        self,
        tracks: list[Track],
        exclude_track_ids: set[str],
    ) -> AsyncGenerator[ExtendTrack, None]:
        for track in tracks:
            if str(track.id) in exclude_track_ids:
                continue
            extend_track = ExtendTrack.from_track(track)
            await extend_track.update_download_info()
            yield extend_track

    async def next_tracks(
        self,
        station_id: str,
        count: int = 50,
    ) -> AsyncGenerator[ExtendTrack, None]:
        if station_id is None:
            raise ValueError("Station is not select!")

        tracks: set[str] = set()

        while len(tracks) < count:
            station_tracks = await self.rotor_station_tracks(
                station=station_id,
                queue=self.__last_track,
            )

            if self.__last_station_id != (station_id, station_tracks.batch_id):
                self.__last_station_id = (station_id, station_tracks.batch_id)
                await self.rotor_station_feedback_radio_started(
                    station_id,
                    self._from_id,
                    station_tracks.batch_id,
                    time.time(),
                )

            for sequence_track in station_tracks.sequence:
                if sequence_track.track is None:
                    continue
                extend_track = ExtendTrack.from_track(sequence_track.track)
                extend_track.station_id = station_id
                extend_track.batch_id = station_tracks.batch_id

                if extend_track.save_name in tracks:
                    continue

                if extend_track.tag.genre in self.settings["blacklist"]:
                    log.warning(
                        "Track %s/%s in BLACKLIST",
                        extend_track.save_name,
                        extend_track.tag.genre,
                    )
                    await self.feedback_track(
                        extend_track.track_id,
                        "skip",
                        station_tracks.batch_id,
                        0,
                    )

                    continue

            first_track = station_tracks.sequence[0].track
            await self.feedback_track(
                first_track.track_id,
                "trackStarted",
                station_tracks.batch_id,
                0,
            )

            last_track = station_tracks.sequence[-1].track
            await self.feedback_track(
                last_track.track_id,
                "trackFinished",
                station_tracks.batch_id,
                last_track.duration_ms / 1000,
            )
            self.__last_track = last_track.track_id

            await extend_track.update_download_info()
            tracks.add(extend_track.save_name)
            yield extend_track

        if self.__settings["last_track"] != self.__last_track:
            self.__settings["last_track"] = self.__last_track
            self.save_settings()

    def get_last_station_info(self) -> tuple[str, str]:
        return self.__last_station_id

    async def get_download_link(
        self,
        track_id: str,
        codec: str,
        bitrate_in_kbps: int,
    ) -> str | None:
        download_info = await self.tracks_download_info(track_id)
        for info in download_info:
            if info.codec == codec and info.bitrate_in_kbps == bitrate_in_kbps:
                direct_link: str = await info.get_direct_link_async()
                return direct_link
        return None

    async def feedback_track(
        self,
        track_id: str,
        feedback: str,
        batch_id: str,
        total_played_seconds: int,
    ) -> None:
        # trackStarted, trackFinished, skip.
        station_id, station_batch_id = self.__last_station_id
        if not station_id or not station_batch_id:
            return
        try:
            await self.rotor_station_feedback(
                station=station_id,
                type_=feedback,
                timestamp=time.time(),
                from_=self._from_id,
                batch_id=batch_id,
                total_played_seconds=total_played_seconds,
                track_id=track_id,
            )
        except Exception:
            log.exception("Error send feedback:")
