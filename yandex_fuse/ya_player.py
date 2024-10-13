import base64
import json
import logging
import struct
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncGenerator

from mutagen.id3 import (
    TALB,
    TCON,
    TIT2,
    TLEN,
    TPE1,
    TYER,
)
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from yandex_music import (
    ClientAsync,
    Track,
    YandexMusicObject,
)
from yandex_music.utils import model

from yandex_fuse.request import ClientRequest

log = logging.getLogger(__name__)


def _list_to_str(attr: str, items: list[Any]) -> str:
    return " ".join([str(getattr(item, attr)) or "" for item in items or []])


@model
class TrackTag(YandexMusicObject):
    artist: str
    title: str
    album: str
    year: str
    genre: str
    duration_ms: int

    @classmethod
    def from_json(cls, data: dict) -> "TrackTag":
        return cls(
            **{
                key: data[key]
                for key in data
                if key in cls.__dataclass_fields__
            }
        )

    def _to_mp4_tag(self, stream: BytesIO) -> bytes | None:
        length = 0
        tag = {}
        while stream.tell() < len(stream.getbuffer()):
            data = stream.read(8)
            if len(data) != 8:
                return None
            atom_length, atom_name = struct.unpack(">I4s", data)
            tag[atom_name] = (atom_length, stream.tell())
            if atom_name == b"mdat":
                break
            length += atom_length
            stream.seek(length)
        if b"mdat" not in tag:
            return None

        tag_stream = BytesIO()
        stream.seek(0)
        tag_stream.write(stream.read(length))

        audiofile = MP4(fileobj=tag_stream)
        audiofile.delete(fileobj=tag_stream)
        if audiofile.tags is None:
            audiofile.add_tags()
        # https://mutagen.readthedocs.io/en/latest/api/mp4.html#mutagen.mp4.MP4Tags
        audiofile["\xa9nam"] = self.title
        audiofile["\xa9alb"] = self.album
        audiofile["\xa9ART"] = self.artist
        audiofile["\xa9day"] = self.year
        audiofile["\xa9gen"] = self.genre
        # audiofile["covr"] =
        stream.seek(0)
        audiofile.save(fileobj=tag_stream)

        stream.seek(0)
        tag_stream.seek(0)
        length = 0

        new_stream = BytesIO()
        while tag_stream.tell() < len(tag_stream.getbuffer()):
            atom_length, atom_name = struct.unpack(">I4s", tag_stream.read(8))
            new_stream.write(struct.pack(">I4s", atom_length, atom_name))
            new_stream.write(tag_stream.read(atom_length))

            length += atom_length

        atom_length, seek = tag[b"mdat"]
        stream.seek(seek - 8)
        new_stream.write(stream.read())

        return bytes(new_stream.getbuffer())

    def _to_mp3_tag(self, stream: BytesIO) -> bytes | None:
        if len(stream.getbuffer()) < 16000:
            return None

        new_stream = BytesIO()
        new_stream.write(stream.read())

        audiofile = MP3(fileobj=new_stream)
        audiofile.delete(fileobj=new_stream)
        if audiofile.tags is None:
            audiofile.add_tags()

        audiofile["TIT2"] = TIT2(encoding=3, text=self.title)
        audiofile["TPE1"] = TPE1(encoding=3, text=self.artist)
        audiofile["TALB"] = TALB(encoding=3, text=self.album)
        audiofile["TYER"] = TYER(encoding=3, text=self.year)
        audiofile["TCON"] = TCON(encoding=3, text=self.genre)
        audiofile["TLEN"] = TLEN(encoding=3, text=str(self.duration_ms))

        new_stream.seek(0)
        audiofile.save(fileobj=new_stream)

        return bytes(new_stream.getbuffer())

    def to_bytes(self, stream: BytesIO) -> bytes | None:
        if len(stream.getbuffer()) < 64:
            return None
        current_offset = stream.tell()
        stream.seek(0)

        try:
            if MP4.score(None, None, bytes(stream.getbuffer())):
                return self._to_mp4_tag(stream)
            if MP3.score("", None, bytes(stream.getbuffer())):
                return self._to_mp3_tag(stream)
        finally:
            stream.seek(current_offset)
        return bytes(stream.getbuffer())


@model
class ExtendTrack(Track):
    station_id: str = ""
    batch_id: int = 0
    size: int = 0
    codec: str = ""
    bitrate_in_kbps: int = 0
    direct_link: str = ""

    @classmethod
    def from_track(cls, track: Track) -> "ExtendTrack":
        return cls(
            **{
                key: track.__dict__[key]
                for key in track.__dict__
                if key in cls.__dataclass_fields__
            }
        )

    def to_dict(self, for_request: bool = False) -> dict:
        self.download_info = None
        return super().to_dict(for_request)

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
                size=f"{size}x{size}"
            )

            if not cover_bytes:
                continue
            log.debug("Cover download %s / %d", self.title, size)

            return base64.b64encode(cover_bytes).decode("ascii")
        return ""

    @property
    def tag(self):
        return TrackTag(
            artist=", ".join(self.artists_name()),
            title=self.title or "-",
            album=_list_to_str("title", self.albums),
            year=_list_to_str("year", self.albums),
            genre=_list_to_str("genre", self.albums),
            duration_ms=self.duration_ms,
        )

    def _choose_best_dowanload_info(self):
        best_bitrate_in_kbps = {"aac": 0, "mp3": 0}
        track_info = {"aac": None, "mp3": None}

        best_codec = self.client.settings["best_codec"].lower()

        if self.download_info is None:
            raise ValueError(
                "Download info for track %s empty!"
                "Call get_download_info(async) before get info." % self.title
            )
        for info in self.download_info:
            log.debug(
                "Track %s %s/%d",
                self.title,
                info.codec,
                info.bitrate_in_kbps,
            )

            best_bitrate_in_kbps[info.codec] = max(
                info.bitrate_in_kbps, best_bitrate_in_kbps[info.codec]
            )

            if info.bitrate_in_kbps >= best_bitrate_in_kbps[info.codec]:
                track_info[info.codec] = info

        track_codec_info = track_info.pop(best_codec)
        if track_codec_info is None:
            _, track_codec_info = track_info.popitem()
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


class YandexMusicPlayer(ClientAsync):
    _default_settings = {
        "token": "",
        "last_track": None,
        "from_id": f"music-{uuid.uuid4()}",
        "station_id": "user:onyourwave",
        "best_codec": "aac",
        "blacklist": [],
    }

    def __init__(self, settings_path: Path, client_session):
        self.__last_track = None
        self.__last_state = None
        self.__settings = None
        self.__current_station = None
        self.__last_station_id = None

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

        if "token" in self.__settings["token"]:
            raise RuntimeError("Token not found in {self.__settings_path}")

        self.__client_session = client_session

        super().__init__(
            self.__settings["token"], request=ClientRequest(client_session)
        )

    def save_settings(self):
        self.__settings_path.write_text(json.dumps(self.__settings))

    @property
    def settings(self):
        return self.__settings

    @property
    def is_init(self) -> bool:
        return self.__settings["token"] is not None

    @property
    def _from_id(self):
        return self.__settings["from_id"]

    async def load_tracks(
        self, tracks: list[Track], exclude_track_ids: list[str]
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

        tracks = set()

        while len(tracks) < count:
            station_tracks = await self.rotor_station_tracks(
                station=station_id, queue=self.__last_track
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

    def get_last_station_info(self):
        return self.__last_station_id

    async def get_download_link(
        self, track_id: str, codec: str, bitrate_in_kbps: int
    ) -> str | None:
        download_info = await self.tracks_download_info(track_id)
        for info in download_info:
            if info.codec == codec and info.bitrate_in_kbps == bitrate_in_kbps:
                return await info.get_direct_link_async()

        return None

    async def feedback_track(
        self,
        track_id: str,
        feedback: str,
        batch_id: str,
        total_played_seconds: int,
    ):
        # trackStarted, trackFinished, skip.
        if self.__last_station_id is None:
            return
        station_id, station_batch_id = self.__last_station_id

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
