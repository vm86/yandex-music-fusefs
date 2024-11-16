# ruff: noqa: S101
from __future__ import annotations

import json
from io import BytesIO
from subprocess import check_output
from tempfile import NamedTemporaryFile

import pytest

from yandex_fuse.ya_player import (
    MP3_HEADER_MIN_SIZE,
    MP4_HEADER_MIN_SIZE,
    TrackTag,
)


class TestTag:
    def ffprobe_tag(self, data: bytes) -> dict[str, str]:
        with NamedTemporaryFile("wb") as tmp_fd:
            tmp_fd.write(data)
            tmp_fd.flush()
            json_output = json.loads(
                check_output(
                    [
                        "ffprobe",
                        "-loglevel",
                        "panic",
                        "-print_format",
                        "json",
                        "-show_format",
                        "-show_streams",
                        tmp_fd.name,
                    ]
                )
            )
        return {k.lower(): v for k, v in json_output["format"]["tags"].items()}

    @pytest.fixture
    def tag(self) -> TrackTag:
        return TrackTag(
            artist="Test_artist",
            title="Test_title",
            album="Test_album",
            year="2024",
            genre="Test_genre",
            duration_ms=100.0,
        )

    def tagging(self, track_tag: TrackTag, audio: BytesIO, codec: str) -> None:
        after_tag = track_tag.to_bytes(audio, codec)
        assert after_tag is not None

        audio_tags = self.ffprobe_tag(after_tag)

        tag_json = track_tag.to_dict()

        for key in ("artist", "title", "album", "genre"):
            assert audio_tags.get(key, "") == tag_json[key], audio_tags

        assert audio_tags["date"] == tag_json["year"]

    def partial_tagging(
        self,
        track_tag: TrackTag,
        audio: BytesIO,
        codec: str,
        read_size: int,
        min_tag_size: int,
    ) -> None:
        audio_parital = BytesIO()
        while data := audio.read(read_size):
            audio_parital.write(data)
            last_seek = audio_parital.tell()

            after_tag = track_tag.to_bytes(audio_parital, codec)
            if len(audio_parital.getbuffer()) < min_tag_size:
                assert after_tag is None
            else:
                assert after_tag is not None
            assert audio_parital.tell() == last_seek

        audio_parital.seek(0)

        assert audio.getvalue() == audio_parital.getvalue()

    def test_tag_mp4(self, tag: TrackTag, m4a_audio: BytesIO) -> None:
        self.tagging(tag, m4a_audio, "aac")

    def test_tag_partial_data_mp4(
        self, tag: TrackTag, m4a_audio: BytesIO
    ) -> None:
        self.partial_tagging(tag, m4a_audio, "aac", 49, MP4_HEADER_MIN_SIZE)

    def test_tag_mp3(self, tag: TrackTag, mp3_audio: BytesIO) -> None:
        self.tagging(tag, mp3_audio, "mp3")

    def test_tag_partial_data_mp3(
        self, tag: TrackTag, mp3_audio: BytesIO
    ) -> None:
        self.partial_tagging(tag, mp3_audio, "mp3", 1024, MP3_HEADER_MIN_SIZE)

    def test_tag_flac(self, tag: TrackTag, flac_audio: BytesIO) -> None:
        self.tagging(tag, flac_audio, "flac")

    def test_tag_partial_data_flac(
        self, tag: TrackTag, flac_audio: BytesIO
    ) -> None:
        self.partial_tagging(tag, flac_audio, "flac", 1024, 9098)
