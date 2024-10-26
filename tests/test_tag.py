# ruff: noqa: S101
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

    def test_tag_mp4(self, tag: TrackTag, m4a_wtihout_tag: BytesIO) -> None:
        after_tag = tag.to_bytes(m4a_wtihout_tag)
        assert after_tag is not None

        with NamedTemporaryFile("wb") as tmp_fd:
            tmp_fd.write(after_tag)
            tmp_fd.flush()
            ffprobe_output = check_output(
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
        json_output = json.loads(ffprobe_output)

        tag_json = tag.to_dict()

        for key in ("artist", "title", "album", "genre"):
            assert json_output["format"]["tags"][key] == tag_json[key]

        assert json_output["format"]["tags"]["date"] == tag_json["year"]

    def test_tag_partial_data_mp4(
        self, tag: TrackTag, m4a_wtihout_tag: BytesIO
    ) -> None:
        m4a_parital = BytesIO()
        while data := m4a_wtihout_tag.read(49):
            m4a_parital.write(data)
            last_seek = m4a_parital.tell()

            after_tag = tag.to_bytes(m4a_parital)
            if len(m4a_parital.getbuffer()) < MP4_HEADER_MIN_SIZE:
                assert after_tag is None
            else:
                assert after_tag is not None
            assert m4a_parital.tell() == last_seek

        m4a_parital.seek(0)
        assert m4a_wtihout_tag.getvalue() == m4a_parital.getvalue()

    def test_tag_mp3(self, tag: TrackTag, mp3_wtihout_tag: BytesIO) -> None:
        after_tag = tag.to_bytes(mp3_wtihout_tag)
        assert after_tag is not None

        with NamedTemporaryFile("wb") as tmp_fd:
            tmp_fd.write(after_tag)
            tmp_fd.flush()
            ffprobe_output = check_output(
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
        json_output = json.loads(ffprobe_output)

        tag_json = tag.to_dict()

        for key in ("artist", "title", "album", "genre"):
            assert json_output["format"]["tags"][key] == tag_json[key]

        assert json_output["format"]["tags"]["date"] == tag_json["year"]

    def test_tag_partial_data_mp3(
        self, tag: TrackTag, mp3_wtihout_tag: BytesIO
    ) -> None:
        mp3_parital = BytesIO()
        while data := mp3_wtihout_tag.read(1024):
            mp3_parital.write(data)
            last_seek = mp3_parital.tell()

            after_tag = tag.to_bytes(mp3_parital)
            if len(mp3_parital.getbuffer()) < MP3_HEADER_MIN_SIZE:
                assert after_tag is None
            else:
                assert after_tag is not None
            assert mp3_parital.tell() == last_seek

        mp3_parital.seek(0)
        assert mp3_wtihout_tag.getvalue() == mp3_parital.getvalue()
