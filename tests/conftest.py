from io import BytesIO
from pathlib import Path

import pytest


def _load_audio(codec: str) -> BytesIO:
    return BytesIO(
        initial_bytes=Path(f"tests/contrib/test.{codec}").read_bytes()
    )


@pytest.fixture
def m4a_audio() -> BytesIO:
    return _load_audio("m4a")


@pytest.fixture
def mp3_audio() -> BytesIO:
    return _load_audio("mp3")


@pytest.fixture
def flac_audio() -> BytesIO:
    return _load_audio("flac")
