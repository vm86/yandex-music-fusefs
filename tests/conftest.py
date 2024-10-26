from io import BytesIO
from pathlib import Path

import pytest


@pytest.fixture
def m4a_wtihout_tag() -> BytesIO:
    return BytesIO(initial_bytes=Path("tests/contrib/test.m4a").read_bytes())


@pytest.fixture
def mp3_wtihout_tag() -> BytesIO:
    return BytesIO(initial_bytes=Path("tests/contrib/test.mp3").read_bytes())
