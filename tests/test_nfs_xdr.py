# ruff: noqa: S101
# mypy: ignore-errors

from yandex_fuse.nfs.xdr import XdrDecoder, XdrEncoder

UINT32_SAMPLE = 0xDEADBEEF
UINT64_SAMPLE = 0x1122334455667788
TRAILING_MARKER = 42


def test_uint32_roundtrip() -> None:
    data = XdrEncoder().uint32(UINT32_SAMPLE).to_bytes()
    assert XdrDecoder(data).uint32() == UINT32_SAMPLE


def test_uint64_roundtrip() -> None:
    data = XdrEncoder().uint64(UINT64_SAMPLE).to_bytes()
    assert XdrDecoder(data).uint64() == UINT64_SAMPLE


def test_opaque_roundtrip_and_padding() -> None:
    encoder = XdrEncoder().opaque(b"abc").uint32(TRAILING_MARKER)
    data = encoder.to_bytes()

    # "abc" (3 bytes) is padded to a 4-byte boundary.
    assert len(data) == 4 + 4 + 4

    decoder = XdrDecoder(data)
    assert decoder.opaque() == b"abc"
    assert decoder.uint32() == TRAILING_MARKER


def test_string_roundtrip() -> None:
    data = XdrEncoder().string("привет").to_bytes()
    assert XdrDecoder(data).string() == "привет"


def test_raw_has_no_length_prefix() -> None:
    data = XdrEncoder().raw(b"\x00" * 8).uint32(7).to_bytes()
    assert len(data) == 8 + 4
    assert XdrDecoder(data).uint64() == 0
