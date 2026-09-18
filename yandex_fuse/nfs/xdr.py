from __future__ import annotations

import struct


class XdrDecoder:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def uint32(self) -> int:
        value = struct.unpack_from(">I", self._data, self._pos)[0]
        self._pos += 4
        return int(value)

    def uint64(self) -> int:
        value = struct.unpack_from(">Q", self._data, self._pos)[0]
        self._pos += 8
        return int(value)

    def opaque(self) -> bytes:
        length = self.uint32()
        padded = (length + 3) & ~3
        value = self._data[self._pos : self._pos + length]
        self._pos += padded
        return value

    def string(self) -> str:
        return self.opaque().decode("utf-8", errors="replace")


class XdrEncoder:
    def __init__(self) -> None:
        self._chunks: list[bytes] = []

    def uint32(self, value: int) -> XdrEncoder:
        self._chunks.append(struct.pack(">I", value & 0xFFFFFFFF))
        return self

    def uint64(self, value: int) -> XdrEncoder:
        self._chunks.append(struct.pack(">Q", value & 0xFFFFFFFFFFFFFFFF))
        return self

    def opaque(self, data: bytes) -> XdrEncoder:
        self.uint32(len(data))
        self._chunks.append(data)
        self._pad(len(data))
        return self

    def raw(self, data: bytes) -> XdrEncoder:
        """Fixed-length opaque with no length prefix (cookieverf3 etc.)."""
        self._chunks.append(data)
        self._pad(len(data))
        return self

    def string(self, value: str) -> XdrEncoder:
        return self.opaque(value.encode("utf-8"))

    def _pad(self, length: int) -> None:
        pad = (-length) % 4
        if pad:
            self._chunks.append(b"\x00" * pad)

    def to_bytes(self) -> bytes:
        return b"".join(self._chunks)
