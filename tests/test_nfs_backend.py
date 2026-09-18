# ruff: noqa: S101
# mypy: ignore-errors

from unittest import mock

import pytest

from yandex_fuse.nfs.backend import (
    AUTH_PAGE_INODE,
    AUTH_PAGE_NAME,
    YaMusicNfsBackend,
)
from yandex_fuse.virt_fs import ROOT_INODE

AUTH_HTML = b"<html>auth</html>"


def _music_fs(*, needs_auth: bool) -> mock.MagicMock:
    music_fs = mock.MagicMock()
    music_fs.needs_auth = needs_auth
    music_fs.auth_page.return_value = AUTH_HTML
    music_fs.dir_entries.return_value = []
    return music_fs


def test_getattr_auth_page_when_needs_auth() -> None:
    backend = YaMusicNfsBackend(_music_fs(needs_auth=True))

    attr = backend.getattr(AUTH_PAGE_INODE)

    assert attr is not None
    assert attr.size == len(AUTH_HTML)


def test_getattr_auth_page_hidden_when_authorized() -> None:
    backend = YaMusicNfsBackend(_music_fs(needs_auth=False))

    assert backend.getattr(AUTH_PAGE_INODE) is None


def test_lookup_finds_index_html_when_needs_auth() -> None:
    backend = YaMusicNfsBackend(_music_fs(needs_auth=True))

    assert backend.lookup(ROOT_INODE, AUTH_PAGE_NAME) == AUTH_PAGE_INODE


def test_lookup_index_html_missing_when_authorized() -> None:
    music_fs = _music_fs(needs_auth=False)
    music_fs.lookup_by_name.return_value = None
    backend = YaMusicNfsBackend(music_fs)

    assert backend.lookup(ROOT_INODE, AUTH_PAGE_NAME) is None


def test_readdir_root_includes_auth_page_when_needs_auth() -> None:
    backend = YaMusicNfsBackend(_music_fs(needs_auth=True))

    entries = backend.readdir(ROOT_INODE)

    assert (AUTH_PAGE_NAME, AUTH_PAGE_INODE) in entries


def test_readdir_root_excludes_auth_page_when_authorized() -> None:
    backend = YaMusicNfsBackend(_music_fs(needs_auth=False))

    entries = backend.readdir(ROOT_INODE)

    assert (AUTH_PAGE_NAME, AUTH_PAGE_INODE) not in entries


@pytest.mark.asyncio
async def test_read_auth_page_returns_content_slice() -> None:
    backend = YaMusicNfsBackend(_music_fs(needs_auth=True))

    chunk = await backend.read(AUTH_PAGE_INODE, 0, 5)

    assert chunk == AUTH_HTML[:5]


@pytest.mark.asyncio
async def test_read_auth_page_returns_none_when_authorized() -> None:
    backend = YaMusicNfsBackend(_music_fs(needs_auth=False))

    assert await backend.read(AUTH_PAGE_INODE, 0, 5) is None
