import hashlib
from io import BytesIO
from pathlib import Path

import pytest

import Moose.utils.db_io as db_io


def git_blob_hash(data: bytes) -> str:
    h = hashlib.sha1(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()


def test_generate_chunks_and_hash():
    data = b"some content\n" * 50
    buff = BytesIO(data)
    chunks, sha1 = db_io.generate_chunks_and_hash(buff, len(data), chunk_size=13)
    content = b"".join(chunks)
    assert content == data
    assert sha1.hexdigest() == git_blob_hash(content)


class MockResponse:
    def __init__(self, data: bytes):
        self._data = data

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i : i + chunk_size]

    def raise_for_status(self):
        return None


class MockSession:
    def __init__(self, response: MockResponse | None = None):
        self._response = response
        self.calls = []

    def get(self, url, *args, **kwargs):
        self.calls.append(url)
        if self._response is None:
            raise AssertionError("Network should not be called")
        return self._response


def test_generate_chunks_and_hash_Response():
    """Test with a mocked requests Response"""
    data = b"streamed content\n" * 100

    resp = MockResponse(data)

    chunks, sha1 = db_io.generate_chunks_and_hash(resp, len(data), chunk_size=23)

    received = b"".join(chunks)

    assert received == data
    assert sha1.hexdigest() == git_blob_hash(data)


def test_generate_chunks_and_hash_size_mismatch():
    data = b"abcdef"

    buff = BytesIO(data)

    chunks, _ = db_io.generate_chunks_and_hash(buff, size=len(data) + 1)

    with pytest.raises(ValueError, match=r"Size mismatch*"):
        list(chunks)


def test_download_file_skips_and_validates(tmp_path):
    """Check the SKIP and VALIDATE behaviors"""
    data = b"content\n" * 5
    name = "test.dat"
    info = {"name": name, "size": len(data), "sha": git_blob_hash(data), "download_url": "http://x"}

    target_dir = tmp_path
    fpath = target_dir.joinpath(name)
    fpath.write_bytes(data)

    # SKIP: file already exists, so no download
    session = MockSession(response=None)
    db_io.download_file(session, info, target_dir, mode=db_io.FileExistsBehavior.SKIP)
    assert session.calls == []

    # VALIDATE: existing file matches sha, so no download
    session = MockSession(response=None)
    db_io.download_file(session, info, target_dir, mode=db_io.FileExistsBehavior.VALIDATE)
    assert session.calls == []


def test_download_file_raise_on_exists(tmp_path):
    """Check RAISE behavior for existing file."""
    data = b"old\n"
    name = "already.dat"
    info = {"name": name, "size": len(data), "sha": git_blob_hash(data), "download_url": "http://x"}

    target_dir = tmp_path
    fpath = target_dir.joinpath(name)
    fpath.write_bytes(data)

    with pytest.raises(OSError):
        db_io.download_file(MockSession(response=None), info, target_dir, mode=db_io.FileExistsBehavior.RAISE)


def test_download_file_performs_download(tmp_path):
    data = b"streamed bytes\n" * 20
    name = "remote.dat"
    info = {"name": name, "size": len(data), "sha": git_blob_hash(data), "download_url": "http://remote/file"}

    target_dir = tmp_path
    session = MockSession(response=MockResponse(data))
    db_io.download_file(session, info, target_dir, mode=db_io.FileExistsBehavior.OVERWRITE)

    written = target_dir.joinpath(name).read_bytes()
    assert written == data
    assert db_io.hash_file(target_dir.joinpath(name)) == info["sha"]


def test_migrate_file_skip_and_overwrite(tmp_path):
    """Check migration behavior for SKIP and OVERWRITE when file already exists."""
    src = tmp_path.joinpath("src.txt")
    src.write_text("new")
    dest_dir = tmp_path.joinpath("dest")
    dest_dir.mkdir()
    dest_file = dest_dir.joinpath("src.txt")
    dest_file.write_text("old")

    # SKIP: should not overwrite
    db_io.migrate_file(src, dest_dir, mode=db_io.FileExistsBehavior.SKIP)
    assert dest_file.read_text() == "old"

    # OVERWRITE: should replace content
    db_io.migrate_file(src, dest_dir, mode=db_io.FileExistsBehavior.OVERWRITE)
    assert dest_file.read_text() == "new"


def test_set_and_get_database_path_and_fallback(monkeypatch, tmp_path):
    """IMPORTANT: after this test runs, the original path must be restored.
    Else, if `test_Simulation.py` follows it, some tests will fail since they look in this temp path.
    This issue will only be caught when using pytest --randomize.

    monkeypatch.setattr will reverse this.
    """
    # set and get
    p = tmp_path.joinpath("dbdir")
    p.mkdir()
    monkeypatch.setattr(db_io, "USER_DIR", None)
    db_io.set_database_path(p)
    assert db_io.get_database_path() == p


def test_get_database_path_with_fallback(monkeypatch, tmp_path):
    """IMPORTANT: after this test runs, the original path must be restored.
    Else, if `test_Simulation.py` follows it, some tests will fail since they look in this temp path.
    This issue will only be caught when using pytest --randomize.

    monkeypatch.setattr will reverse this.
    """
    # fallback to packaged data when USER_DIR does not exist
    fake_pkg = tmp_path.joinpath("pkgdata")
    fake_pkg.mkdir()
    monkeypatch.setattr(db_io, "USER_DIR", None)
    monkeypatch.setattr(db_io, "USER_DIR", Path(tmp_path / "nonexistent_dir"))
    monkeypatch.setattr(db_io, "_find_packaged_data", lambda: (fake_pkg, []))
    # avoid repeated notification toggling affecting other tests
    monkeypatch.setattr(db_io, "HAS_BEEN_NOTIFIED", False)
    assert db_io.get_database_path() == fake_pkg


def test_database_files_returns_db_stems(monkeypatch, tmp_path):
    p = tmp_path.joinpath("datafolder")
    p.mkdir()
    (p / "a.db").write_text("x")
    (p / "b.db").write_text("y")
    (p / "ignored.txt").write_text("no")

    monkeypatch.setattr(db_io, "get_database_path", lambda: p)
    files = db_io.database_files()
    assert set(files) == {"a", "b"}
