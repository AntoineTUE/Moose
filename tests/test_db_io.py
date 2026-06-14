import hashlib
from pathlib import Path
import pytest
from io import BytesIO

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
