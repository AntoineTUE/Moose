"""Module for downloading line-by-line databases for use in Moose.

Currently Moose ships with some database files bundled.

This has some issues:
    * The size of the package is large (80 MB).
    * Each install or update of Moose downloads or copies the data
    * The default search path for the databases is the install location of Moose, which would be opaque to the user.
    * Searching for the install location in the current environment can be very slow, increasing import times.

To address these issues, this module handles downloading of these database files, and will store them in a more standardized location.

The location is a subfolder in the user directory, depending on the OS:

=== "Windows"
    `%USERPROFILE%/moose-spectra`
=== "Linux"
    `~/moose-spectra`
=== "MacOS"
    `~/moose-spectra`

To get started with Moose in the future, a user will first need to call the [download_databases][..] function, or put their own files in the database folder.

This will become the norm in a future release, after a grace period (to be decided).

In the meantime, it will be possible to migrate the bundled databases to the new location, using either [migrate][..] or [download_databases][..]
"""

from _hashlib import HASH
from collections.abc import Callable, Iterable

import hashlib
from pathlib import Path
from importlib import resources
import shutil

from requests import Session, Response
from tqdm import tqdm

GITHUB_API_URL = "https://api.github.com"
REPO = "AntoineTUE/Moose"
SRC_DATA_PATH = "src/Moose/data"  # case-sensitive
_PKGD_PATH: Path = resources.files("Moose").joinpath("data")  # This can be very slow the first time
_PKGD_FILES = list(_PKGD_PATH.glob("*"))
USER_DIR = Path.home().joinpath("moose-spectra")
_CHUNK_SIZE = 2097152


def generate_chunks_and_hash(buff, size: int, chunk_size=_CHUNK_SIZE) -> tuple[Iterable[bytes], HASH]:
    """Stream the contents of a file-like object or Response, while hashing the contents.

    Uses the same hashing as git, so it can be compared to files in the repo.

    See also: https://git-scm.com/book/ms/v2/Git-Internals-Git-Objects

    Note:
        The hash digest should only be called after the iterator has been exhausted and the full content has been processed.
        If the iterator is exhausted but the size is different from `size`, a ValueError is raised.

    Example:
    ```python
    file = Path("test.tmp")
    with file.open('rb') as fo:
        chunks, sha1 = stream_with_hash(fo,file.stat().st_size)
        for chunk in chunks:
            pass

        digest = sha1.hexdigest()
    ```
    """
    sha1: HASH = hashlib.sha1()
    sha1.update(f"blob {size}\0".encode())

    def iterator():
        bytes_read = 0
        if hasattr(buff, "read"):  # file-like object
            while chunk := buff.read(chunk_size):
                sha1.update(chunk)
                bytes_read += len(chunk)
                yield chunk
        else:  # requests.Response
            for chunk in buff.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                sha1.update(chunk)
                bytes_read += len(chunk)
                yield chunk
        if bytes_read != size:
            raise ValueError(f"Size mismatch when reading stream. Expected {size=}, {bytes_read=}.")

    return iterator(), sha1


def hash_file(file: Path):
    """Compute a file hash, for comparing files based on contents."""
    with file.open("rb") as fo:
        chunks, hashed = generate_chunks_and_hash(fo, file.stat().st_size)
        # iterate over chucks to compute the hash
        for _ in chunks:
            pass
    return hashed.hexdigest()


def list_data_files_in_repo(ses: Session, branch="main") -> list[dict]:
    """Retrieve the list of files in the data directory in the Moose repository.

    Uses the public GitHub API, without need for auth.

    For each file, returns a dictionary with useful keys such as:
        * name
        * sha
        * size
        * download_url
        * type

    Args:
        ses (Session):  A persistent requests Session
        branch (str):   The name of the git branch used for the look up, default: `'main'`.

    Returns:
        info (list[dict]):  A list of dictionaries with info per file that existst under the path.
    """
    response = ses.get(f"{GITHUB_API_URL}/repos/{REPO}/contents/{SRC_DATA_PATH}", params={"ref": branch})
    return response.json()


def download_file(ses: Session, info: dict[str, str | int], target_dir: Path, overwrite=False):
    """Download a file from the Moose repo based on information returned from the GitHub API.

    If `overwrite=False` and the file already exists, the download will be skipped.

    Similarly, if `overwrite=True`, but the file hash on disk is the same as upstream, the files are already the same and nothing will be downloaded.

    Finally, if data is downloaded, it checks if the file SHA hash matches the expected hash to verify the download.

    Args:
        ses (Session):  A persistent requests Session
        info (dict):    A dict with file path information as retrieved from the GitHub API.
        target_dir (Path):  The target directory to store the file in, if download succeeds
        overwrite (bool): Flag to overwrite files that already exist or not (default: False)

    Raises:
        OSError:    raised when the target file already exists and `overwrite=False`
        ValueError: raised when there is a mismatch in the SHA hash reported by GitHub and the actual download.
    """
    name: str = info["name"]  # ty:ignore[invalid-assignment]
    response: Response = ses.get(info["download_url"], stream=True, timeout=30)
    response.raise_for_status()
    file_path = target_dir.joinpath(name)
    if file_path.exists() and not overwrite:
        raise OSError(f"File {name} already exists.")
    if file_path.exists():
        existing_hash = hash_file(file_path)
        if existing_hash == info["sha"]:
            print(f"{file_path} is up-to-date, skipping download.")
            return  # No need to download
    with file_path.open("wb") as f, tqdm(total=info["size"], unit="B", unit_scale=True, desc=name) as pbar:
        chunks, sha1 = generate_chunks_and_hash(response, info["size"])  # ty:ignore[invalid-argument-type]
        for chunk in chunks:
            f.write(chunk)
            pbar.update(len(chunk))
    if sha1.hexdigest() != info["sha"]:
        raise ValueError(
            f"Hash of downloaded file {file_path} does not match {info['sha']}, this indicates a download error."
        )


def download_databases_from_repo(target_dir: Path | None = None, overwrite=False):
    """Download all databases and ancillary files from the Moose git repository to the target directory.

    These are the files in the repo path `src/Moose/data`.

    Args:
        target_dir (Path|None): The target directory to download to. Must be a directory, not a file path. If missing, falls back to [USER_DIR][..]
        overwrite (bool): Flag to overwrite existing files, or not (default:False)
    """
    target_dir = USER_DIR if target_dir is None else target_dir
    if not target_dir.exists():
        raise OSError(f"{target_dir} does not exist, please create it first")
    session = Session()
    file_info = list_data_files_in_repo(session)
    errors = {}
    for info in tqdm(file_info):
        try:
            download_file(session, info, target_dir, overwrite)
        except (OSError, ValueError) as e:
            errors[info["name"]] = e.args[0]
    if errors != {}:
        print("Notice:")
        for file, error in errors.items():
            print(f"\t- {file}: {error}")


def migrate_file(file: Path, target_dir: Path, overwrite=False):
    """Copy a file to the directory `target_dir`.

    This fuction is intended to migrate bundled files to the new `moose-spectra` folder in the user directory.
    """
    if file.is_dir():
        raise OSError(f"{file=} is a directory, please provide a file path.")
    if not target_dir.is_dir() or not target_dir.exists():
        raise OSError(f"{target_dir=} is not an existing directory.")
    target_file = target_dir.joinpath(file.name)
    if not overwrite and target_file.exists():
        raise OSError(
            f"A file named {file.name} already exists in {target_dir}. Set `overwrite=True` if you want to ignore this error."
        )
    shutil.copy2(file, target_file)


def migrate():
    """Migrate files shipped with the Moose package to the user directory."""
    if not USER_DIR.exists():
        USER_DIR.mkdir(exist_ok=True)  # Do not raise if exist, though it should not at this point.
    for file in _PKGD_FILES:
        try:
            migrate_file(file, target_dir=USER_DIR)
        except OSError as e:
            if "already exists" in e.args[0]:
                continue
            else:
                raise


def download_databases(overwrite=False):
    """Download databases from the Moose repository to the `moose-spectra` folder in the user directory.

    Will first attempt to migrate the files shipped with Moose to avoid downloading.

    This latter behaviour will be deprecated after a transition period, when Moose stops shipping the databases.
    """
    if not USER_DIR.exists():
        USER_DIR.mkdir(exist_ok=True)  # Do not raise if exist, though it should not at this point.
    if not overwrite:
        # only migrate if this will not be overwritten
        migrate()
    download_databases_from_repo(USER_DIR, overwrite=overwrite)
