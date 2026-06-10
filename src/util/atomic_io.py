import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def atomic_write(path: str) -> Iterator[str]:
    """Yield a temporary path that is atomically renamed to ``path`` on success.

    The temporary file is created in the destination directory so that
    ``os.replace`` is a same-filesystem rename, which is atomic on POSIX.
    If the block raises, the temporary file is removed and any existing
    file at ``path`` is left untouched, so readers never observe a
    partially written file.

    Usage:
        with atomic_write("./cache/data.parquet") as tmp:
            df.to_parquet(tmp)
    """
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=os.path.basename(path) + ".", suffix=".tmp")
    os.close(fd)
    # mkstemp creates the file with 0600; restore the umask-derived default
    # so the final file's permissions match a plain open() for writing.
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp_path, 0o666 & ~umask)
    try:
        yield tmp_path
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
