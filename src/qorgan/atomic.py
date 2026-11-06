"""Whole-file rewrites that can never leave half a file (QA 2026-09-29, N5).

The JSONL stores (citizen reports, analyst feedback, the Level-2 seeds) are rewritten whole on
delete / purge / feedback. A crash or a full disk in the middle of a plain `write_text` left a
truncated store; here the text goes to a temp file in the same directory, is fsynced, and
replaces the target with one atomic `os.replace` -- a reader sees the old file or the new one.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO


def write_text_atomic(path: Path, text: str) -> None:
    write_binary_atomic(path, lambda handle: handle.write(text.encode("utf-8")))


def write_binary_atomic(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Whatever `write` puts in the handle replaces `path` in one step, or `path` is untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp)
        raise
