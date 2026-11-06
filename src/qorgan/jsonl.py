"""Reading the JSONL stores (citizen reports, analyst feedback, the Level-2 seeds).

Two things every loader must get right, so they live in one place:
- only "\\n" ends a record. `str.splitlines` also breaks on U+2028, U+2029 and U+0085, which JSON
  leaves unescaped inside a string: one pasted transcript used to make its whole store unreadable;
- a corrupt line is refused by number, never by content. A parse error quotes its input, and these
  files hold call text, so it would reach HTTP responses and logs past the open-case gate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

_T = TypeVar("_T")
_LOGGER = logging.getLogger(__name__)
_MAX_SKIPPED_RECORDS = 2  # a torn write or two; 1 % of a large store


def read_records(path: Path, parse: Callable[[str], _T], *, skip_corrupt: bool = False) -> list[_T]:
    """`parse` applied to each non-blank line of `path`. Raises `ValueError` on a corrupt line, or with
    `skip_corrupt` leaves it out: for a deletion, which must not wait for a repair and rewrites the file
    without it. A line nobody can read may still hold what is being forgotten, so losing it is the
    safe direction. Only a few, though: a store that is mostly unreadable means a schema mismatch, and
    rewriting it would erase everything, so that still raises."""
    records: list[_T] = []
    corrupt = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        try:
            records.append(parse(line))
        except ValueError:
            if not skip_corrupt:
                raise ValueError(f"{path.name}: corrupt record at line {number}") from None
            corrupt += 1
            _LOGGER.warning("%s: leaving out the corrupt record at line %d", path.name, number)
    if corrupt > max(_MAX_SKIPPED_RECORDS, (len(records) + corrupt) // 100):
        raise ValueError(f"{path.name}: {corrupt} corrupt records, too many to leave out")
    return records


def count_records(path: Path) -> int:
    """Non-blank lines of `path`, readable or not."""
    return sum(1 for line in path.read_text(encoding="utf-8").split("\n") if line.strip())
