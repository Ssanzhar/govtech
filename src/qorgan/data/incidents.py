"""Deterministic Level-2 incident synthesis (D5-1).

L2 works over *incidents* (a scored call with linking metadata: phone number, timestamp,
script-family id) rather than dialogues. This module manufactures a realistic incident
stream from a few scam **script families** -- each family reuses a small pool of phone
numbers (so incidents in a family share numbers -> the number graph links them into one
"organization") and a pool of transcripts (so they also cluster on text). A family flagged
`is_novel` is a *new scheme*: its incidents land in the recent window so novelty + recency
detectors can surface it.

Given the same `seed`, output is byte-stable. Real transcripts/numbers are supplied by
`scripts/demo_seed.py`; the synthesis logic here is pure and deterministic.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from qorgan.data.schema import Incident, Label

# Scam risk assigned to every synthesized incident (they are confirmed scams for L2).
_INCIDENT_RISK = 0.9
# Novel-family incidents land in the last fraction of the window (a fresh, growing scheme).
_NOVEL_RECENT_FRACTION = 0.8
_SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class ScriptFamily:
    """One scam script family: a pool of transcripts + reused phone numbers (a would-be
    organization). `weight` sets its relative incident volume; `is_novel` marks a new scheme."""

    id: str
    transcripts: tuple[str, ...]
    phone_numbers: tuple[str, ...]
    is_novel: bool = False
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.transcripts:
            raise ValueError(f"ScriptFamily {self.id!r} needs at least one transcript")
        if not self.phone_numbers:
            raise ValueError(f"ScriptFamily {self.id!r} needs at least one phone number")


def synthesize_incidents(
    families: Sequence[ScriptFamily],
    *,
    count: int,
    seed: int,
    start_time: datetime,
    span_days: float = 30.0,
) -> list[Incident]:
    """Sample `count` incidents across `families`, deterministically for `seed`.

    Each incident draws a transcript + phone number from its family's pools and a timestamp
    in `[start_time, start_time + span_days]` (novel families only in the recent tail).
    Raises `ValueError` if `families` is empty or `count <= 0`.
    """
    if not families:
        raise ValueError("families must not be empty")
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")

    rng = random.Random(seed)
    family_list = list(families)
    weights = [max(0.0, family.weight) for family in family_list]
    span_seconds = span_days * _SECONDS_PER_DAY

    incidents: list[Incident] = []
    for index in range(count):
        family = rng.choices(family_list, weights=weights, k=1)[0]
        incidents.append(
            _build_incident(
                index=index,
                family=family,
                transcript=rng.choice(family.transcripts),
                phone_number=rng.choice(family.phone_numbers),
                timestamp=_sample_timestamp(rng, start_time, span_seconds, family.is_novel),
            )
        )
    return incidents


def _sample_timestamp(
    rng: random.Random, start_time: datetime, span_seconds: float, is_novel: bool
) -> datetime:
    low = _NOVEL_RECENT_FRACTION * span_seconds if is_novel else 0.0
    offset = rng.uniform(low, span_seconds)
    return start_time + timedelta(seconds=offset)


def _build_incident(
    *, index: int, family: ScriptFamily, transcript: str, phone_number: str, timestamp: datetime
) -> Incident:
    return Incident(
        id=f"inc_{index:04d}",
        dialogue_id=f"{family.id}_{index}",
        transcript=transcript,
        label=Label(risk=_INCIDENT_RISK, is_hard_negative=False),
        phone_number=phone_number,
        timestamp=timestamp,
        script_family=family.id,
    )
