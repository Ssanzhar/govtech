"""Hard-signal cue lexicon for the hybrid `linear` classifier feature.

Loads + validates a small multilingual (RU/KK/mixed) lexicon of **request-oriented** phrases
per hard-signal tactic -- asks a legitimate call never makes (OTP / credentials / safe-account
/ remote-access / secrecy). This lexicon is the offline, deterministic knowledge that
`features.py` turns into 5 interpretable cue features. It is seeded from the taxonomy examples
(and optionally expanded via Gemini at build time), then committed as data, so inference needs
no API/network.

`lexicon_hash` gives a content fingerprint the trained bundle stores, so loading a model
against a drifted lexicon can be caught (train/inference feature consistency).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from qorgan.taxonomy import get_taxonomy


class CueLexiconError(ValueError):
    """Raised when the cue lexicon is missing, malformed, or inconsistent with the taxonomy."""


class CueLexicon(BaseModel):
    """Validated request-cue phrases per hard-signal tactic (immutable)."""

    model_config = ConfigDict(frozen=True)

    version: int
    entries: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def _entries_non_blank(self) -> "CueLexicon":
        if not self.entries:
            raise ValueError("cue lexicon has no entries")
        for tactic_id, cues in self.entries.items():
            if not cues:
                raise ValueError(f"cue lexicon entry {tactic_id!r} has no cues")
            for cue in cues:
                if not cue or not cue.strip():
                    raise ValueError(f"cue lexicon entry {tactic_id!r} has a blank cue")
        return self


def load_cue_lexicon(path: Path | None = None) -> CueLexicon:
    """Load + validate the cue lexicon YAML (defaults to `config.cue_lexicon_path`).

    Validates that its tactics are exactly the taxonomy's hard-signal tactics (none missing,
    none unknown) and that no cue is blank. Raises `CueLexiconError` on any problem.
    """
    from qorgan.config import get_config

    resolved = path or get_config().cue_lexicon_path
    if not resolved.exists():
        raise CueLexiconError(f"Cue lexicon file not found: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CueLexiconError(f"Invalid YAML in {resolved}: {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("cues"), dict):
        raise CueLexiconError(f"Cue lexicon {resolved} must be a mapping with a 'cues' mapping")

    entries = {str(tactic_id): tuple(cues or ()) for tactic_id, cues in raw["cues"].items()}
    try:
        lexicon = CueLexicon(version=int(raw.get("version", 1)), entries=entries)
    except (ValidationError, ValueError) as exc:
        raise CueLexiconError(f"Cue lexicon validation failed for {resolved}: {exc}") from exc

    _validate_against_taxonomy(lexicon, resolved)
    return lexicon


def _validate_against_taxonomy(lexicon: CueLexicon, path: Path) -> None:
    expected = set(get_taxonomy().hard_signal_ids())
    present = set(lexicon.entries)
    missing = expected - present
    unknown = present - expected
    if missing:
        raise CueLexiconError(f"Cue lexicon {path} is missing hard-signal tactics: {sorted(missing)}")
    if unknown:
        raise CueLexiconError(
            f"Cue lexicon {path} has unknown / non-hard-signal tactics: {sorted(unknown)}"
        )


def lexicon_hash(lexicon: CueLexicon) -> str:
    """Content fingerprint of a lexicon: sha256 over its cues, order-independent.

    Reordering tactics or the cues within a tactic does not change the hash; adding or
    changing a cue does.
    """
    canonical = json.dumps(
        {tactic_id: sorted(cues) for tactic_id, cues in lexicon.entries.items()},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
