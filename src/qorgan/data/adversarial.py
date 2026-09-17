"""Adversarial paraphrases: scam calls rewritten to avoid every hard-signal cue
(PLAN_2026-09 A9, ADR D15 "assume the adversary has the lexicon").

The device tier is public by construction, so an attacker can read
`data/lexicon/hard_signal_cues.yaml` and script around it. This module builds the eval set
that measures what that costs us: each scam in `test` / `ood` is paraphrased (Gemini,
build-time) under the constraint that none of the cue phrases survive, the result is
verified *locally* with the same matcher the classifier uses (`features.match_cues`),
retried with the leaked cues named, scrubbed, and stored with the source label so recall
can be compared pair by pair (`qorgan.eval.adversarial`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from qorgan.classifier.cue_lexicon import CueLexicon
from qorgan.data.build_corpus import scrub_dialogue
from qorgan.data.schema import Dialogue, Label, Utterance

ADVERSARIAL_ID_PREFIX = "adv-"
ADVERSARIAL_SPLIT = "adversarial"
DEFAULT_MAX_ATTEMPTS = 3
_TRUTH_THRESHOLD = 0.5
# Letters that exist in Kazakh Cyrillic but not in Russian: enough to tell the two apart
# (a paraphrase that switched language would corrupt the per-language rows).
_KAZAKH_LETTERS = frozenset("әғқңөұүһі")
_KAZAKH_SHARE_RU_MAX = 0.005  # a Russian text has ~none
_KAZAKH_SHARE_KK_MIN = 0.02  # a Kazakh text has plenty

Paraphraser = Callable[[str], dict[str, Any]]


class ParaphraseFailure(RuntimeError):
    """No cue-free, schema-valid paraphrase within the attempt budget."""


def source_positives(dialogues: Sequence[Dialogue]) -> tuple[Dialogue, ...]:
    return tuple(d for d in dialogues if not d.label.is_hard_negative and d.label.risk >= _TRUTH_THRESHOLD)


def cue_hits(text: str, lexicon: CueLexicon) -> tuple[str, ...]:
    """Every lexicon cue present in `text` (case-insensitive substring, the classifier's rule)."""
    lowered = text.lower()
    return tuple(cue for cues in lexicon.entries.values() for cue in cues if cue.lower() in lowered)


def language_matches(text: str, language: str) -> bool:
    """Cheap script check: `ru` must carry (almost) no Kazakh-only letters, `kk` must carry
    a clear share of them; `mixed` accepts anything."""
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return False
    share = sum(1 for c in letters if c in _KAZAKH_LETTERS) / len(letters)
    if language == "ru":
        return share <= _KAZAKH_SHARE_RU_MAX
    if language == "kk":
        return share >= _KAZAKH_SHARE_KK_MIN
    return True


_LANGUAGE_NAMES = {"ru": "Russian only", "kk": "Kazakh only", "mixed": "the same Kazakh/Russian mix"}
_LANGUAGE_VIOLATION = "(wrong language)"


def build_paraphrase_prompt(source: Dialogue, lexicon: CueLexicon, *, previous_hits: Sequence[str] = ()) -> str:
    tactics = ", ".join(tag.id for tag in source.label.tactic_tags) or "the same scam intent"
    cues = "\n".join(f"- {cue}" for cues in lexicon.entries.values() for cue in cues)
    dialogue = "\n".join(f"{u.speaker}: {u.text}" for u in source.utterances)
    retry = ""
    if previous_hits:
        leaked = [h for h in previous_hits if h != _LANGUAGE_VIOLATION]
        notes = []
        if leaked:
            notes.append("still contained these forbidden phrases: " + "; ".join(leaked) + ". Remove them and every close variant")
        if _LANGUAGE_VIOLATION in previous_hits:
            notes.append(f"was not written in {_LANGUAGE_NAMES[source.language]} -- keep the original language")
        retry = "\nYour previous attempt " + "; and it ".join(notes) + ".\n"
    return (
        "You are rewriting a scripted scam phone call for an adversarial robustness test of a "
        "fraud-detection model. Rewrite the conversation so that a fraud analyst would still "
        f"recognise the SAME scam tactics ({tactics}) and the same goal, written in "
        f"{_LANGUAGE_NAMES[source.language]} (language code {source.language}), with the same speakers, the same "
        "number of turns, similar length and the same realism.\n"
        "HARD CONSTRAINT: the new text must not contain ANY of the following phrases, nor a "
        "trivially close variant (a different case, spelling or word ending still counts):\n"
        f"{cues}\n"
        "Express the same requests indirectly -- paraphrase, euphemism, implication, splitting a "
        "request across turns -- but keep the scam's goal intact. Do not add anything a real "
        "institution would say to reassure the customer.\n"
        f"{retry}"
        "Return JSON only: {\"utterances\": [{\"speaker\": \"caller|callee\", \"text\": \"...\"}]}\n\n"
        f"Conversation:\n{dialogue}"
    )


def paraphrase_dialogue(
    source: Dialogue,
    lexicon: CueLexicon,
    *,
    paraphrase: Paraphraser,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Dialogue:
    """Cue-free, scrubbed paraphrase of `source` carrying its label (spans dropped)."""
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")
    hits: tuple[str, ...] = ()
    for _ in range(max_attempts):
        payload = paraphrase(build_paraphrase_prompt(source, lexicon, previous_hits=hits))
        candidate = _build(source, payload)
        if candidate is None:
            hits = ()
            continue
        hits = cue_hits(candidate.transcript(), lexicon)
        if not language_matches(candidate.transcript(), source.language):
            hits = (*hits, _LANGUAGE_VIOLATION)
        if not hits:
            return scrub_dialogue(candidate)
    raise ParaphraseFailure(f"{source.id}: no cue-free paraphrase in {max_attempts} attempts (last hits: {hits})")


def _build(source: Dialogue, payload: dict[str, Any]) -> Dialogue | None:
    raw = payload.get("utterances") if isinstance(payload, dict) else None
    utterances = [
        Utterance(speaker=str(item["speaker"]).strip(), text=str(item["text"]).strip())
        for item in (raw or [])
        if isinstance(item, dict) and str(item.get("speaker", "")).strip() and str(item.get("text", "")).strip()
    ]
    if not utterances:
        return None
    try:
        return Dialogue(
            id=f"{ADVERSARIAL_ID_PREFIX}{source.id}",
            language=source.language,
            utterances=tuple(utterances),
            label=Label(risk=source.label.risk, tactic_tags=source.label.tactic_tags, trigger_spans=(), is_hard_negative=False),
        )
    except ValidationError:
        return None
