"""Source Level-2 script families from the corpus + a hand-authored novel scheme, and seed
the incident stream (D5-1). `scripts/demo_seed.py` is the thin CLI over this.

Known families group corpus *scam* transcripts by tactic and each get a small reused phone
pool (so incidents in a family share numbers). The novel family (`crypto_giveaway_new`) is a
scheme deliberately absent from the training taxonomy, so the L2 novelty detector should
surface it as a "new scheme".
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from qorgan.data.incidents import ScriptFamily, synthesize_incidents
from qorgan.data.schema import Dialogue, Incident
from qorgan.data.scrub import scrub_text

# (family id, tactic ids that define it, operating phone numbers). Each family operates
# from its own line(s), disjoint across families, so the phone-number co-occurrence graph
# yields one organization per family -- the reliable real-world link, since scam script
# TEXT is too semantically similar across families for embeddings to separate (see
# docs/eval_report.md). A family's incidents all reuse its number(s) -> one component.
_FAMILY_DEFINITIONS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("bank_security", ("impersonation_bank", "otp_request", "credentials_request"),
     ("+7 700 101 20 30",)),
    ("police_finpol", ("impersonation_gov_police", "fear_threat"),
     ("+7 701 202 30 40",)),
    ("investment_qr", ("investment_scam", "payment_redirect"),
     ("+7 702 303 40 50",)),
    ("telecom_delivery", ("impersonation_telecom_delivery",),
     ("+7 705 404 50 60",)),
    ("prize_social", ("prize_lottery", "mule_recruitment"),
     ("+7 707 505 60 70",)),
)

_MAX_TRANSCRIPTS_PER_FAMILY = 60

# A new scheme the model has never seen (no matching taxonomy family) -> should be flagged novel.
_NOVEL_FAMILY = ScriptFamily(
    id="crypto_giveaway_new",
    transcripts=(
        "Здравствуйте! Известный предприниматель проводит раздачу криптовалюты. Пришлите "
        "0.05 биткоина на этот кошелёк и получите вдвое больше обратно в течение часа.",
        "Только сегодня официальная крипто-раздача: отправьте монеты на указанный адрес и "
        "мгновенно удвойте баланс, количество участников ограничено.",
        "Сәлеметсіз бе! Криптовалюта раздачасы: әмиянға аударыңыз, екі есе қайтарамыз, тек бүгін.",
    ),
    phone_numbers=("+7 708 909 10 11",),
    is_novel=True,
    weight=0.6,
)


def build_families_from_dialogues(dialogues: Sequence[Dialogue]) -> list[ScriptFamily]:
    """Group corpus scam dialogues into script families by tactic; append the novel family.

    A family's `weight` is the number of matching corpus dialogues, so bigger families
    generate proportionally more incidents. Families with no matching transcripts are
    dropped. Raises `ValueError` if nothing matched (empty/negative corpus).

    Every corpus-derived transcript is passed through `scrub_text` before it enters a
    `ScriptFamily`, so no raw PII (phone numbers, IINs, cards, emails) ever reaches
    `Incident.transcript` / `Organization.representative_script` (CLAUDE.md SS6/SS9).
    """
    scams = [d for d in dialogues if not d.label.is_hard_negative]
    families: list[ScriptFamily] = []
    for family_id, tactics, numbers in _FAMILY_DEFINITIONS:
        tactic_set = set(tactics)
        transcripts = [
            scrub_text(d.transcript())
            for d in scams
            if {t.id for t in d.label.tactic_tags} & tactic_set
        ]
        if not transcripts:
            continue
        families.append(
            ScriptFamily(
                id=family_id,
                transcripts=tuple(transcripts[:_MAX_TRANSCRIPTS_PER_FAMILY]),
                phone_numbers=numbers,
                # weight = full pre-truncation match count (the family's real-world size), so a
                # family whose pool is capped at _MAX_TRANSCRIPTS_PER_FAMILY still samples
                # proportionally more incidents. Intentional: volume tracks size, not pool cap.
                weight=float(len(transcripts)),
            )
        )
    if not families:
        raise ValueError("No scam dialogues matched any script family; is the corpus empty?")
    families.append(_NOVEL_FAMILY)
    return families


def seed_incidents(
    dialogues: Sequence[Dialogue],
    *,
    count: int,
    seed: int,
    start_time: datetime,
    hmac_key: bytes,
    span_days: float = 30.0,
) -> list[Incident]:
    """Build families from `dialogues` and synthesize `count` incidents (numbers hashed with `hmac_key`)."""
    families = build_families_from_dialogues(dialogues)
    return synthesize_incidents(
        families, count=count, seed=seed, start_time=start_time, hmac_key=hmac_key, span_days=span_days
    )


def write_incidents_jsonl(incidents: Sequence[Incident], path: Path) -> None:
    """Write incidents as JSONL (one per line), creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(i.model_dump_json() for i in incidents) + "\n", encoding="utf-8")


def load_incidents_jsonl(path: Path) -> list[Incident]:
    """Load an incidents JSONL written by `write_incidents_jsonl`."""
    if not path.exists():
        raise FileNotFoundError(f"Incidents not found: {path}. Run scripts/demo_seed.py first.")
    return [Incident.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
