"""Deterministic generation plan for the `shift_local` held-out split (ADR D65).

Every call slot (language x brief x persona x outcome x length x seed) is fixed by one seed
before any text is generated, so the set is reproducible and no slot can be chosen after
seeing an output. The label of a slot comes from its brief.

The generator is a local open-weights model from a different family than the training data
(Gemini) and the planned training generator (Claude); see `GENERATORS`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml

PLAN_SEED = 20261009
LANGUAGE_TARGETS: dict[str, int] = {"kk": 50, "ru": 50, "mixed": 40, "en": 30}
TURN_RANGE = (8, 18)
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 2200
MAX_ATTEMPTS = 3
ATTEMPT_SEED_STRIDE = 100_003
SCAM_RISK = 0.95
LEGIT_RISK = 0.02

# Which local model writes which language. No single model writes everything.
GENERATORS: dict[str, dict[str, str]] = {
    "qwen35_kk": {
        "repo": "mradermacher/Qwen3.5-9B-Kazakh-GGUF",
        "revision": "b68107190e1e995f4d2576bccd8b4b1e27805956",
        "file": "Qwen3.5-9B-Kazakh.Q4_K_M.gguf",
        "sha256": "4e3ca9f6cb59d9ffb00d0301ae5dd59e47ee2a818153a7d5caa76c74f02021e2",
        "base": "issai/Qwen3.5-9B-Kazakh (ISSAI, Qwen3.5 family, Kazakh continued pre-training + instruct)",
        "licence": "apache-2.0",
    },
    "yandexgpt5_lite": {
        "repo": "yandex/YandexGPT-5-Lite-8B-instruct-GGUF",
        "revision": "9fe287d2f512503046bb008aed350f2b4bbb903d",
        "file": "YandexGPT-5-Lite-8B-instruct-Q4_K_M.gguf",
        "sha256": "d9ff5b826f20fbcc2f898f9f2349ac21241579f8fbb79cd32148a333623ba228",
        "base": "yandex/YandexGPT-5-Lite-8B-instruct (Yandex, own pre-training, Russian-first)",
        "licence": "YandexGPT-5-Lite-8B license (custom; see the repo's LICENSE)",
    },
}
LANGUAGE_GENERATOR: dict[str, str] = {"kk": "qwen35_kk", "mixed": "qwen35_kk", "ru": "yandexgpt5_lite", "en": "yandexgpt5_lite"}

PERSONAS: tuple[str, ...] = (
    "a 67-year-old retired schoolteacher in Shymkent",
    "a 34-year-old taxi driver in Almaty",
    "a 22-year-old university student in Astana",
    "a 45-year-old accountant in Karaganda",
    "a 58-year-old farmer near Taraz",
    "a 29-year-old young mother on maternity leave in Aktobe",
    "a 71-year-old pensioner living alone in Pavlodar",
    "a 39-year-old small shop owner in Atyrau",
    "a 51-year-old nurse in Semey",
    "a 26-year-old IT specialist in Almaty",
    "a 63-year-old retired miner in Temirtau",
    "a 33-year-old office manager in Kostanay",
)
SCAM_OUTCOMES: tuple[str, ...] = (
    "They believe the caller and start to go along with the request.",
    "They are confused and ask many questions, but keep listening.",
    "They grow suspicious halfway through and refuse.",
    "They say they will call their bank or relative back first, and end the call.",
    "They are in a hurry and almost agree before the call ends.",
)
LEGIT_OUTCOMES: tuple[str, ...] = (
    "They are friendly and the call ends normally.",
    "They are a little wary at first, but the caller's answers reassure them.",
    "They are busy and keep the call short.",
    "They ask a few practical questions before agreeing.",
)
LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "kk": "Kazakh only, in Cyrillic script, everyday spoken Kazakh.",
    "ru": "Russian only. The call takes place in Kazakhstan.",
    "mixed": (
        "Kazakh-Russian code-switching, the way many people in Almaty and Astana talk: both speakers "
        "mix Kazakh and Russian words and sentences throughout the call. Cyrillic script."
    ),
    "en": "English only. The call takes place in Kazakhstan (Kazakh names, tenge, local places).",
}
SYSTEM_PROMPT = (
    "You are a screenwriter who writes realistic, natural telephone conversations for a public-awareness "
    "research project in Kazakhstan. You write only the conversation itself."
)
_LABEL_NOTES = {
    "scam": "The caller is a fraudster but never admits it; the call should sound natural, the way real calls do.",
    "legit": "The caller is genuine and acts in good faith.",
}


@dataclass(frozen=True)
class Brief:
    id: str
    label: str
    brief: str
    tactics: tuple[str, ...] = ()
    hard_negative: bool = True


@dataclass(frozen=True)
class Slot:
    slot_id: str
    language: str
    brief: Brief
    persona: str
    outcome: str
    turns: int
    seed: int
    generator: str

    @property
    def risk(self) -> float:
        return SCAM_RISK if self.brief.label == "scam" else LEGIT_RISK

    @property
    def is_hard_negative(self) -> bool:
        return self.brief.label == "legit" and self.brief.hard_negative

    def attempt_seed(self, attempt: int) -> int:
        return self.seed + attempt * ATTEMPT_SEED_STRIDE


def load_briefs(path: Path) -> tuple[Brief, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    briefs = tuple(
        Brief(
            id=b["id"], label=b["label"], brief=b["brief"].strip(),
            tactics=tuple(b.get("tactics", ())), hard_negative=bool(b.get("hard_negative", True)),
        )
        for b in data["briefs"]
    )
    ids = [b.id for b in briefs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate brief ids")
    bad = [b.id for b in briefs if b.label not in _LABEL_NOTES or (b.label == "legit" and b.tactics)]
    if bad:
        raise ValueError(f"briefs with an unknown label or legit tactics: {bad}")
    return briefs


def build_plan(briefs: tuple[Brief, ...], *, seed: int = PLAN_SEED, targets: dict[str, int] | None = None) -> tuple[Slot, ...]:
    """Half scam, half legit per language (odd counts give legit the extra slot); each label's
    briefs are cycled in a per-language shuffled order so every brief is used before any repeats."""
    rng = random.Random(seed)
    by_label = {label: [b for b in briefs if b.label == label] for label in ("scam", "legit")}
    slots: list[Slot] = []
    for language, total in (targets or LANGUAGE_TARGETS).items():
        counts = {"scam": total // 2, "legit": total - total // 2}
        for label in ("scam", "legit"):
            order = by_label[label][:]
            rng.shuffle(order)
            outcomes = SCAM_OUTCOMES if label == "scam" else LEGIT_OUTCOMES
            for index in range(counts[label]):
                slots.append(Slot(
                    slot_id=f"sl_{language}_{label}_{index + 1:02d}",
                    language=language,
                    brief=order[index % len(order)],
                    persona=rng.choice(PERSONAS),
                    outcome=rng.choice(outcomes),
                    turns=rng.randint(*TURN_RANGE),
                    seed=rng.randrange(1, 2**31 - ATTEMPT_SEED_STRIDE * MAX_ATTEMPTS),
                    generator=LANGUAGE_GENERATOR[language],
                ))
    return tuple(slots)


def user_prompt(slot: Slot) -> str:
    return "\n".join((
        "Write one telephone conversation.",
        f"Situation: {slot.brief.brief}",
        _LABEL_NOTES[slot.brief.label],
        f"The person who answers is {slot.persona}. {slot.outcome}",
        f"Language: {LANGUAGE_INSTRUCTIONS[slot.language]}",
        f"Length: about {slot.turns} turns in total, alternating speakers.",
        "Format: one turn per line. Start every line with CALLER: or CALLEE: followed by the spoken words. "
        "No title, no narration, no stage directions, no explanations before or after.",
    ))


def raw_row(slot: Slot, utterances: tuple[tuple[str, str], ...]) -> dict:
    """The row in the raw shape `qorgan.data.shift_set.parse_raw_row` reads. No trigger
    phrases: the label is brief-level, and nobody picks spans out of a held-out call."""
    return {
        "id": slot.slot_id,
        "language": slot.language,
        "scenario": slot.brief.id,
        "utterances": [{"speaker": s, "text": t} for s, t in utterances],
        "risk": slot.risk,
        "is_hard_negative": slot.is_hard_negative,
        "tactic_tags": [{"id": t, "weight": 1.0} for t in slot.brief.tactics],
        "trigger_phrases": [],
    }
