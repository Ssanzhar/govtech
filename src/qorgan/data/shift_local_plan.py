"""Deterministic generation plan for the `shift_local` held-out splits (ADR D65).

Every call slot (language x brief x persona x outcome x length x seed) is fixed by one seed
before any text is generated, so the set is reproducible and no slot can be chosen after
seeing an output. The label of a slot comes from its brief.

`LOCAL_SPLITS` registers every batch: `shift_local` (D65), `shift_local_2` (PLAN_2026-10 §4
A3: same briefs, same model and sampling, a new seed, kk/ru only after D70) and `shift_local_3`
(A7, the v2.2 judge: the second batch's design under another seed) and `shift_local_4` (ADR
D73, the v3 judge: the same briefs plus `shift_local_briefs_v3.yaml`, 160 calls) and
`shift_local_5` (ADR D74, the v3.1 judge: those plus `shift_local_briefs_v31.yaml`). A batch is
planned with `plan_for(name, load_split_briefs(name, configs_dir))`; the first batch's plan is
pinned by a fingerprint test.

The generator is a local open-weights model from a different family than the training data
(Gemini) and the planned training generator (Claude); see `GENERATORS`.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

PLAN_SEED = 20261009
# `mixed` was planned at 40. The only local Kazakh-capable model never produced kk-ru
# code-switching in the pilot (0 / 8 calls, both inter- and intra-sentence prompts; it falls back
# to pure Kazakh), so mixed is 0 here and stays pending for court_authored / real_asr (ADR D65).
LANGUAGE_TARGETS: dict[str, int] = {"kk": 50, "ru": 50, "mixed": 0, "en": 30}
TURN_RANGE = (8, 18)
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_TOKENS = 2200
MAX_ATTEMPTS = 3
ATTEMPT_SEED_STRIDE = 100_003
SCAM_RISK = 0.95
LEGIT_RISK = 0.02

# Which local model writes which language. The plan was two families (Qwen-Kazakh for kk/mixed,
# YandexGPT for ru/en); the link delivered ~0.1-1 MB/s for a day and the YandexGPT download was
# dropped, so ONE family (Qwen3.5, ISSAI's Kazakh fine-tune) writes every language (ADR D65).
# YandexGPT stays registered so a second family can be added for ru/en without code changes.
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
LANGUAGE_GENERATOR: dict[str, str] = {"kk": "qwen35_kk", "mixed": "qwen35_kk", "ru": "qwen35_kk", "en": "qwen35_kk"}

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
# The Kazakh fine-tune answers in Kazakh unless told otherwise in the target language itself
# (pilot, plan seed 7: 0 / 6 ru and mixed calls in the right language with English-only
# instructions), so every instruction carries a native-language directive. Within-sentence
# code-switching was never produced (0 / 5 pilot calls); `mixed` therefore asks for the
# speaker-level switch (Russian caller, Kazakh answerer, slips both ways) -- ADR D65.
LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "kk": "Kazakh only, in Cyrillic script, everyday spoken Kazakh. Диалог тек қазақ тілінде болсын.",
    "ru": (
        "Russian only, not Kazakh. The call takes place in Kazakhstan. "
        "Весь диалог пиши только на русском языке, без казахских слов и фраз."
    ),
    "mixed": (
        "Kazakh-Russian code-switching as it is heard in Kazakhstan: the caller speaks mostly Russian, "
        "the person answering replies mostly in Kazakh, and both slip into the other language at times. "
        "Cyrillic script. Звонящий говорит по-русски, ал жауап беруші негізінен қазақша сөйлейді."
    ),
    "en": (
        "English only, not Kazakh or Russian. The call takes place in Kazakhstan "
        "(Kazakh names, tenge, local places). Write every line in English."
    ),
}
SYSTEM_PROMPT = (
    "You are a screenwriter who writes realistic, natural telephone conversations for a public-awareness "
    "research project in Kazakhstan. You write only the conversation itself."
)
_SYSTEM_LANGUAGE = {
    "kk": "Сен диалогты тек қазақ тілінде жазасың.",
    "ru": "Ты пишешь диалог только на русском языке. Никакого казахского текста.",
    "mixed": (
        "Звонящий говорит по-русски. Жауап беретін адам қазақша сөйлейді, кейде орысша сөздер қосады."
    ),
    "en": "You write the conversation in English only.",
}


@dataclass(frozen=True)
class LocalSplit:
    """One batch written by the local model: its seed, slot-id prefix, language targets, the
    earlier batches it must not near-duplicate, and the cross-split near-duplicate threshold its
    build enforces (None = exact-overlap only, as `shift_local` was built under D65)."""

    name: str
    plan_seed: int
    slot_prefix: str
    language_targets: Mapping[str, int]
    generator_note: str
    adr: str
    prior: tuple[str, ...] = ()
    near_duplicate_jaccard: float | None = None
    plan_task: str | None = None
    briefs_files: tuple[str, ...] = ("shift_local_briefs.yaml",)  # under configs/, concatenated in order


_NOTE = (
    "generated {date} offline with llama.cpp by local open-weights models from non-Gemini, non-Claude "
    "families (see data/provenance/{name}.json); briefs from public scam descriptions; no access to the "
    "corpus, prompts or lexicons; accepted by automated checks only"
)
SHIFT_LOCAL_2_SEED = 20261022
# The v2.2 judge (D63 v2.1 A6 verdict, PLAN_2026-10 A7): `shift_local_2` was looked at once by v2.1.
SHIFT_LOCAL_3_SEED = 20261024
# D70: kk/ru only. The English slots go to Kazakh, the weakest language (PLAN_2026-10 §0); mixed
# stays 0 for the D65 reason (the model never code-switches). Same total as `shift_local`.
SHIFT_LOCAL_2_TARGETS: dict[str, int] = {
    "kk": LANGUAGE_TARGETS["kk"] + LANGUAGE_TARGETS["en"], "ru": LANGUAGE_TARGETS["ru"], "mixed": 0, "en": 0,
}
# ADR D73: the v3 judge. 160 calls so each v3 tactic gets several scams and each twin several
# look-alikes next to the old briefs; kk-heavy like the second batch.
SHIFT_LOCAL_4_SEED = 20261029
SHIFT_LOCAL_4_TARGETS: Mapping[str, int] = {"kk": 96, "ru": 64, "mixed": 0, "en": 0}
# ADR D74: the v3.1 judge. `shift_local_4` was looked at once by v3; same design and size, plus
# family and everyday legitimate briefs (the classes A6 (c) failed on).
SHIFT_LOCAL_5_SEED = 20261101
LOCAL_SPLITS: Mapping[str, LocalSplit] = MappingProxyType({
    "shift_local": LocalSplit(
        name="shift_local", plan_seed=PLAN_SEED, slot_prefix="sl",
        language_targets=MappingProxyType(LANGUAGE_TARGETS),
        generator_note=_NOTE.format(date="2026-10-09", name="shift_local"), adr="D65",
    ),
    "shift_local_2": LocalSplit(
        name="shift_local_2", plan_seed=SHIFT_LOCAL_2_SEED, slot_prefix="sl2",
        language_targets=MappingProxyType(SHIFT_LOCAL_2_TARGETS),
        generator_note=_NOTE.format(date="2026-10-22", name="shift_local_2"), adr="D65",
        prior=("shift_local",), near_duplicate_jaccard=0.7, plan_task="PLAN_2026-10 §4 A3",
    ),
    "shift_local_3": LocalSplit(
        name="shift_local_3", plan_seed=SHIFT_LOCAL_3_SEED, slot_prefix="sl3",
        language_targets=MappingProxyType(SHIFT_LOCAL_2_TARGETS),
        generator_note=_NOTE.format(date="2026-10-23", name="shift_local_3"), adr="D65",
        prior=("shift_local", "shift_local_2"), near_duplicate_jaccard=0.7, plan_task="PLAN_2026-10 A7",
    ),
    "shift_local_4": LocalSplit(
        name="shift_local_4", plan_seed=SHIFT_LOCAL_4_SEED, slot_prefix="sl4",
        language_targets=MappingProxyType(SHIFT_LOCAL_4_TARGETS),
        generator_note=_NOTE.format(date="2026-10-24", name="shift_local_4"), adr="D73",
        prior=("shift_local", "shift_local_2", "shift_local_3"), near_duplicate_jaccard=0.7, plan_task="ADR D73",
        briefs_files=("shift_local_briefs.yaml", "shift_local_briefs_v3.yaml"),
    ),
    "shift_local_5": LocalSplit(
        name="shift_local_5", plan_seed=SHIFT_LOCAL_5_SEED, slot_prefix="sl5",
        language_targets=MappingProxyType(SHIFT_LOCAL_4_TARGETS),
        generator_note=_NOTE.format(date="2026-10-24", name="shift_local_5"), adr="D74",
        prior=("shift_local", "shift_local_2", "shift_local_3", "shift_local_4"), near_duplicate_jaccard=0.7,
        plan_task="ADR D74",
        briefs_files=("shift_local_briefs.yaml", "shift_local_briefs_v3.yaml", "shift_local_briefs_v31.yaml"),
    ),
})


def system_prompt(language: str) -> str:
    return f"{SYSTEM_PROMPT} {_SYSTEM_LANGUAGE[language]}"
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
    persona: str | None = None  # pins the callee (D73: a call to a child); the draw still happens


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
            persona=b.get("persona"),
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


def load_split_briefs(split: str, configs_dir: Path) -> tuple[Brief, ...]:
    """Every brief file of a registered batch, concatenated in order; ids stay unique."""
    briefs = tuple(b for name in LOCAL_SPLITS[split].briefs_files for b in load_briefs(configs_dir / name))
    ids = [b.id for b in briefs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{split}: duplicate brief ids across {LOCAL_SPLITS[split].briefs_files}")
    return briefs


def build_plan(
    briefs: tuple[Brief, ...], *, seed: int = PLAN_SEED, targets: Mapping[str, int] | None = None, prefix: str = "sl",
) -> tuple[Slot, ...]:
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
                brief = order[index % len(order)]
                drawn = rng.choice(PERSONAS)  # always drawn, so a pinned persona moves no other slot
                slots.append(Slot(
                    slot_id=f"{prefix}_{language}_{label}_{index + 1:02d}",
                    language=language,
                    brief=brief,
                    persona=brief.persona or drawn,
                    outcome=rng.choice(outcomes),
                    turns=rng.randint(*TURN_RANGE),
                    seed=rng.randrange(1, 2**31 - ATTEMPT_SEED_STRIDE * MAX_ATTEMPTS),
                    generator=LANGUAGE_GENERATOR[language],
                ))
    return tuple(slots)


def plan_for(split: str, briefs: tuple[Brief, ...]) -> tuple[Slot, ...]:
    """The fixed plan of a registered batch (`KeyError` for an unknown split)."""
    spec = LOCAL_SPLITS[split]
    return build_plan(briefs, seed=spec.plan_seed, targets=spec.language_targets, prefix=spec.slot_prefix)


def user_prompt(slot: Slot) -> str:
    return "\n".join((
        f"Language: {LANGUAGE_INSTRUCTIONS[slot.language]}",
        "Write one telephone conversation.",
        f"Situation: {slot.brief.brief}",
        _LABEL_NOTES[slot.brief.label],
        f"The person who answers is {slot.persona}. {slot.outcome}",
        f"Length: about {slot.turns} turns in total, alternating speakers.",
        "Format: one turn per line. Start every line with CALLER: or CALLEE: followed by the spoken words. "
        "No title, no narration, no stage directions, no explanations before or after.",
        f"Remember the language: {LANGUAGE_INSTRUCTIONS[slot.language]}",
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
