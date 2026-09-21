"""Generate golden parity fixtures for the on-device JS core (PLAN_2026-09 B2).

For a fixed set of transcripts this records, from the *Python* implementation:
- every embedding the real e5 model produced (keyed by text, base64 float32) -- the JS
  tests feed these back as their "embedder", so the comparison isolates the port from the
  embedding runtime;
- `predict.score()` output and `explain()` for ru + kk;
- the live-session trajectory (`live.session.advance` per utterance): window text, meter
  score / latch / hard signals, band, tags, new evidence, recommendation.

Run: `QORGAN_CLASSIFIER_BACKEND=linear python scripts/export_parity_fixtures.py`
Outputs: tests_js/fixtures/parity.json (+ site/models/weights.json copied from the bundle).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from qorgan.asr.stream import CommittedUtterance
from qorgan.classifier import embed as embed_mod
from qorgan.classifier import predict
from qorgan.config import get_config
from qorgan.data.demo_transcripts import LIVE_DEMO_CALLS
from qorgan.data.schema import UTTERANCE_JOIN, Dialogue
from qorgan.explain.explainer import explain
from qorgan.live.session import advance, initial_session

REPO_ROOT = Path(__file__).resolve().parents[1]
_E5_PREFIX = "query: "
_AUTHORED_PER_CLASS = 10
_TEST_CASES = 6
# The cross-runtime decision gate (tests_js/integration, PLAN B3: same decision on >= 99 %)
# needs statistical power the 28 golden trajectories cannot give: 200 transcripts with
# Python's risk/tags only (Node embeds them itself), all authored + seeded test/ood samples.
RUNTIME_GATE_FILENAME = "runtime_gate.json"
_RUNTIME_GATE_CASES = 200
_RUNTIME_GATE_SEED = 42


class RecordingEmbedder:
    """Delegates to the real SentenceTransformer and remembers every vector by text."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.vectors: dict[str, np.ndarray] = {}

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        out = self._inner.encode(texts, normalize_embeddings=normalize_embeddings, convert_to_numpy=convert_to_numpy)
        for text, vector in zip(texts, out):
            key = text[len(_E5_PREFIX):] if text.startswith(_E5_PREFIX) else text
            self.vectors[key] = np.asarray(vector, dtype=np.float32)
        return out


def _load(path: Path) -> list[Dialogue]:
    return [Dialogue.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select_cases(processed: Path) -> list[dict[str, Any]]:
    authored = _load(processed / "authored_heldout.jsonl")
    scams = [d for d in authored if d.label.risk >= 0.5][:_AUTHORED_PER_CLASS]
    legit = [d for d in authored if d.label.risk < 0.5][:_AUTHORED_PER_CLASS]
    test = _load(processed / "test.jsonl")[:_TEST_CASES]
    cases = [
        {"id": d.id, "locale": "kk" if d.language == "kk" else "ru", "utterances": [u.text for u in d.utterances]}
        for d in [*scams, *legit, *test]
    ]
    for name, text in LIVE_DEMO_CALLS.items():
        cases.append({"id": name, "locale": "kk" if name.endswith("_kk") else "ru", "utterances": text.split(UTTERANCE_JOIN)})
    return cases


def _select_runtime_gate(processed: Path) -> list[Dialogue]:
    import random

    authored = _load(processed / "authored_heldout.jsonl")
    rest = [*_load(processed / "test.jsonl"), *_load(processed / "ood.jsonl")]
    random.Random(_RUNTIME_GATE_SEED).shuffle(rest)
    return [*authored, *rest][:_RUNTIME_GATE_CASES]


def export_runtime_gate(processed: Path, out: Path) -> int:
    """`{cases: [{id, transcript, risk, flagged, tags}]}` -- Python's single-shot verdicts."""
    threshold = get_config().risk_threshold
    cases = []
    for d in _select_runtime_gate(processed):
        transcript = d.transcript()
        result = predict.score(transcript, backend="linear")
        cases.append({
            "id": d.id, "language": d.language, "transcript": transcript, "risk": result.risk,
            "flagged": result.risk >= threshold, "tags": [{"id": t.id, "weight": t.weight} for t in result.tags],
        })
    out.write_text(json.dumps({"threshold": threshold, "embed_backend": get_config().embed_backend, "cases": cases}, ensure_ascii=False), encoding="utf-8")
    return len(cases)


def _score_case(case: dict[str, Any]) -> dict[str, Any]:
    transcript = UTTERANCE_JOIN.join(case["utterances"])
    result = predict.score(transcript, backend="linear")
    explanations = {}
    for locale in ("ru", "kk"):
        e = explain(result, transcript, locale)
        explanations[locale] = {
            "reason": e.reason, "confidence_label": e.confidence_label, "caveat": e.caveat,
            "human_note": e.human_note, "highlights": [s.text for s in e.highlights],
        }
    state = initial_session(case["locale"], backend="linear")
    turns = []
    for text in case["utterances"]:
        state, update = advance(state, CommittedUtterance(text=text, confidence=1.0))
        turns.append({
            "window_text": update.window_text,
            "score": update.meter.score, "latched": update.meter.latched,
            "hard_signal_ids": sorted(update.meter.hard_signal_ids), "band": update.band,
            "risk": update.result.risk,
            "tags": [{"id": t.id, "weight": t.weight} for t in update.result.tags],
            "new_evidence": [s.text for s in update.new_evidence],
            "recommendation": update.recommendation.model_dump(),
        })
    return {
        **case,
        "transcript": transcript,
        "score": {
            "risk": result.risk, "raw_confidence": result.raw_confidence,
            "tags": [{"id": t.id, "weight": t.weight} for t in result.tags],
            "attributions": [{"text": s.text, "start": s.start, "end": s.end} for s in result.attributions],
        },
        "explanations": explanations,
        "live": {"tags": [{"id": t.id, "weight": t.weight} for t in state.tags], "turns": turns},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export JS parity fixtures.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "tests_js" / "fixtures" / "parity.json")
    parser.add_argument("--runtime-gate-out", type=Path, default=REPO_ROOT / "tests_js" / "fixtures" / RUNTIME_GATE_FILENAME)
    args = parser.parse_args(argv)
    cfg = get_config()
    n_gate = export_runtime_gate(cfg.data_dir / "processed", args.runtime_gate_out)
    print(f"runtime gate: {n_gate} transcripts -> {args.runtime_gate_out}")

    inner = embed_mod.get_embedder()  # the configured backend (int8 ONNX by default)
    recorder = RecordingEmbedder(inner)
    for key, value in list(embed_mod._MODEL_CACHE.items()):
        if value is inner:
            embed_mod._MODEL_CACHE[key] = recorder  # every embed_texts() call now records

    cases = [_score_case(case) for case in _select_cases(cfg.data_dir / "processed")]
    weights_src = cfg.linear_model_dir / "web" / "weights.json"
    weights_dst = REPO_ROOT / "site" / "models" / "weights.json"
    weights_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weights_src, weights_dst)

    payload = {
        "format_version": 1,
        "embed_model_name": cfg.embed_model_name,
        "embed_backend": cfg.embed_backend,
        "weights_sha256": hashlib.sha256(weights_dst.read_bytes()).hexdigest(),
        "config": {
            "risk_threshold": cfg.risk_threshold, "enter": cfg.risk_threshold_enter, "exit": cfg.risk_threshold_exit,
            "alpha_up": cfg.meter_alpha_up, "alpha_down": cfg.meter_alpha_down,
        },
        "embeddings": {text: base64.b64encode(vec.tobytes()).decode("ascii") for text, vec in recorder.vectors.items()},
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"{len(cases)} cases, {len(recorder.vectors)} embeddings -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"weights -> {weights_dst}")


if __name__ == "__main__":
    main()
