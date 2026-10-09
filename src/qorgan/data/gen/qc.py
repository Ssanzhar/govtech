"""Automated QC gates for a generated run (ADR D63). Every number in the pilot report comes
from here; `run_qc` writes nothing but returns a JSON-able report.

Gates (per row, aggregated per chunk and per run):
 1. schema       -- `rows.check_row`: schema, cell agreement, verbatim phrases, label rules.
 2. scrub        -- every utterance is a `scrub_text` fixed point (ADR D45).
 3. near_dup     -- e5 cosine to the nearest same-label row (in the run and in train) above a
                    threshold calibrated on train itself (no train pair exceeds it).
 4. novelty      -- token ROUGE-L F1 to the nearest same-language row (run + train) < 0.7.
 5. disjoint     -- normalised-transcript sha256 not in any processed split. HASHES ONLY: the
                    eval splits are hashed, never returned, printed or compared by content.
 6. lexicon_leak -- share of scam rows per scheme with NO hard-signal cue (target >= 0.30), and
                    the count of negatives on which a cue fires.
 7. fingerprint  -- cross-validated AUC of an LR telling run rows from Gemini train rows on
                    scrubbed text (char n-grams) and on embeddings; 0.5 = indistinguishable.
Also reported: role-slot agreement (Layer 2 vs the cell's role) and scheme agreement (Layer 3
rules vs the cell's scheme).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from qorgan.data.build_corpus import normalize_for_dedup
from qorgan.data.gen.grid import GridCell, read_manifest
from qorgan.data.gen.rows import RowCheck, check_row
from qorgan.data.schema import Dialogue
from qorgan.data.scrub import scrub_text

ROUGE_L_MAX = 0.7
LEAK_TARGET = 0.30
NEAR_DUP_FLOOR = 0.92  # research §3.1 starting point; raised to train's own max if that is higher
# Every processed split a generated row must not collide with (by content hash only).
DISJOINT_SPLITS = ("train", "val", "test", "authored_heldout", "ood", "shift", "adversarial", "adversarial_legit")
POOL_SPLIT = "train"
_AUGMENTED_SUFFIX = "-asr"  # ASR-styled copies of train rows (ADR D31): near-dups by construction
_TOKEN = re.compile(r"\w+")
_PROBE_FOLDS = 5
_BOOTSTRAP = 1000
_MIN_PROBE_ROWS = 60  # the 5-fold fingerprint probe needs enough rows per stratum (chunk QC skips it)


# ---------------------------------------------------------------- loading


def chunk_cell_ids(run_dir: Path, chunk: str) -> frozenset[str]:
    """The cell ids a chunk spec asks for (the spec is the contract the writer received)."""
    spec_path = run_dir / "chunks" / f"{chunk}.spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"no chunk spec {spec_path}")
    return frozenset(c["cell_id"] for c in json.loads(spec_path.read_text(encoding="utf-8"))["cells"])


def load_run_rows(run_dir: Path, chunk: str | None = None) -> tuple[tuple[GridCell, ...], list[tuple[str, dict]]]:
    """Manifest cells + raw rows; with `chunk`, only that chunk's cells and output file."""
    cells = read_manifest(run_dir / "manifest.jsonl")
    if chunk is not None:
        wanted = chunk_cell_ids(run_dir, chunk)
        cells = tuple(c for c in cells if c.cell_id in wanted)
    rows: list[tuple[str, dict]] = []
    for path in sorted((run_dir / "out").glob("*.jsonl")):
        if chunk is not None and path.stem != chunk:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append((path.stem, json.loads(line)))
            except json.JSONDecodeError:
                rows.append((path.stem, {"id": f"{path.stem}:{number}", "_unparseable": True}))
    return cells, rows


def load_split(processed: Path, name: str) -> list[Dialogue]:
    path = processed / f"{name}.jsonl"
    if not path.exists():
        return []
    return [Dialogue.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def transcript_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_dedup(text).encode("utf-8")).hexdigest()


def split_hashes(processed: Path, splits: Sequence[str] = DISJOINT_SPLITS) -> dict[str, frozenset[str]]:
    """{split: hashes}. Reads each split only to hash it; nothing about its content leaves."""
    out: dict[str, frozenset[str]] = {}
    for name in splits:
        path = processed / f"{name}.jsonl"
        if not path.exists():
            continue
        hashes = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                hashes.add(transcript_hash("\n".join(u["text"] for u in record["utterances"])))
        out[name] = frozenset(hashes)
    return out


# ---------------------------------------------------------------- pure gate helpers


def scrub_failures(dialogue: Dialogue) -> int:
    return sum(1 for u in dialogue.utterances if scrub_text(u.text) != u.text)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Token LCS length, bit-parallel (Allison-Dix / Hyyro) over Python big ints."""
    if not a or not b:
        return 0
    masks: dict[str, int] = {}
    for i, tok in enumerate(a):
        masks[tok] = masks.get(tok, 0) | (1 << i)
    full = (1 << len(a)) - 1
    v = full
    for tok in b:
        u = v & masks.get(tok, 0)
        v = ((v + u) | (v - u)) & full
    return len(a) - bin(v).count("1")


def rouge_l(a: Sequence[str], b: Sequence[str]) -> float:
    """ROUGE-L F1 (beta = 1) between token sequences."""
    lcs = lcs_length(a, b)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(b), lcs / len(a)
    return 2 * precision * recall / (precision + recall)


def max_rouge_l(query: Sequence[str], pool: Sequence[Sequence[str]]) -> float:
    return max((rouge_l(query, other) for other in pool), default=0.0)


def nearest_cosine(vectors: np.ndarray, pool: np.ndarray, *, exclude_self: bool) -> np.ndarray:
    """Max cosine of each row of `vectors` (unit norm) to `pool`; self-pairs dropped when the
    two matrices are the same set."""
    if len(vectors) == 0 or len(pool) == 0:
        return np.zeros(len(vectors), dtype=np.float32)
    sims = vectors @ pool.T
    if exclude_self:
        np.fill_diagonal(sims, -1.0)
    return sims.max(axis=1)


def cue_free_share(dialogues: Sequence[Dialogue], has_cue: Callable[[str], bool]) -> tuple[int, int]:
    """(rows with no hard-signal cue, rows)."""
    free = sum(1 for d in dialogues if not has_cue(d.transcript()))
    return free, len(dialogues)


def cv_auc(features, labels: np.ndarray, *, seed: int) -> dict:
    """Stratified k-fold out-of-fold AUC of a balanced LR, with a bootstrap 95 % interval."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    folds = StratifiedKFold(n_splits=_PROBE_FOLDS, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    scores = cross_val_predict(clf, features, labels, cv=folds, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(_BOOTSTRAP):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() != labels[idx].max():
            boots.append(roc_auc_score(labels[idx], scores[idx]))
    low, high = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (auc, auc)
    return {"auc": round(auc, 4), "ci95": [round(low, 4), round(high, 4)], "n_pos": int(labels.sum()), "n_neg": int(len(labels) - labels.sum())}


def matched_sample(pool: Sequence[Dialogue], target: Sequence[Dialogue], *, seed: int) -> list[Dialogue]:
    """Pool rows matching the target's (language, scam/legit) counts, seeded."""
    rng = np.random.default_rng(seed)
    out: list[Dialogue] = []
    for (language, scam), n in _strata(target).items():
        bucket = [d for d in pool if d.language == language and d.label.is_scam == scam]
        if bucket:
            picks = rng.choice(len(bucket), size=min(n, len(bucket)), replace=False)
            out.extend(bucket[i] for i in sorted(picks))
    return out


def _strata(dialogues: Sequence[Dialogue]) -> dict[tuple[str, bool], int]:
    counts: dict[tuple[str, bool], int] = {}
    for d in dialogues:
        key = (d.language, d.label.is_scam)
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------- embeddings


class EmbeddingStore:
    """ONNX e5 embeddings (the server's documented proxy of the browser embedder, ADR D32),
    cached by text sha256 in the run's parent directory (gitignored, never the shared cache)."""

    def __init__(self, cache_path: Path, embedder=None) -> None:
        self._path = cache_path
        self._embedder = embedder
        self._cache: dict[str, np.ndarray] = {}
        if cache_path.exists():
            data = np.load(cache_path)
            self._cache = {k: data[k] for k in data.files}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        from qorgan.classifier.embed import OnnxEmbedder, embed_texts
        from qorgan.config import get_config

        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._cache]
        if missing:
            if self._embedder is None:
                self._embedder = OnnxEmbedder.from_dir(get_config().embed_onnx_dir)
            vectors = embed_texts([t for _, t in missing], embedder=self._embedder)
            for (k, _), v in zip(missing, vectors):
                self._cache[k] = v
            self._path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self._path, **self._cache)
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack([self._cache[k] for k in keys]).astype(np.float32)


# ---------------------------------------------------------------- the report


def _schema_gate(cells, rows) -> tuple[list[RowCheck], dict]:
    from qorgan.taxonomy import get_taxonomy

    taxonomy = get_taxonomy()
    by_cell = {c.cell_id: c for c in cells}
    v2 = frozenset(taxonomy.label_space("v2"))
    hard = frozenset(taxonomy.hard_negative_ids())
    checks: list[RowCheck] = []
    per_chunk: dict[str, dict] = {}
    seen: set[str] = set()
    for chunk, raw in rows:
        if raw.get("_unparseable"):
            check = RowCheck(row_id=str(raw["id"]), errors=("unparseable JSON",))
        else:
            check = check_row(raw, by_cell, v2=v2, hard_negative_ids=hard)
            if check.row_id in seen:
                check = RowCheck(row_id=check.row_id, errors=("duplicate row id",))
            seen.add(check.row_id)
        checks.append(check)
        stats = per_chunk.setdefault(chunk, {"rows": 0, "valid": 0, "warnings": 0})
        stats["rows"] += 1
        stats["valid"] += int(check.dialogue is not None)
        stats["warnings"] += len(check.warnings)
    missing_cells = sorted(set(by_cell) - seen)
    return checks, {
        "rows": len(checks),
        "valid": sum(1 for c in checks if c.dialogue is not None),
        "missing_cells": missing_cells,
        "errors": {c.row_id: list(c.errors) for c in checks if c.errors},
        "warning_counts": _count(w.split(" ")[0] + " " + w.split(" ")[1] if " " in w else w for c in checks for w in c.warnings),
        "warnings": {c.row_id: list(c.warnings) for c in checks if c.warnings},
        "per_chunk": per_chunk,
    }


def _count(values) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _layer_agreement(valid: Sequence[Dialogue], cells: dict[str, GridCell]) -> dict:
    from qorgan.explain.roles import detect_role, get_role_lexicon
    from qorgan.explain.schemes import derive_scheme

    lexicon = get_role_lexicon()
    role_rows = [d for d in valid if cells[d.id].role is not None]
    role_hits = 0
    role_misses: dict[str, str | None] = {}
    for d in role_rows:
        claim = detect_role(d.transcript(), lexicon)
        got = claim.role_id if claim else None
        if got == cells[d.id].role:
            role_hits += 1
        else:
            role_misses[d.id] = got
    no_role = [d for d in valid if cells[d.id].role is None]
    no_role_fired = sum(1 for d in no_role if detect_role(d.transcript(), lexicon) is not None)
    scams = [d for d in valid if d.label.is_scam]
    scheme_hits = sum(1 for d in scams if derive_scheme([t.id for t in d.label.tactic_tags], cells[d.id].role) == cells[d.id].scheme)
    return {
        "role_slot": {"rows_with_role": len(role_rows), "detected_correctly": role_hits,
                      "recall": round(role_hits / len(role_rows), 4) if role_rows else None,
                      "misses": role_misses, "rows_without_role": len(no_role), "fired_without_role": no_role_fired},
        "scheme_rules": {"scam_rows": len(scams), "derived_equals_cell": scheme_hits},
    }


def _leak_gate(valid: Sequence[Dialogue], cells: dict[str, GridCell]) -> dict:
    from qorgan.classifier.cue_lexicon import load_cue_lexicon
    from qorgan.classifier.features import match_cues

    lexicon = load_cue_lexicon()

    def has_cue(text: str) -> bool:
        return bool(match_cues(text, lexicon))

    per_scheme: dict[str, dict] = {}
    for scheme in sorted({cells[d.id].scheme for d in valid if d.label.is_scam}):
        rows = [d for d in valid if cells[d.id].scheme == scheme]
        free, n = cue_free_share(rows, has_cue)
        per_scheme[scheme] = {"cue_free": free, "n": n, "share": round(free / n, 3)}
    scams = [d for d in valid if d.label.is_scam]
    free, n = cue_free_share(scams, has_cue)
    by_phrasing = {}
    for phrasing in ("direct", "indirect"):
        rows = [d for d in scams if cells[d.id].phrasing == phrasing]
        f, m = cue_free_share(rows, has_cue)
        by_phrasing[phrasing] = {"cue_free": f, "n": m}
    negatives = [d for d in valid if not d.label.is_scam]
    fired = [d.id for d in negatives if has_cue(d.transcript())]
    return {
        "target": LEAK_TARGET,
        "scams_cue_free": free, "scams": n, "share": round(free / n, 3) if n else None,
        "by_phrasing": by_phrasing,
        "schemes_below_target": sorted(k for k, v in per_scheme.items() if v["share"] < LEAK_TARGET),
        "per_scheme": per_scheme,
        "negatives_with_a_cue": len(fired), "negatives": len(negatives), "negative_ids_with_a_cue": fired,
    }


def _text_gates(valid, pool, cells) -> dict:
    pool_tokens = {lang: [tokens(d.transcript()) for d in pool if d.language == lang] for lang in ("kk", "ru", "mixed", "en")}
    run_tokens = {d.id: tokens(d.transcript()) for d in valid}
    scores: dict[str, float] = {}
    for d in valid:
        same_lang_run = [run_tokens[o.id] for o in valid if o.id != d.id and o.language == d.language]
        scores[d.id] = round(max(max_rouge_l(run_tokens[d.id], pool_tokens[d.language]), max_rouge_l(run_tokens[d.id], same_lang_run)), 4)
    flagged = sorted(k for k, v in scores.items() if v >= ROUGE_L_MAX)
    values = np.array(list(scores.values())) if scores else np.zeros(1)
    return {"threshold": ROUGE_L_MAX, "flagged": flagged, "max": float(values.max()), "mean": round(float(values.mean()), 4),
            "p95": round(float(np.percentile(values, 95)), 4), "per_row": scores}


def _near_dup_gate(valid, pool, store: EmbeddingStore) -> tuple[dict, np.ndarray, np.ndarray]:
    pool_vecs = store.embed([d.transcript() for d in pool])
    run_vecs = store.embed([d.transcript() for d in valid])
    pool_scam = np.array([d.label.is_scam for d in pool])
    run_scam = np.array([d.label.is_scam for d in valid])
    train_nn = np.concatenate([
        nearest_cosine(pool_vecs[pool_scam == s], pool_vecs[pool_scam == s], exclude_self=True) for s in (True, False)
    ])
    train_max = float(train_nn.max())
    threshold = max(NEAR_DUP_FLOOR, round(train_max, 4))
    nn_pool = np.zeros(len(valid))
    nn_run = np.zeros(len(valid))
    for s in (True, False):
        idx = np.where(run_scam == s)[0]
        if len(idx):
            nn_pool[idx] = nearest_cosine(run_vecs[idx], pool_vecs[pool_scam == s], exclude_self=False)
            nn_run[idx] = nearest_cosine(run_vecs[idx], run_vecs[idx], exclude_self=True) if len(idx) > 1 else 0.0
    flagged_pool = [valid[i].id for i in range(len(valid)) if nn_pool[i] > threshold]
    flagged_run = [valid[i].id for i in range(len(valid)) if nn_run[i] > threshold]
    return {
        "train_within_label_nn": {"max": round(train_max, 4), "p99": round(float(np.percentile(train_nn, 99)), 4),
                                  "p50": round(float(np.percentile(train_nn, 50)), 4), "pairs_over_0.92": int((train_nn > NEAR_DUP_FLOOR).sum())},
        "threshold": threshold,
        "run_vs_train_nn": {"max": round(float(nn_pool.max()), 4), "p50": round(float(np.median(nn_pool)), 4)},
        "run_vs_run_nn": {"max": round(float(nn_run.max()), 4), "p50": round(float(np.median(nn_run)), 4)},
        "flagged_vs_train": flagged_pool, "flagged_within_run": flagged_run,
    }, run_vecs, pool_vecs


# Schemes / negative classes whose topic already exists in the v1 (Gemini) corpus. The probe on
# this subset separates generator STYLE from the new TOPICS the v2 grid adds on purpose.
V1_TOPICS = frozenset({
    "bank_safe_account", "bank_card_phish", "law_enforcement_threat", "investment_broker",
    "prize_compensation", "mule_recruitment", "legit_bank_call", "legit_gov_service",
    "family_money_request", "delivery_notification", "general_conversation",
})


def _probe(rows, pool, seed: int, store: EmbeddingStore | None) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer

    sample = matched_sample(pool, rows, seed=seed)
    texts = [scrub_text(d.transcript()) for d in rows] + [scrub_text(d.transcript()) for d in sample]
    labels = np.array([1] * len(rows) + [0] * len(sample))
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True).fit_transform(texts)
    out = {"char_ngram": cv_auc(char, labels, seed=seed)}
    if store is not None:
        out["embedding"] = cv_auc(store.embed([d.transcript() for d in list(rows) + sample]), labels, seed=seed)
    return out


def _fingerprint_gate(valid, pool, seed: int, store: EmbeddingStore | None, cells: dict[str, GridCell]) -> dict:
    overlap = [d for d in valid if (cells[d.id].scheme or cells[d.id].negative_class) in V1_TOPICS]
    return {
        "comparison": "run (Claude) vs Gemini train rows matched on language x label",
        **_probe(valid, pool, seed, store),
        "v1_topics_only": {"rows": len(overlap), **_probe(overlap, pool, seed, store)},
    }


def row_verdicts(report: dict, valid_ids: Sequence[str]) -> dict[str, list[str]]:
    """Row-level rejections: {row id: reasons}. A row is accepted when it is schema-valid, a
    scrub fixed point, not a near-duplicate (vs train or the run) and below the ROUGE-L cap.
    Pure over the report; schema-invalid rows carry their schema errors."""
    reasons: dict[str, list[str]] = {rid: ["schema: " + "; ".join(errs)] for rid, errs in report["schema"]["errors"].items()}
    for rid in report["scrub"]["rows_not_fixed_point"]:
        reasons.setdefault(rid, []).append("scrub: not a fixed point")
    near = report.get("near_dup", {})
    for rid in near.get("flagged_vs_train", []):
        reasons.setdefault(rid, []).append("near-duplicate of a train row")
    for rid in near.get("flagged_within_run", []):
        reasons.setdefault(rid, []).append("near-duplicate within the run")
    for rid in report["novelty_rouge_l"]["flagged"]:
        reasons.setdefault(rid, []).append("ROUGE-L >= 0.7")
    for name, ids in report["disjoint"]["overlaps"].items():
        for rid in ids:
            reasons.setdefault(rid, []).append(f"collides with a {name} row (hash)")
    return {rid: reasons[rid] for rid in sorted(reasons)}


def chunk_verdict(n_cells: int, rejected: Mapping[str, list[str]], valid_ids: Sequence[str], min_share: float) -> dict:
    accepted = sorted(set(valid_ids) - set(rejected))
    share = len(accepted) / n_cells if n_cells else 0.0
    return {"cells": n_cells, "accepted": len(accepted), "share": round(share, 4), "min_share": min_share,
            "verdict": "pass" if share >= min_share else "fail", "accepted_ids": accepted}


def run_qc(run_dir: Path, *, embed: bool = True, probe_seed: int = 42, processed: Path | None = None, chunk: str | None = None) -> dict:
    from qorgan.config import GEN_CHUNK_MIN_VALID_SHARE, get_config

    processed = processed or get_config().data_dir / "processed"
    cells, rows = load_run_rows(run_dir, chunk)
    by_cell = {c.cell_id: c for c in cells}
    checks, schema = _schema_gate(cells, rows)
    valid = [c.dialogue for c in checks if c.dialogue is not None]
    scrub = {d.id: scrub_failures(d) for d in valid}
    hashes = split_hashes(processed)
    disjoint = {name: sorted(d.id for d in valid if transcript_hash(d.transcript()) in hs) for name, hs in hashes.items()}
    pool = [d for d in load_split(processed, POOL_SPLIT) if not d.id.endswith(_AUGMENTED_SUFFIX)]
    report = {
        "run": run_dir.name,
        "schema": schema,
        "scrub": {"rows_not_fixed_point": sorted(k for k, v in scrub.items() if v), "utterances_not_fixed_point": sum(scrub.values())},
        "disjoint": {"splits_checked": sorted(hashes), "overlaps": {k: v for k, v in disjoint.items() if v}},
        "lexicon_leak": _leak_gate(valid, by_cell),
        "layers": _layer_agreement(valid, by_cell),
        "novelty_rouge_l": _text_gates(valid, pool, by_cell),
        "pool": {"split": POOL_SPLIT, "rows": len(pool), "excluded": f"*{_AUGMENTED_SUFFIX} styled copies"},
    }
    store = EmbeddingStore(run_dir.parent / "_cache" / "onnx_embeddings.npz") if embed else None
    if store is not None:
        report["near_dup"] = _near_dup_gate(valid, pool, store)[0]
    report["fingerprint"] = _fingerprint_gate(valid, pool, probe_seed, store, by_cell) if len(valid) >= _MIN_PROBE_ROWS else {"skipped": f"< {_MIN_PROBE_ROWS} rows"}
    report["chunk"] = chunk
    report["rejected"] = row_verdicts(report, [d.id for d in valid])
    report["acceptance"] = chunk_verdict(len(cells), report["rejected"], [d.id for d in valid], GEN_CHUNK_MIN_VALID_SHARE)
    report["summary"] = _summary(report, len(cells))
    return report


def _summary(report: dict, n_cells: int) -> dict:
    leak = report["lexicon_leak"]
    near = report.get("near_dup", {})
    return {
        "cells": n_cells,
        "rows": report["schema"]["rows"],
        "schema_valid": report["schema"]["valid"],
        "scrub_failures": len(report["scrub"]["rows_not_fixed_point"]),
        "eval_overlaps": sum(len(v) for v in report["disjoint"]["overlaps"].values()),
        "near_dup_flagged": len(near.get("flagged_vs_train", [])) + len(near.get("flagged_within_run", [])) if near else None,
        "rouge_l_flagged": len(report["novelty_rouge_l"]["flagged"]),
        "cue_free_scam_share": leak["share"],
        "schemes_below_leak_target": leak["schemes_below_target"],
        "negatives_with_a_cue": leak["negatives_with_a_cue"],
        "role_slot_recall": report["layers"]["role_slot"]["recall"],
        "scheme_rule_agreement": f"{report['layers']['scheme_rules']['derived_equals_cell']}/{report['layers']['scheme_rules']['scam_rows']}",
        "fingerprint_auc_char": report["fingerprint"].get("char_ngram", {}).get("auc"),
        "fingerprint_auc_embedding": report["fingerprint"].get("embedding", {}).get("auc"),
        "accepted": report["acceptance"]["accepted"],
        "rejected": len(report["rejected"]),
        "verdict": report["acceptance"]["verdict"],
    }
