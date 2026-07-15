"""Unified classifier interface — the one contract every caller (app, eval) depends on.

`score(transcript) -> ScoreResult` routes to a backend selected via
`QORGAN_CLASSIFIER_BACKEND` (config-driven, default `llm`). Swapping the backend must
never touch callers.

Backends:
- `llm`   -- Gemini structured classifier (D1-5, `llm_classifier.py`). Ships first.
- `mock`  -- deterministic, no-network, no-API-key backend for demos/tests. Returns
             hand-authored canned results for the bundled demo transcripts and a
             taxonomy-keyword heuristic for arbitrary pasted text.
- `xlmr`  -- fine-tuned XLM-R backend (D3). Loads the exported bundle from
             `config.xlmr_model_dir`, returns calibrated risk + decoded tactic tags +
             Captum IG trigger spans. Raises `XlmrModelNotFoundError` (clear fallback
             message) if no model has been trained/exported yet. All torch imports are
             lazy, so the `llm`/`mock` paths never load torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qorgan.classifier import llm_classifier
from qorgan.config import get_config
from qorgan.data.demo_transcripts import DEMO_TRANSCRIPTS
from qorgan.data.schema import ScoreResult, Span, TacticTag
from qorgan.taxonomy import get_taxonomy

_SUPPORTED_BACKENDS = ("llm", "xlmr", "linear", "mock")
# How many IG trigger spans to surface for an xlmr verdict.
_XLMR_ATTRIBUTION_TOP_K = 8
# `linear` backend: highlight up to this many utterances scoring at/above the margin.
_LINEAR_ATTRIBUTION_TOP_K = 3
_LINEAR_ATTRIBUTION_MIN_SCORE = 0.5

# Risk levels used by the deterministic mock heuristic. Not a "trained" model -- just
# enough signal to make the no-API-key demo path grounded and non-trivial.
_MOCK_RISK_HARD_SIGNAL = 0.9
_MOCK_RISK_MULTI_TACTIC = 0.6
_MOCK_RISK_SINGLE_TACTIC = 0.35
_MOCK_RISK_NO_SIGNAL = 0.05


class UnknownBackendError(ValueError):
    """Raised when a requested classifier backend name is not recognized."""


def score(transcript: str, *, backend: str | None = None) -> ScoreResult:
    """Score `transcript` for scam risk using the configured (or overridden) backend."""
    if not transcript or not transcript.strip():
        raise ValueError("transcript must not be empty")

    cfg = get_config()
    active_backend = backend or cfg.classifier_backend

    if active_backend == "llm":
        return llm_classifier.classify(transcript)
    if active_backend == "mock":
        return _mock_score(transcript)
    if active_backend == "xlmr":
        return _xlmr_score(transcript)
    if active_backend == "linear":
        return _linear_score(transcript)
    raise UnknownBackendError(
        f"Unknown classifier backend {active_backend!r}; expected one of {_SUPPORTED_BACKENDS}"
    )


class XlmrModelNotFoundError(RuntimeError):
    """Raised when the `xlmr` backend is selected but no exported model is present."""


@dataclass(frozen=True)
class XlmrBundle:
    """Everything the `xlmr` backend needs at inference, loaded once from the export dir."""

    model: Any
    tokenizer: Any
    label_space: tuple[str, ...]
    max_length: int
    tactic_threshold: float
    temperature: float
    device: Any


_XLMR_BUNDLE_CACHE: dict[str, XlmrBundle] = {}


def _xlmr_score(transcript: str, *, bundle: XlmrBundle | None = None) -> ScoreResult:
    """Score `transcript` with the fine-tuned XLM-R bundle: calibrated risk + tactic tags +
    IG trigger spans. `bundle` is injectable for tests; production loads it lazily/cached."""
    import torch

    from qorgan.classifier import calibrate
    from qorgan.classifier.attribution import integrated_gradient_spans
    from qorgan.classifier.labels import decode_tactics

    active = bundle or _get_xlmr_bundle()
    active.model.to(active.device)
    active.model.eval()

    enc = active.tokenizer(
        transcript, return_tensors="pt", truncation=True, max_length=active.max_length
    )
    input_ids = enc["input_ids"].to(active.device)
    attention_mask = enc["attention_mask"].to(active.device)
    with torch.no_grad():
        out = active.model(input_ids=input_ids, attention_mask=attention_mask)

    risk_logit = float(out["risk_logit"].reshape(-1)[0].item())
    tactic_logits = out["tactic_logits"].reshape(-1).tolist()
    calibrated_risk = calibrate.apply_temperature([risk_logit], active.temperature)[0]
    tactic_probs = calibrate.apply_temperature(tactic_logits, 1.0)  # plain sigmoid for tactics
    tags = tuple(
        TacticTag(id=tactic_id, weight=min(1.0, max(0.0, prob)))
        for tactic_id, prob in decode_tactics(tactic_probs, active.label_space, active.tactic_threshold)
    )
    spans = integrated_gradient_spans(
        active.model,
        active.tokenizer,
        transcript,
        device=active.device,
        max_length=active.max_length,
        top_k=_XLMR_ATTRIBUTION_TOP_K,
    )
    # Calibrated certainty of the decision (either direction), in [0.5, 1.0].
    confidence = max(calibrated_risk, 1.0 - calibrated_risk)
    return ScoreResult(
        risk=calibrated_risk,
        tags=tags,
        attributions=spans,
        backend="xlmr",
        raw_confidence=confidence,
    )


def _get_xlmr_bundle() -> XlmrBundle:
    cfg = get_config()
    key = str(cfg.xlmr_model_dir)
    if key not in _XLMR_BUNDLE_CACHE:
        _XLMR_BUNDLE_CACHE[key] = load_xlmr_bundle(cfg.xlmr_model_dir)
    return _XLMR_BUNDLE_CACHE[key]


def load_xlmr_bundle(model_dir) -> XlmrBundle:  # pragma: no cover - loads the real model
    """Reconstruct the `ScamClassifierModel` + tokenizer from an exported bundle dir.

    The architecture is rebuilt from the base model's *config only* (no ~1GB weight
    download), then the fine-tuned `state_dict` is loaded over it.
    """
    import json

    import torch
    from transformers import AutoTokenizer

    from qorgan.classifier.model import ScamClassifierModel, build_encoder_from_config

    metadata_path = model_dir / "metadata.json"
    weights_path = model_dir / "model.pt"
    if not metadata_path.exists() or not weights_path.exists():
        raise XlmrModelNotFoundError(
            f"No exported XLM-R model in {model_dir}. Train one with "
            "`python -m qorgan.classifier.train`, or set QORGAN_CLASSIFIER_BACKEND=llm|mock."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    device = torch.device("cpu")
    encoder = build_encoder_from_config(metadata["base_model"])
    model = ScamClassifierModel(encoder, metadata["num_tactics"])
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(metadata["base_model"])
    return XlmrBundle(
        model=model,
        tokenizer=tokenizer,
        label_space=tuple(metadata["label_space"]),
        max_length=metadata["max_length"],
        tactic_threshold=metadata["tactic_threshold"],
        temperature=metadata["temperature"],
        device=device,
    )


_LINEAR_BUNDLE_CACHE: dict[str, Any] = {}


def _linear_score(transcript: str, *, bundle: Any = None, embedder: Any = None) -> ScoreResult:
    """Score `transcript` with the embeddings + calibrated-LR backend: calibrated risk +
    decoded tactic tags + grounded top-utterance highlights. For a hybrid bundle the risk head
    consumes `[embedding | cue features | reassurance]` and fired cues are merged in as grounded
    tags/spans. `bundle`/`embedder` are injectable for tests; production loads them cached."""
    from qorgan.classifier import embed as embed_mod
    from qorgan.classifier import features as feat
    from qorgan.classifier.attribution import select_top_utterance_spans
    from qorgan.classifier.labels import decode_tactics
    from qorgan.data.schema import UTTERANCE_JOIN

    active = bundle or _get_linear_bundle()
    utterances = transcript.split(UTTERANCE_JOIN)

    if active.hard_signal_enabled:
        kwargs = dict(
            embedder=embedder,
            model_name=active.embed_model_name,
            lexicon=active.lexicon,
            reassurance_patterns=active.reassurance_patterns,
        )
        blocks = feat.compute_feature_blocks([transcript], **kwargs)
        risk = float(active.risk_clf.predict_proba(feat.hybrid_matrix(blocks))[0, 1])
        tactic_probs = active.tactic_clf.predict_proba(blocks.embedding)[0].tolist()
        utterance_blocks = feat.compute_feature_blocks(utterances, **kwargs)
        utterance_scores = active.risk_clf.predict_proba(feat.hybrid_matrix(utterance_blocks))[:, 1].tolist()
        cue_matches = blocks.matches[0]
    else:
        features = embed_mod.embed_texts([transcript], embedder=embedder, model_name=active.embed_model_name)
        risk = float(active.risk_clf.predict_proba(features)[0, 1])
        tactic_probs = active.tactic_clf.predict_proba(features)[0].tolist()
        utterance_features = embed_mod.embed_texts(utterances, embedder=embedder, model_name=active.embed_model_name)
        utterance_scores = active.risk_clf.predict_proba(utterance_features)[:, 1].tolist()
        cue_matches = ()

    tags = [
        TacticTag(id=tactic_id, weight=min(1.0, max(0.0, prob)))
        for tactic_id, prob in decode_tactics(tactic_probs, active.label_space, active.tactic_threshold)
    ]
    spans = list(
        select_top_utterance_spans(
            utterances, utterance_scores,
            top_k=_LINEAR_ATTRIBUTION_TOP_K, min_score=_LINEAR_ATTRIBUTION_MIN_SCORE,
        )
    )
    tags, spans = _merge_cue_evidence(tags, spans, cue_matches)

    return ScoreResult(
        risk=risk,
        tags=tuple(tags),
        attributions=tuple(spans),
        backend="linear",
        raw_confidence=max(risk, 1.0 - risk),  # calibrated certainty of the decision
    )


def _merge_cue_evidence(tags: list, spans: list, cue_matches) -> tuple[list, list]:
    """Fold fired hard-signal cues into the tags + highlight spans as grounded evidence.

    A matched cue is a verbatim span tied to a tactic, so that tactic is tagged at
    weight 1.0 — added if missing, *upgraded* if the tactic head already tagged it with
    lower confidence (a verbatim lexicon hit is stronger evidence than the embedding
    head, and downstream hard-signal handling keys on this weight). Span dedup as before;
    spans returned in transcript order.
    """
    tag_ids = {tag.id for tag in tags}
    merged_tags = list(tags)
    span_keys = {(span.start, span.end) for span in spans}
    merged_spans = list(spans)
    for match in cue_matches:
        if match.tactic_id not in tag_ids:
            merged_tags.append(TacticTag(id=match.tactic_id, weight=1.0))
            tag_ids.add(match.tactic_id)
        key = (match.span.start, match.span.end)
        if key not in span_keys:
            merged_spans.append(match.span)
            span_keys.add(key)
    cue_ids = {match.tactic_id for match in cue_matches}
    merged_tags = [
        TacticTag(id=tag.id, weight=1.0) if tag.id in cue_ids else tag for tag in merged_tags
    ]
    merged_spans.sort(key=lambda span: span.start)
    return merged_tags, merged_spans


def _get_linear_bundle() -> Any:
    from qorgan.classifier.linear_train import load_linear

    cfg = get_config()
    key = str(cfg.linear_model_dir)
    if key not in _LINEAR_BUNDLE_CACHE:
        _LINEAR_BUNDLE_CACHE[key] = load_linear(cfg.linear_model_dir)
    return _LINEAR_BUNDLE_CACHE[key]


def _span(transcript: str, phrase: str) -> Span:
    """Build a `Span` for `phrase` by locating it verbatim in `transcript`.

    Raises `ValueError` (via `str.index`) if `phrase` is not actually present -- a
    build-time self-check for the hand-authored canned results below.
    """
    start = transcript.index(phrase)
    return Span(text=phrase, start=start, end=start + len(phrase))


def _canned_results() -> dict[str, ScoreResult]:
    bank_ru = DEMO_TRANSCRIPTS["scam_bank_ru"]
    investment = DEMO_TRANSCRIPTS["scam_investment_kk_ru"]
    hard_negative = DEMO_TRANSCRIPTS["hard_negative_bank_call_ru"]

    return {
        bank_ru: ScoreResult(
            risk=0.96,
            tags=[
                TacticTag(id="impersonation_bank"),
                TacticTag(id="urgency"),
                TacticTag(id="secrecy"),
                TacticTag(id="otp_request"),
                TacticTag(id="safe_account"),
            ],
            attributions=[
                _span(bank_ru, "это служба безопасности вашего банка"),
                _span(bank_ru, "действовать нужно прямо сейчас"),
                _span(bank_ru, "Никому не говорите об этом звонке"),
                _span(bank_ru, "Продиктуйте код из SMS"),
                _span(bank_ru, "переведите деньги на безопасный счёт"),
            ],
            backend="mock",
            raw_confidence=None,
        ),
        investment: ScoreResult(
            risk=0.82,
            tags=[
                TacticTag(id="investment_scam"),
                TacticTag(id="payment_redirect"),
                TacticTag(id="urgency"),
            ],
            attributions=[
                _span(investment, "Гарантированный доход 30% в месяц"),
                _span(investment, "оплатить по этому QR-коду прямо сейчас"),
            ],
            backend="mock",
            raw_confidence=None,
        ),
        hard_negative: ScoreResult(
            risk=0.03,
            tags=[],
            attributions=[],
            backend="mock",
            raw_confidence=None,
        ),
    }


def _mock_score(transcript: str) -> ScoreResult:
    canned = _canned_results()
    if transcript in canned:
        return canned[transcript]
    return _heuristic_score(transcript)


def _heuristic_score(transcript: str) -> ScoreResult:
    """Deterministic, taxonomy-grounded keyword scorer for arbitrary pasted text.

    Not a trained model -- a safety net so the no-API-key demo never crashes or shows an
    ungrounded result for text outside the bundled demo set. Real classification is
    `llm`/`xlmr`.
    """
    taxonomy = get_taxonomy()
    lowered = transcript.lower()

    tags: list[TacticTag] = []
    spans: list[Span] = []
    matched_hard_signal = False

    for tactic in taxonomy.tactics:
        examples = (*tactic.examples_ru, *tactic.examples_kk)
        match = next((example for example in examples if example.lower() in lowered), None)
        if match is None:
            continue
        start = lowered.find(match.lower())
        actual_text = transcript[start : start + len(match)]
        spans.append(Span(text=actual_text, start=start, end=start + len(actual_text)))
        tags.append(TacticTag(id=tactic.id, weight=1.0 if tactic.hard_signal else 0.6))
        matched_hard_signal = matched_hard_signal or tactic.hard_signal

    if matched_hard_signal:
        risk = _MOCK_RISK_HARD_SIGNAL
    elif len(tags) >= 2:
        risk = _MOCK_RISK_MULTI_TACTIC
    elif len(tags) == 1:
        risk = _MOCK_RISK_SINGLE_TACTIC
    else:
        risk = _MOCK_RISK_NO_SIGNAL

    return ScoreResult(
        risk=risk,
        tags=tags,
        attributions=spans,
        backend="mock",
        raw_confidence=None,
    )
