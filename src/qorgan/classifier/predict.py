"""Unified classifier interface — the one contract every caller (app, eval) depends on.

`score(transcript) -> ScoreResult` routes to a backend selected via
`QORGAN_CLASSIFIER_BACKEND` (config-driven, default `llm`). Swapping the backend must
never touch callers.

Backends:
- `llm`   -- Gemini structured classifier (D1-5, `llm_classifier.py`). Ships first.
- `mock`  -- deterministic, no-network, no-API-key backend for demos/tests. Returns
             hand-authored canned results for the bundled demo transcripts and a
             taxonomy-keyword heuristic for arbitrary pasted text.
- `xlmr`  -- fine-tuned XLM-R backend. Not implemented yet (lands Day 3); raises
             `NotImplementedError` with a clear fallback message.
"""

from __future__ import annotations

from qorgan.classifier import llm_classifier
from qorgan.config import get_config
from qorgan.data.demo_transcripts import DEMO_TRANSCRIPTS
from qorgan.data.schema import ScoreResult, Span, TacticTag
from qorgan.taxonomy import get_taxonomy

_SUPPORTED_BACKENDS = ("llm", "xlmr", "mock")

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
        raise NotImplementedError(
            "The 'xlmr' backend is not implemented yet (lands Day 3 -- fine-tuned XLM-R). "
            "Set QORGAN_CLASSIFIER_BACKEND=llm or 'mock' in the meantime."
        )
    raise UnknownBackendError(
        f"Unknown classifier backend {active_backend!r}; expected one of {_SUPPORTED_BACKENDS}"
    )


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
