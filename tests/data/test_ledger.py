"""TDD tests for `qorgan.data.ledger` -- the inspection ledger for the authored held-out set.

Why (PLAN_2026-09 A2): five `authored_heldout` negatives were read while engineering the
reassurance / KK features, so their scores are not an unbiased generalization signal. The
ledger records exactly which ids were inspected, when, and why; the eval harness uses it to
report the clean and inspected subsets separately.
"""

from datetime import date
from pathlib import Path

import pytest

from qorgan.data.anchors import build_anchor_dialogues
from qorgan.data.ledger import InspectionLedger, LedgerError, load_inspection_ledger

_VALID = """\
version: 1
entries:
  - id: real_neg_bank_fraud_alert_ru
    inspected_on: 2026-07-13
    reason: false positive at 0.990 on the embedding-only model
  - id: real_neg_bank_card_delivery_kk
    inspected_on: 2026-07-15
    reason: sentinel for the KK-boundary retrain gate
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "ledger.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_entries_and_exposes_ids(tmp_path):
    ledger = load_inspection_ledger(_write(tmp_path, _VALID))
    assert isinstance(ledger, InspectionLedger)
    assert ledger.ids == frozenset({"real_neg_bank_fraud_alert_ru", "real_neg_bank_card_delivery_kk"})
    assert ledger.entries[0].inspected_on == date(2026, 7, 13)
    assert "sentinel" in ledger.entries[1].reason


def test_ledger_is_immutable(tmp_path):
    ledger = load_inspection_ledger(_write(tmp_path, _VALID))
    with pytest.raises(Exception):
        ledger.entries = ()  # type: ignore[misc]


def test_missing_file_raises(tmp_path):
    with pytest.raises(LedgerError):
        load_inspection_ledger(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "text",
    [
        "version: 1\nentries:\n  - id: x\n    reason: r\n",  # missing date
        "version: 1\nentries:\n  - id: x\n    inspected_on: 2026-07-13\n    reason: ''\n",  # blank reason
        "version: 1\nentries:\n  - id: x\n    inspected_on: 2026-07-13\n    reason: r\n  - id: x\n    inspected_on: 2026-07-14\n    reason: r\n",  # duplicate id
        "version: 1\nentries: []\n",  # empty ledger is a mistake, not a state
        "just a string\n",
    ],
)
def test_invalid_ledgers_raise(tmp_path, text):
    with pytest.raises(LedgerError):
        load_inspection_ledger(_write(tmp_path, text))


def test_repo_ledger_ids_exist_in_an_evaluation_split():
    """Every ledger id must name a real evaluated record -- an authored anchor or a row of an
    evaluation split -- so the `(inspected)` subsets can never be built from phantom ids. The
    ledger is not anchors-only: `eval.run` applies it to any split, and `shift` rows have been
    read too (ADR D43)."""
    import json
    from pathlib import Path as _Path

    ledger = load_inspection_ledger()
    known = {d.id for d in build_anchor_dialogues()}
    processed = _Path(__file__).resolve().parents[2] / "data" / "processed"
    for split in ("val", "test", "authored_heldout", "ood", "adversarial", "adversarial_legit", "shift"):
        path = processed / f"{split}.jsonl"
        if path.exists():
            known |= {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert ledger.ids <= known, ledger.ids - known
    assert ledger.ids >= {
        # the five authored negatives read during the July feature engineering
        "real_neg_bank_fraud_alert_ru",
        "real_neg_telecom_tariff_ru",
        "real_neg_bank_card_ready_ru",
        "real_neg_telecom_tariff_notice_mixed",
        "real_neg_bank_card_delivery_kk",
        # the two shift negatives read while measuring the cross-generator gap (D35, D42)
        "shift_legit_mixed_05",
        "shift_legit_kk_11",
    }
