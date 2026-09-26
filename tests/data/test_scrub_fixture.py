"""The committed JS parity fixture for `scrub_text` must match the Python implementation
(regenerate with `python scripts/export_scrub_fixtures.py`)."""

import json
from pathlib import Path

from qorgan.data.scrub import scrub_text

_FIXTURE = Path(__file__).resolve().parents[2] / "tests_js" / "fixtures" / "scrub.json"


def test_scrub_fixture_matches_python():
    cases = json.loads(_FIXTURE.read_text(encoding="utf-8"))["cases"]
    assert cases, "fixture is empty"
    for case in cases:
        assert scrub_text(case["input"]) == case["expected"], case["input"]
