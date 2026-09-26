"""A stored `consent_version` must prove what the citizen agreed to (legal review M3).

The page (`site/i18n.js`) exports `CONSENT_VERSION` and holds the wording under the
`report.consent*` keys in every language. The server registers each version with the SHA-256
of that wording (`reports.model.CITIZEN_CONSENT_VERSIONS`). Editing the wording without a new
version, or shipping a version the server does not know, fails here.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from qorgan.reports.model import CITIZEN_CONSENT_VERSIONS, CONSENT_VERSION_PATTERN

_I18N = Path(__file__).resolve().parents[2] / "site" / "i18n.js"
_LOCALE_BLOCK = re.compile(r"^const (\w+) = \{$", re.MULTILINE)
_CONSENT_LINE = re.compile(r'^\s*"(report\.consent[\w.]*)":\s*("(?:[^"\\]|\\.)*"),?\s*$', re.MULTILINE)
_EXPORTED_VERSION = re.compile(r'^export const CONSENT_VERSION = "([^"]+)";', re.MULTILINE)


def consent_texts(source: str) -> dict[str, dict[str, str]]:
    """{locale: {key: text}} for every `report.consent*` key, by the locale block it sits in."""
    blocks = [(m.start(), m.group(1)) for m in _LOCALE_BLOCK.finditer(source)]
    texts: dict[str, dict[str, str]] = {}
    for match in _CONSENT_LINE.finditer(source):
        locale = [name for start, name in blocks if start < match.start()][-1]
        texts.setdefault(locale, {})[match.group(1)] = json.loads(match.group(2))
    return texts


def consent_digest(texts: dict[str, dict[str, str]]) -> str:
    canonical = json.dumps(texts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_the_page_wording_matches_the_registered_digest_of_its_version():
    source = _I18N.read_text(encoding="utf-8")
    version = _EXPORTED_VERSION.search(source).group(1)
    texts = consent_texts(source)
    assert set(texts) == {"ru", "kk", "en"} and all(texts[loc].get("report.consent") for loc in texts)
    assert version in CITIZEN_CONSENT_VERSIONS, f"page sends unregistered consent version {version!r}"
    assert consent_digest(texts) == CITIZEN_CONSENT_VERSIONS[version], (
        "the consent wording changed: register a new version (never edit an existing entry)"
    )


def test_registered_versions_are_well_formed_and_unique_digests():
    assert all(re.fullmatch(CONSENT_VERSION_PATTERN, v) for v in CITIZEN_CONSENT_VERSIONS)
    digests = list(CITIZEN_CONSENT_VERSIONS.values())
    assert all(re.fullmatch(r"[0-9a-f]{64}", d) for d in digests) and len(set(digests)) == len(digests)
