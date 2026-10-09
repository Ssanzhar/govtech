"""The Android build path (ADR D67): the TWA config and the Digital Asset Links template agree,
the template is served where Chrome looks for it, and no signing key is committed."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from qorgan.api import app

_ROOT = Path(__file__).resolve().parents[2]
_TWA = _ROOT / "android" / "twa" / "twa-manifest.json"
_ASSETLINKS = _ROOT / "site" / ".well-known" / "assetlinks.json"


def test_twa_manifest_matches_the_web_manifest_and_asset_links():
    twa = json.loads(_TWA.read_text(encoding="utf-8"))
    web = json.loads((_ROOT / "site" / "manifest.webmanifest").read_text(encoding="utf-8"))
    links = json.loads(_ASSETLINKS.read_text(encoding="utf-8"))
    assert twa["startUrl"] == web["start_url"] == "/live.html"
    assert twa["themeColor"] == web["theme_color"] and twa["backgroundColor"] == web["background_color"]
    assert twa["fallbackType"] == "customtabs"  # never a WebView fallback: it would lose isolation
    assert [entry["target"]["package_name"] for entry in links] == [twa["packageId"]]
    assert links[0]["relation"] == ["delegate_permission/common.handle_all_urls"]


def test_asset_links_template_is_served_as_json():
    response = TestClient(app).get("/.well-known/assetlinks.json")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()[0]["target"]["namespace"] == "android_app"


def test_no_signing_key_is_tracked():
    tracked = subprocess.run(["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert not [p for p in tracked if p.endswith((".keystore", ".jks", ".p12"))]
    twa = json.loads(_TWA.read_text(encoding="utf-8"))
    assert twa["signingKey"]["path"].endswith(".keystore")  # the name .gitignore keeps out of git
    assert "*.keystore" in (_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
