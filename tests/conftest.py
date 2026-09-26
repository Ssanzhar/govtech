"""Suite-wide test isolation for secrets that gate the Level-2 console and the audit log.

`load_config()` reads `.env` into the process environment without overriding variables that
are already set, so a developer's local analyst keys / audit-chain key would otherwise leak
into tests. Every test starts from the known test values below (override with monkeypatch,
e.g. to "" to test the fail-closed paths) and from empty in-process rate-limiter windows.
"""

from __future__ import annotations

import pytest

from support.analysts import ANALYST_KEYS_ENV, AUDIT_CHAIN_KEY


@pytest.fixture(autouse=True)
def _console_secrets_and_fresh_limiters(monkeypatch):
    monkeypatch.setenv("QORGAN_ANALYST_KEYS", ANALYST_KEYS_ENV)
    monkeypatch.setenv("QORGAN_AUDIT_CHAIN_KEY", AUDIT_CHAIN_KEY)
    from qorgan.api_admin_auth import reset_limiters

    reset_limiters()
    yield
    reset_limiters()
