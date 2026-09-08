"""A public URL in front of a metered API is a way to lose money quietly.

Two controls, tested separately because they answer different questions: a
shared token decides who may ask, and the caps decide how much may be spent.
The caps are what actually protects the budget, because a token a browser has
to send is a token that will eventually leak.
"""

from __future__ import annotations

import pytest

from app.ratelimit import RunGuard, client_key

NOW = 1_000_000.0


class FakeRequest:
    def __init__(self, headers=None, host="203.0.113.9"):
        self.headers = headers or {}
        self.client = type("C", (), {"host": host})()


# --- who may ask ---------------------------------------------------------


def test_no_token_configured_lets_everyone_through() -> None:
    """Local development and the test suite must be unaffected."""
    guard = RunGuard(token="")
    assert guard.check_token(None) is None
    assert guard.check_token("anything") is None


def test_a_configured_token_must_match() -> None:
    guard = RunGuard(token="s3cret")
    assert guard.check_token("s3cret") is None
    assert guard.check_token("wrong!") is not None
    assert guard.check_token("") is not None
    assert guard.check_token(None) is not None


def test_a_token_of_the_wrong_length_is_refused() -> None:
    """The length check is the one comparison that cannot be constant-time."""
    guard = RunGuard(token="s3cret")
    assert guard.check_token("s3cre") is not None
    assert guard.check_token("s3crett") is not None


# --- how much may be spent ----------------------------------------------


def test_the_daily_cap_stops_spending() -> None:
    guard = RunGuard(daily_total=3, per_ip_hourly=0, token="")
    for i in range(3):
        assert guard.check_quota("a", now=NOW) is None
        guard.record("a", now=NOW)

    denied = guard.check_quota("a", now=NOW)
    assert denied is not None
    assert "3 searches a day" in denied

    # A different address does not get its own daily budget.
    assert guard.check_quota("b", now=NOW) is not None


def test_the_daily_cap_is_a_rolling_window() -> None:
    guard = RunGuard(daily_total=1, per_ip_hourly=0, token="")
    guard.record("a", now=NOW)
    assert guard.check_quota("a", now=NOW + 3600) is not None
    assert guard.check_quota("a", now=NOW + 86401) is None


def test_the_per_address_limit_is_per_address() -> None:
    guard = RunGuard(daily_total=0, per_ip_hourly=2, token="")
    for _ in range(2):
        guard.record("a", now=NOW)

    assert guard.check_quota("a", now=NOW) is not None
    assert guard.check_quota("b", now=NOW) is None, "one caller must not block another"
    assert guard.check_quota("a", now=NOW + 3601) is None


def test_a_refused_run_is_not_counted() -> None:
    """check_quota must not consume budget; only record() does."""
    guard = RunGuard(daily_total=2, per_ip_hourly=0, token="")
    for _ in range(10):
        assert guard.check_quota("a", now=NOW) is None
    assert guard.snapshot(now=NOW)["daily_used"] == 0


def test_zero_means_unlimited() -> None:
    guard = RunGuard(daily_total=0, per_ip_hourly=0, token="")
    for _ in range(50):
        guard.record("a", now=NOW)
    assert guard.check_quota("a", now=NOW) is None


def test_the_snapshot_reports_what_is_left() -> None:
    guard = RunGuard(daily_total=5, per_ip_hourly=2, token="abc")
    guard.record("a", now=NOW)
    snap = guard.snapshot(now=NOW)
    assert snap == {
        "token_required": True,
        "daily_limit": 5,
        "daily_used": 1,
        "per_ip_hourly_limit": 2,
    }


# --- who a limit is counted against -------------------------------------


def test_the_forwarded_client_is_preferred_behind_a_proxy() -> None:
    """Render terminates TLS at its proxy, so the socket peer is always Render."""
    request = FakeRequest({"x-forwarded-for": "198.51.100.7, 10.0.0.1"})
    assert client_key(request) == "198.51.100.7"


def test_the_socket_peer_is_used_when_there_is_no_proxy() -> None:
    assert client_key(FakeRequest(host="203.0.113.9")) == "203.0.113.9"


def test_an_unidentifiable_caller_still_gets_a_key() -> None:
    """They share one bucket rather than escaping the limit."""
    request = FakeRequest(host="")
    assert client_key(request) == "unknown"


# --- through the API ----------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from app import api as api_module

    guard = RunGuard(daily_total=2, per_ip_hourly=0, token="letmein")
    monkeypatch.setattr(api_module, "GUARD", guard)
    return TestClient(api_module.app), guard


def test_the_endpoint_refuses_a_missing_token(client) -> None:
    http, _ = client
    response = http.post("/runs", json={"name": "Ada Lovelace"})
    assert response.status_code == 401
    assert "token" in response.json()["detail"]


def test_the_endpoint_accepts_the_token_and_counts_the_run(client) -> None:
    http, guard = client
    response = http.post(
        "/runs", json={"name": "Ada Lovelace"}, headers={"X-Run-Token": "letmein"}
    )
    assert response.status_code == 202
    assert guard.snapshot()["daily_used"] == 1


def test_the_endpoint_refuses_once_the_budget_is_gone(client) -> None:
    http, _ = client
    headers = {"X-Run-Token": "letmein"}
    for _ in range(2):
        assert http.post("/runs", json={"name": "Ada"}, headers=headers).status_code == 202

    response = http.post("/runs", json={"name": "Ada"}, headers=headers)
    assert response.status_code == 429
    assert "a day" in response.json()["detail"]


def test_an_unsearchable_request_does_not_spend_budget(client) -> None:
    """A malformed payload must not be able to exhaust the day's runs."""
    http, guard = client
    response = http.post("/runs", json={}, headers={"X-Run-Token": "letmein"})
    assert response.status_code == 400
    assert guard.snapshot()["daily_used"] == 0


# --- cross-origin, once the frontend is on its own host ------------------


def build_app(monkeypatch, origins: str | None):
    """A fresh app with ALLOWED_ORIGINS set, since middleware binds at import."""
    import importlib

    from app import api as api_module

    if origins is None:
        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("ALLOWED_ORIGINS", origins)
    return importlib.reload(api_module)


def test_no_cors_headers_when_no_origins_are_configured(monkeypatch) -> None:
    """Development is same-origin through the Vite proxy; nothing to allow."""
    from fastapi.testclient import TestClient

    module = build_app(monkeypatch, None)
    response = TestClient(module.app).get(
        "/health", headers={"Origin": "https://anything.example"}
    )
    assert "access-control-allow-origin" not in response.headers


def test_a_configured_origin_is_allowed(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    module = build_app(monkeypatch, "https://webutation-web.onrender.com")
    response = TestClient(module.app).get(
        "/health", headers={"Origin": "https://webutation-web.onrender.com"}
    )
    assert (
        response.headers["access-control-allow-origin"]
        == "https://webutation-web.onrender.com"
    )


def test_an_unlisted_origin_is_not_allowed(monkeypatch) -> None:
    """A wildcard would let any page spend this deployment's search budget."""
    from fastapi.testclient import TestClient

    module = build_app(monkeypatch, "https://webutation-web.onrender.com")
    response = TestClient(module.app).get(
        "/health", headers={"Origin": "https://attacker.example"}
    )
    assert "access-control-allow-origin" not in response.headers


def test_the_preflight_permits_the_run_token_header(monkeypatch) -> None:
    """Without this the browser refuses the header and searches fail as CORS."""
    from fastapi.testclient import TestClient

    module = build_app(monkeypatch, "https://webutation-web.onrender.com")
    response = TestClient(module.app).options(
        "/runs",
        headers={
            "Origin": "https://webutation-web.onrender.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-run-token",
        },
    )
    assert response.status_code == 200
    allowed = response.headers["access-control-allow-headers"].lower()
    assert "x-run-token" in allowed
    assert "content-type" in allowed
