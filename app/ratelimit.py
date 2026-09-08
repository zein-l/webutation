"""Who may start a run, and how many runs may be started.

A public URL in front of a metered search API is a way to lose money quietly.
Two separate controls, because they answer different questions:

* a shared token decides *who* may ask, and stops a crawler that found the
  endpoint from using it at all;
* the caps decide *how much* may be spent, and hold even when the token has
  leaked — which, for a token that a browser has to send, it eventually will.

The caps are the part that protects the budget. The token only raises the bar.

State is in memory, so it resets when the service restarts and is not shared
between instances. That is honest for a single Starter instance and is stated
rather than hidden: a deployment that scales out needs this in Postgres or
Redis instead.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class RunGuard:
    """Admission control for the one endpoint that spends money."""

    #: Runs allowed from one client per hour. Enough to try each combination of
    #: inputs on one subject and get it wrong twice.
    per_ip_hourly: int = field(default_factory=lambda: _int_env("MAX_RUNS_PER_IP_HOURLY", 10))
    #: Runs allowed across every client per day. The real budget stop.
    daily_total: int = field(default_factory=lambda: _int_env("MAX_RUNS_PER_DAY", 100))
    #: Shared secret required in ``X-Run-Token``. Empty disables the check, so
    #: local development and the test suite are unaffected.
    token: str = field(default_factory=lambda: (os.environ.get("RUN_ACCESS_TOKEN") or "").strip())

    _by_ip: dict[str, deque[float]] = field(default_factory=dict, repr=False)
    _today: deque[float] = field(default_factory=deque, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    HOUR = 3600.0
    DAY = 86400.0

    def check_token(self, supplied: str | None) -> str | None:
        """None if the caller may proceed, else why not.

        Compared without short-circuiting on the first differing byte. The
        token is low-value and the endpoint is slow, so a timing attack is not
        a realistic route in — but constant-time comparison is one line and
        removes the question.
        """
        if not self.token:
            return None
        given = (supplied or "").strip()
        if len(given) != len(self.token):
            return "missing or invalid run token"
        mismatch = 0
        for a, b in zip(given, self.token):
            mismatch |= ord(a) ^ ord(b)
        return None if mismatch == 0 else "missing or invalid run token"

    def check_quota(self, client: str, now: float | None = None) -> str | None:
        """None if a run may start, else why not. Does not record the run."""
        now = time.time() if now is None else now
        with self._lock:
            self._expire(now)
            if self.daily_total > 0 and len(self._today) >= self.daily_total:
                return (
                    f"this deployment allows {self.daily_total} searches a day and has "
                    "used them. Each one costs a live API call, so the cap is a budget, "
                    "not a throttle. It resets 24 hours after the earliest run counted."
                )
            seen = self._by_ip.get(client)
            if self.per_ip_hourly > 0 and seen is not None and len(seen) >= self.per_ip_hourly:
                return (
                    f"{self.per_ip_hourly} searches an hour from one address is the "
                    "limit. Cached subjects are free to re-run from the report page."
                )
        return None

    def record(self, client: str, now: float | None = None) -> None:
        """Count a run that is actually starting."""
        now = time.time() if now is None else now
        with self._lock:
            self._expire(now)
            self._today.append(now)
            self._by_ip.setdefault(client, deque()).append(now)

    def _expire(self, now: float) -> None:
        while self._today and now - self._today[0] >= self.DAY:
            self._today.popleft()
        for client, seen in list(self._by_ip.items()):
            while seen and now - seen[0] >= self.HOUR:
                seen.popleft()
            if not seen:
                del self._by_ip[client]

    def snapshot(self, now: float | None = None) -> dict:
        """What is left, for the health endpoint to report."""
        now = time.time() if now is None else now
        with self._lock:
            self._expire(now)
            return {
                "token_required": bool(self.token),
                "daily_limit": self.daily_total,
                "daily_used": len(self._today),
                "per_ip_hourly_limit": self.per_ip_hourly,
            }


#: One guard for the process. Tests build their own.
GUARD = RunGuard()


def client_key(request) -> str:
    """The address a limit is counted against.

    Render terminates TLS at its proxy, so the socket peer is always the proxy
    and ``X-Forwarded-For`` carries the real client. The left-most entry is the
    one the client sent and can be spoofed; that is acceptable here, because
    the per-address limit is a courtesy and the daily total is the control that
    cannot be evaded by forging a header.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return getattr(getattr(request, "client", None), "host", "") or "unknown"
