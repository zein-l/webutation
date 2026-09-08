"""Memory reporting, so a vanished run can be diagnosed without the dashboard.

The deployed service was killed mid-search twice. "The run vanished" looks
identical whether the process crashed, was redeployed, or was killed for
memory; a peak figure beside a limit is what separates them.
"""

from __future__ import annotations

from app.meminfo import _kb_field, memory_report

SAMPLE = """Name:\tpython
VmPeak:\t 1234567 kB
VmSize:\t 1200000 kB
VmRSS:\t  262144 kB
VmHWM:\t  524288 kB
Threads:\t8
"""


def test_fields_are_read_in_megabytes() -> None:
    assert _kb_field(SAMPLE, "VmRSS") == 256.0
    assert _kb_field(SAMPLE, "VmHWM") == 512.0


def test_a_missing_field_is_none_rather_than_zero() -> None:
    """Zero would read as "used no memory", which is a different claim."""
    assert _kb_field(SAMPLE, "VmNope") is None
    assert _kb_field("", "VmRSS") is None


def test_a_malformed_field_is_none() -> None:
    assert _kb_field("VmRSS:\tnot-a-number kB\n", "VmRSS") is None


def test_the_report_has_the_expected_shape() -> None:
    """On Windows the files do not exist, and None is the honest answer."""
    report = memory_report()
    assert set(report) >= {"rss_mb", "peak_rss_mb", "limit_mb"}
    for key in ("rss_mb", "peak_rss_mb", "limit_mb"):
        assert report[key] is None or isinstance(report[key], float)


def test_health_exposes_it(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from app import api as api_module

    body = TestClient(api_module.app).get("/health").json()
    assert "memory" in body
    assert set(body["memory"]) >= {"rss_mb", "peak_rss_mb", "limit_mb"}
