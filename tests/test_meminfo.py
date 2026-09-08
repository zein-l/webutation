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


# --- does the cache survive a deploy? -----------------------------------


def test_storage_reports_unknown_rather_than_false_off_linux() -> None:
    """No mount table means the question cannot be answered, not answered no.

    A cache on the container's own filesystem and a cache on an attached disk
    look identical from outside until something is lost, so guessing "not
    persistent" would be as misleading as guessing "persistent".
    """
    from app.meminfo import storage_report

    report = storage_report()
    assert set(report) >= {"path", "persistent", "cached_responses"}
    assert report["persistent"] in (None, True, False)


def test_storage_does_not_walk_the_whole_cache(tmp_path, monkeypatch) -> None:
    """Render polls /health continuously; this must stay a cheap call."""
    from app import meminfo

    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    nested = tmp_path / "images"
    nested.mkdir()
    for i in range(50):
        (nested / f"{i}.bin").write_bytes(b"x")

    monkeypatch.setattr(meminfo, "_CACHE_DIR", tmp_path)
    report = meminfo.storage_report()

    assert report["cached_responses"] == 2, "top-level responses only, not a walk"


def test_health_exposes_storage() -> None:
    from fastapi.testclient import TestClient

    from app import api as api_module

    body = TestClient(api_module.app).get("/health").json()
    assert "storage" in body
    assert set(body["storage"]) >= {"path", "persistent", "cached_responses"}
