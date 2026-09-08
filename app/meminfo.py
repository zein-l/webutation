"""How much memory this process is using, how much it is allowed, and
whether the directory it writes to survives a restart.

Reported by ``/health`` because the deployed service was killed mid-search
twice, and "the run vanished" is a symptom that could equally be a crash, a
deploy, or an eviction. A peak figure next to a limit distinguishes them
without access to the host's dashboard.

Linux only, and deliberately so: this reads the kernel's own accounting rather
than adding a dependency, and every deployment target here is a Linux
container. On anything else the fields come back None rather than guessed at.
"""

from __future__ import annotations

from pathlib import Path

#: Where the kernel reports this process's memory. ``VmHWM`` is the high-water
#: mark — the largest resident size since the process started — which is the
#: only figure that survives a spike and is therefore the one worth reporting.
_STATUS = Path("/proc/self/status")

#: cgroup v2, then v1. A container's limit is not /proc/meminfo, which reports
#: the host's memory and would make a 512MB instance look like it had 64GB.
_CGROUP_LIMITS = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)


def _kb_field(text: str, field: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(f"{field}:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) / 1024
    return None


def memory_limit_mb() -> float | None:
    """The container's memory ceiling, or None when there is not one."""
    for path in _CGROUP_LIMITS:
        try:
            raw = path.read_text().strip()
        except OSError:
            continue
        if raw == "max" or not raw.isdigit():
            continue
        value = int(raw) / 1024 / 1024
        # cgroup v1 reports an enormous sentinel when unlimited.
        if value > 1024 * 1024:
            continue
        return round(value, 1)
    return None


def memory_report() -> dict:
    """Current and peak resident size, with the limit if the kernel states one.

    ``peak_rss_mb`` is what a reader wants after a run that died: it records the
    worst moment, not the calm afterwards.
    """
    try:
        text = _STATUS.read_text()
    except OSError:
        return {"rss_mb": None, "peak_rss_mb": None, "limit_mb": memory_limit_mb()}

    current = _kb_field(text, "VmRSS")
    peak = _kb_field(text, "VmHWM")
    limit = memory_limit_mb()
    report = {
        "rss_mb": round(current, 1) if current is not None else None,
        "peak_rss_mb": round(peak, 1) if peak is not None else None,
        "limit_mb": limit,
    }
    if peak is not None and limit:
        report["peak_pct_of_limit"] = round(100 * peak / limit, 1)
    return report


#: Where the runtime writes: responses, images, uploads and the signing key.
#: Relative, resolved against the working directory, matching app/cache.py.
_CACHE_DIR = Path("cache")

_MOUNTS = Path("/proc/self/mounts")


def storage_report() -> dict:
    """Whether the cache directory is a mount of its own, and what is in it.

    A cache on the container's own filesystem is thrown away by every deploy;
    a cache on an attached disk is not. Those two look identical from outside
    until the moment something is lost, which is the wrong time to find out.

    ``persistent`` is read from the kernel's mount table rather than assumed
    from configuration, because the question is whether the disk is mounted
    here and now, not whether a file somewhere asks for one.
    """
    resolved = _CACHE_DIR.resolve()
    report: dict = {
        "path": str(resolved),
        "persistent": None,
        "cached_responses": None,
    }

    # Top level only. Render polls this endpoint continuously and a warm cache
    # holds hundreds of files; walking it recursively on every health check
    # would turn a liveness probe into disk work. The count answers "is
    # anything here", which is the question, and cached responses live at the
    # top level anyway.
    try:
        report["cached_responses"] = sum(
            1 for entry in resolved.iterdir() if entry.suffix == ".json"
        )
    except OSError:
        pass

    try:
        mounts = _MOUNTS.read_text().splitlines()
    except OSError:
        # Not Linux. Unknown is the honest answer, not False.
        return report

    target = str(resolved)
    report["persistent"] = any(
        len(parts) > 1 and parts[1] == target
        for parts in (line.split() for line in mounts)
    )
    return report
