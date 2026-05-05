"""
Per-provider rolling latency tracker using an exponential weighted average (EWA).

After every real request the observed latency is fed in.  Routing decisions use
the EWA instead of a hardcoded static baseline, so the router adapts to actual
provider performance over time.

The EWA formula:
    ewa = alpha * new_sample + (1 - alpha) * ewa_prev

alpha=0.2 means each new sample contributes 20% weight; the effective window is
roughly the last 1/alpha = 5 observations.  This keeps the estimate stable while
still reacting quickly to sustained latency changes.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

# EWA smoothing factor α = 0.2.
# Each new sample contributes 20% weight; the effective window ≈ 1/α = 5 observations.
# This balances stability (avoids overreacting to a single slow request) with
# responsiveness (adapts within ~5 requests if a provider's latency shifts).
# Decrease towards 0.05 for more stability; increase towards 0.5 to react faster.
_ALPHA = 0.2


class LatencyTracker:
    def __init__(self) -> None:
        self._ewa: Dict[str, float] = {}   # provider -> EWA latency in ms
        self._counts: Dict[str, int] = {}  # number of samples seen
        self._lock = threading.Lock()

    def record(self, provider: str, latency_ms: float) -> None:
        """Update the EWA for a provider after observing a real latency."""
        with self._lock:
            prev = self._ewa.get(provider)
            if prev is None:
                # First sample: seed the EWA with the observed value
                self._ewa[provider] = latency_ms
            else:
                self._ewa[provider] = _ALPHA * latency_ms + (1 - _ALPHA) * prev
            self._counts[provider] = self._counts.get(provider, 0) + 1

    def get(self, provider: str, fallback_ms: int) -> float:
        """Return observed EWA latency, or fallback if no data yet."""
        with self._lock:
            return self._ewa.get(provider, float(fallback_ms))

    def sample_count(self, provider: str) -> int:
        with self._lock:
            return self._counts.get(provider, 0)

    def snapshot(self) -> Dict[str, float]:
        """Return a copy of the current EWA state for all providers."""
        with self._lock:
            return dict(self._ewa)


# ── Module-level singleton ────────────────────────────────────────────────────
latency_tracker = LatencyTracker()
