"""Thread-safe in-memory metrics and request history store."""
from __future__ import annotations

import threading
from collections import deque
from typing import Any, Dict, List


class MetricsStore:
    def __init__(self, max_history: int = 500) -> None:
        self._history: deque[Dict[str, Any]] = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._counter = 0

    def record(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            self._counter += 1
            entry["id"] = self._counter
            self._history.append(entry)

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            history = list(self._history)

        if not history:
            return {
                "total_requests": 0,
                "avg_effective_cost": 0.0,
                "avg_latency_ms": 0.0,
                "total_cost_saved": 0.0,
                "provider_breakdown": {},
            }

        total = len(history)
        avg_cost = sum(r["effective_cost"] for r in history) / total
        avg_latency = sum(r["actual_latency_ms"] for r in history) / total

        total_saved = 0.0
        for r in history:
            if r.get("candidates"):
                max_cost = max(c["effective_cost"] for c in r["candidates"])
                total_saved += max_cost - r["effective_cost"]

        provider_counts: Dict[str, int] = {}
        provider_costs: Dict[str, float] = {}
        for r in history:
            p = r["provider"]
            provider_counts[p] = provider_counts.get(p, 0) + 1
            provider_costs[p] = provider_costs.get(p, 0.0) + r["effective_cost"]

        provider_breakdown = {
            p: {
                "display": r["provider_display"],
                "count": provider_counts[p],
                "total_cost": provider_costs[p],
                "avg_cost": provider_costs[p] / provider_counts[p],
                "share_pct": round(provider_counts[p] / total * 100, 1),
            }
            for p, r in {
                p: next(x for x in history if x["provider"] == p)
                for p in provider_counts
            }.items()
        }

        return {
            "total_requests": total,
            "avg_effective_cost": avg_cost,
            "avg_latency_ms": avg_latency,
            "total_cost_saved": total_saved,
            "provider_breakdown": provider_breakdown,
        }

    def get_history(self, n: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            history = list(self._history)
        return list(reversed(history[-n:]))


metrics_store = MetricsStore()
