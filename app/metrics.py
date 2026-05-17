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
                "cluster_breakdown": {},
            }

        total = len(history)
        avg_cost = sum(r["effective_cost"] for r in history) / total
        avg_latency = sum(r["actual_latency_ms"] for r in history) / total

        total_saved = 0.0
        for r in history:
            if r.get("candidates"):
                max_cost = max(c["effective_cost"] for c in r["candidates"])
                total_saved += max_cost - r["effective_cost"]

        cluster_counts: Dict[str, int] = {}
        cluster_costs: Dict[str, float] = {}
        cluster_info: Dict[str, Dict[str, str]] = {}
        for r in history:
            cid = r["cluster_id"]
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
            cluster_costs[cid] = cluster_costs.get(cid, 0.0) + r["effective_cost"]
            if cid not in cluster_info:
                cluster_info[cid] = {
                    "provider": r["cluster_provider"],
                    "gpu": r["cluster_gpu"],
                }

        cluster_breakdown = {
            c: {
                "provider": cluster_info[c]["provider"],
                "gpu": cluster_info[c]["gpu"],
                "count": cluster_counts[c],
                "total_cost": cluster_costs[c],
                "avg_cost": cluster_costs[c] / cluster_counts[c],
                "share_pct": round(cluster_counts[c] / total * 100, 1),
            }
            for c in cluster_counts
        }

        return {
            "total_requests": total,
            "avg_effective_cost": avg_cost,
            "avg_latency_ms": avg_latency,
            "total_cost_saved": total_saved,
            "cluster_breakdown": cluster_breakdown,
        }

    def get_history(self, n: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            history = list(self._history)
        return list(reversed(history[-n:]))


metrics_store = MetricsStore()
