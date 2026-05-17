"""
Cluster configurations: 10 cost-optimized inference clusters.

Each cluster includes:
  - name: cluster identifier (D–M)
  - provider: cloud provider or vendor
  - model_support: list of supported model sizes
  - gpu_type: GPU/accelerator type
  - gpu_util_min/max: typical GPU utilization range (%)
  - batch_fill_min/max: typical batch fill range (%)
  - cost_per_token_min/max: USD per token (compute cost)
  - latency_ms_min/max: p50 latency range
  - effective_cost_min/max: total cost including latency impact (USD/token equiv)
  - notes: operational characteristics

The effective_cost_midpoint is used for routing decisions (lower = better).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ClusterConfig:
    cluster_id: str                    # D, E, F, etc.
    provider: str                      # AWS, GCP, Azure, CoreWeave, Lambda, etc.
    model_support: List[str]           # e.g. ["70B"], ["13B", "70B"]
    gpu_type: str                      # H100, A100, TPU v5, etc.
    gpu_util_min: float                # %
    gpu_util_max: float                # %
    batch_fill_min: float              # %
    batch_fill_max: float              # %
    cost_per_token_min: float          # USD
    cost_per_token_max: float          # USD
    latency_ms_min: int                # milliseconds
    latency_ms_max: int                # milliseconds
    effective_cost_min: float          # USD
    effective_cost_max: float          # USD
    notes: str                         # e.g. "On-demand, stable", "Spot, unstable"

    @property
    def effective_cost_midpoint(self) -> float:
        """Midpoint of effective cost range — used for routing."""
        return (self.effective_cost_min + self.effective_cost_max) / 2

    @property
    def latency_midpoint(self) -> int:
        """Midpoint of latency range — conservative baseline for routing."""
        return (self.latency_ms_min + self.latency_ms_max) // 2

    @property
    def cost_per_token_midpoint(self) -> float:
        """Midpoint of cost/token range."""
        return (self.cost_per_token_min + self.cost_per_token_max) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Cluster catalogue
# ─────────────────────────────────────────────────────────────────────────────

CLUSTERS: Dict[str, ClusterConfig] = {
    "D": ClusterConfig(
        cluster_id="D",
        provider="AWS",
        model_support=["70B"],
        gpu_type="H100",
        gpu_util_min=75,
        gpu_util_max=85,
        batch_fill_min=65,
        batch_fill_max=80,
        cost_per_token_min=0.000055,
        cost_per_token_max=0.000095,
        latency_ms_min=35,
        latency_ms_max=55,
        effective_cost_min=0.000060,
        effective_cost_max=0.000110,
        notes="AWS On-Demand H100, stable compute, reserved capacity",
    ),
    "E": ClusterConfig(
        cluster_id="E",
        provider="AWS",
        model_support=["13B", "70B"],
        gpu_type="A100",
        gpu_util_min=80,
        gpu_util_max=95,
        batch_fill_min=70,
        batch_fill_max=90,
        cost_per_token_min=0.000035,
        cost_per_token_max=0.000070,
        latency_ms_min=45,
        latency_ms_max=90,
        effective_cost_min=0.000040,
        effective_cost_max=0.000090,
        notes="AWS Spot A100, low cost but unstable, variable latency",
    ),
    "F": ClusterConfig(
        cluster_id="F",
        provider="GCP",
        model_support=["70B"],
        gpu_type="H100",
        gpu_util_min=85,
        gpu_util_max=95,
        batch_fill_min=75,
        batch_fill_max=90,
        cost_per_token_min=0.000030,
        cost_per_token_max=0.000065,
        latency_ms_min=40,
        latency_ms_max=70,
        effective_cost_min=0.000035,
        effective_cost_max=0.000075,
        notes="GCP Spot H100, best case cost, high utilization",
    ),
    "G": ClusterConfig(
        cluster_id="G",
        provider="Azure",
        model_support=["13B", "70B"],
        gpu_type="A100",
        gpu_util_min=65,
        gpu_util_max=80,
        batch_fill_min=60,
        batch_fill_max=75,
        cost_per_token_min=0.000055,
        cost_per_token_max=0.000110,
        latency_ms_min=30,
        latency_ms_max=60,
        effective_cost_min=0.000060,
        effective_cost_max=0.000120,
        notes="Azure Reserved A100, predictable cost, moderate latency",
    ),
    "H": ClusterConfig(
        cluster_id="H",
        provider="CoreWeave",
        model_support=["70B"],
        gpu_type="H100",
        gpu_util_min=90,
        gpu_util_max=97,
        batch_fill_min=80,
        batch_fill_max=95,
        cost_per_token_min=0.000040,
        cost_per_token_max=0.000075,
        latency_ms_min=30,
        latency_ms_max=50,
        effective_cost_min=0.000045,
        effective_cost_max=0.000080,
        notes="CoreWeave H100, excellent efficiency, low latency",
    ),
    "I": ClusterConfig(
        cluster_id="I",
        provider="Lambda",
        model_support=["13B", "70B"],
        gpu_type="H100",
        gpu_util_min=80,
        gpu_util_max=90,
        batch_fill_min=70,
        batch_fill_max=85,
        cost_per_token_min=0.000050,
        cost_per_token_max=0.000085,
        latency_ms_min=25,
        latency_ms_max=45,
        effective_cost_min=0.000055,
        effective_cost_max=0.000095,
        notes="Lambda H100, lowest latency option, good cost",
    ),
    "J": ClusterConfig(
        cluster_id="J",
        provider="Crusoe",
        model_support=["70B"],
        gpu_type="H100",
        gpu_util_min=85,
        gpu_util_max=92,
        batch_fill_min=75,
        batch_fill_max=88,
        cost_per_token_min=0.000045,
        cost_per_token_max=0.000080,
        latency_ms_min=35,
        latency_ms_max=60,
        effective_cost_min=0.000050,
        effective_cost_max=0.000090,
        notes="Crusoe H100, balanced cost and latency",
    ),
    "K": ClusterConfig(
        cluster_id="K",
        provider="Private",
        model_support=["13B", "70B"],
        gpu_type="A100",
        gpu_util_min=50,
        gpu_util_max=75,
        batch_fill_min=40,
        batch_fill_max=70,
        cost_per_token_min=0.000080,
        cost_per_token_max=0.000160,
        latency_ms_min=50,
        latency_ms_max=120,
        effective_cost_min=0.000090,
        effective_cost_max=0.000180,
        notes="Private Bare Metal A100, highest cost & latency, low utilization",
    ),
    "L": ClusterConfig(
        cluster_id="L",
        provider="GCP",
        model_support=["70B"],
        gpu_type="TPU v5",
        gpu_util_min=85,
        gpu_util_max=98,
        batch_fill_min=80,
        batch_fill_max=95,
        cost_per_token_min=0.000030,
        cost_per_token_max=0.000060,
        latency_ms_min=45,
        latency_ms_max=80,
        effective_cost_min=0.000035,
        effective_cost_max=0.000070,
        notes="GCP TPU v5, competitive cost, high utilization",
    ),
    "M": ClusterConfig(
        cluster_id="M",
        provider="AWS",
        model_support=["70B"],
        gpu_type="H100",
        gpu_util_min=70,
        gpu_util_max=85,
        batch_fill_min=60,
        batch_fill_max=80,
        cost_per_token_min=0.000060,
        cost_per_token_max=0.000110,
        latency_ms_min=55,
        latency_ms_max=120,
        effective_cost_min=0.000070,
        effective_cost_max=0.000130,
        notes="Multi-region failover H100, highest tail latency trade-off",
    ),
}


def get_best_cluster_for_model(model_size: str) -> str:
    """
    Return the cluster ID with the lowest effective cost for a given model size.
    If model_size is not specified, return the overall best cluster.
    """
    candidates = []
    for cluster_id, cluster in CLUSTERS.items():
        # If model size is specified, filter by support
        if model_size and model_size not in cluster.model_support:
            continue
        candidates.append((cluster.effective_cost_midpoint, cluster_id))

    if not candidates:
        # Fallback to best overall cluster
        candidates = [(c.effective_cost_midpoint, cid) for cid, c in CLUSTERS.items()]

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def rank_clusters_by_effective_cost() -> List[tuple[str, float]]:
    """Return all clusters ranked by effective cost (lowest first)."""
    ranked = [(cid, c.effective_cost_midpoint) for cid, c in CLUSTERS.items()]
    ranked.sort(key=lambda x: x[1])
    return ranked
