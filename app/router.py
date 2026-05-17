"""
Core routing engine — Cluster-based selection.

Effective cost formula
──────────────────────
  effective_cost = compute_cost + latency_weight * (latency_ms / 1000)

  compute_cost  = tokens * cost_per_token_midpoint (from cluster spec)
  latency_cost  = latency_weight_$/s * (latency_ms / 1000)

Costs and latency come from cluster specifications:
  • compute_cost  — based on cluster's cost_per_token_midpoint (USD/token)
  • latency_ms    — based on cluster's latency_ms_midpoint (p50)

Routing selects the cluster with the lowest effective cost.
"""
from __future__ import annotations

import math
from typing import List

from adapters.aws import AWSAdapter
from adapters.azure import AzureAdapter
from adapters.base import BaseAdapter
from adapters.gcp import GCPAdapter
from app.latency import latency_tracker
from app.schema import CompletionResponse, ProviderCandidate
from config.clusters import CLUSTERS
from config.config import PROVIDERS


# Singleton adapter instances
_ADAPTERS: List[BaseAdapter] = [AWSAdapter(), GCPAdapter(), AzureAdapter()]


def estimate_tokens(text: str) -> int:
    """Simple word-count heuristic (~1.35 tokens per word)."""
    return max(1, math.ceil(len(text.split()) * 1.35))


def build_candidates(
    prompt: str,
    max_tokens: int,
    latency_weight: float,
) -> List[ProviderCandidate]:
    """
    Evaluate all 10 clusters based on effective cost.
    Returns a list sorted by ascending effective_cost, with selected=True
    on the first (cheapest) entry.
    """
    est_input = estimate_tokens(prompt)
    est_output = min(max_tokens, 200)  # conservative: most responses fit within 200 tokens
    total_tokens = est_input + est_output

    candidates: List[ProviderCandidate] = []

    for cluster_id, cluster_cfg in CLUSTERS.items():
        # ── Cluster-based pricing (midpoint of range) ────────────────────────
        cost_per_token = cluster_cfg.cost_per_token_midpoint
        compute_cost = total_tokens * cost_per_token

        # ── Cluster-based latency (midpoint of range) ────────────────────────
        latency_ms = cluster_cfg.latency_midpoint

        latency_cost = latency_weight * (latency_ms / 1000)
        effective_cost = compute_cost + latency_cost

        # Pick a representative model from cluster support
        model = cluster_cfg.model_support[0] if cluster_cfg.model_support else "unknown"

        gpu_util_range = f"{cluster_cfg.gpu_util_min}–{cluster_cfg.gpu_util_max}%"
        batch_fill_range = f"{cluster_cfg.batch_fill_min}–{cluster_cfg.batch_fill_max}%"

        candidates.append(
            ProviderCandidate(
                cluster_id=cluster_id,
                cluster_provider=cluster_cfg.provider,
                cluster_gpu=cluster_cfg.gpu_type,
                model=model,
                est_input_tokens=est_input,
                est_output_tokens=est_output,
                compute_cost=compute_cost,
                latency_ms=latency_ms,
                latency_cost=latency_cost,
                effective_cost=effective_cost,
                is_mock=False,  # All clusters are treated as available
                selected=False,
                gpu_util_range=gpu_util_range,
                batch_fill_range=batch_fill_range,
                price_source="cluster",
                price_note=cluster_cfg.notes,
                latency_source="baseline",
                latency_samples=0,
            )
        )

    candidates.sort(key=lambda c: c.effective_cost)
    candidates[0].selected = True
    return candidates



async def route_and_complete(
    prompt: str,
    max_tokens: int = 256,
    latency_weight: float | None = None,
) -> CompletionResponse:
    """Select the lowest-cost cluster and run the completion."""
    from config.config import router_config  # noqa: PLC0415
    weight = latency_weight if latency_weight is not None else router_config.latency_weight

    candidates = build_candidates(prompt, max_tokens, weight)
    best = candidates[0]

    # Get cluster config for selected cluster
    cluster_cfg = CLUSTERS[best.cluster_id]

    # Use an adapter based on cluster provider (simple mapping)
    adapter = _get_adapter_for_provider(cluster_cfg.provider)
    result = await adapter.complete(prompt, max_tokens, best.model)

    # Record observed latency for future optimization
    latency_tracker.record(f"cluster_{best.cluster_id}", result.latency_ms)

    # Recompute final cost using cluster's cost_per_token × actual token counts
    est_input = estimate_tokens(prompt)
    est_output = result.output_tokens
    actual_compute_cost = (est_input + est_output) * cluster_cfg.cost_per_token_midpoint
    actual_latency_cost = weight * (result.latency_ms / 1000)
    actual_effective_cost = actual_compute_cost + actual_latency_cost

    return CompletionResponse(
        text=result.text,
        cluster_id=best.cluster_id,
        cluster_provider=cluster_cfg.provider,
        cluster_gpu=cluster_cfg.gpu_type,
        model=best.model,
        actual_latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        compute_cost=actual_compute_cost,
        latency_cost=actual_latency_cost,
        effective_cost=actual_effective_cost,
        latency_weight=weight,
        is_mock=result.is_mock,
        candidates=candidates,
    )


def _get_adapter_for_provider(provider_name: str) -> BaseAdapter:
    """Map cluster provider name to an adapter."""
    provider_map = {
        "AWS": "aws",
        "GCP": "gcp",
        "Azure": "azure",
    }
    adapter_name = provider_map.get(provider_name, "aws")
    for a in _ADAPTERS:
        if a.name == adapter_name:
            return a
    # Fallback to AWS
    return _ADAPTERS[0]

