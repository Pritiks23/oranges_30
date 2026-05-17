"""Pydantic schemas for request / response payloads."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt to route and complete.")
    max_tokens: int = Field(256, ge=1, le=4096, description="Maximum output tokens.")
    latency_weight: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "$/second penalty for latency. Overrides server default when supplied. "
            "Higher values favour faster providers over cheaper ones."
        ),
    )


class ProviderCandidate(BaseModel):
    """Cost breakdown for one cluster evaluated during routing."""
    model_config = ConfigDict(protected_namespaces=())
    cluster_id: str            # e.g. "D", "E", "F"
    cluster_provider: str      # Cloud provider or vendor
    cluster_gpu: str           # GPU type (H100, A100, TPU v5, etc.)
    model: str                 # Model size (e.g. "70B")
    est_input_tokens: int
    est_output_tokens: int
    compute_cost: float        # USD
    latency_ms: float          # latency used for routing (midpoint or observed)
    latency_cost: float        # USD  = weight * latency_s
    effective_cost: float      # compute_cost + latency_cost
    is_mock: bool
    selected: bool = False
    # Cluster characteristics
    gpu_util_range: str = ""   # e.g. "85-95%"
    batch_fill_range: str = "" # e.g. "75-90%"
    # Pricing provenance
    price_source: str = "cluster"    # "cluster" (from cluster spec)
    price_note: str = ""
    latency_source: str = "baseline" # "baseline" (cluster midpoint)
    latency_samples: int = 0


class CompletionResponse(BaseModel):
    text: str
    cluster_id: str              # e.g. "D", "E", "F"
    cluster_provider: str        # e.g. "AWS", "GCP", "CoreWeave"
    cluster_gpu: str             # e.g. "H100", "A100"
    model: str                   # Model size (e.g. "70B")
    actual_latency_ms: float
    input_tokens: int
    output_tokens: int
    compute_cost: float
    latency_cost: float
    effective_cost: float
    latency_weight: float
    is_mock: bool
    candidates: List[ProviderCandidate]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ClusterStatus(BaseModel):
    cluster_id: str
    provider: str                # Cloud provider or vendor
    gpu_type: str
    model_support: List[str]     # e.g. ["70B"] or ["13B", "70B"]
    is_available: bool
    effective_cost_range: str    # e.g. "$0.000045–$0.000080"
    typical_latency_ms: int


class MetricsSummary(BaseModel):
    total_requests: int
    avg_effective_cost: float
    avg_latency_ms: float
    total_cost_saved: float
    cluster_breakdown: Dict[str, Any]


class HistoryEntry(BaseModel):
    id: int
    timestamp: str
    prompt_snippet: str
    cluster_id: str
    cluster_provider: str
    cluster_gpu: str
    model: str
    effective_cost: float
    actual_latency_ms: float
    is_mock: bool


# ── Pricing endpoint schemas ──────────────────────────────────────────────────

class PricingModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    provider: str
    provider_display: str
    model_key: str
    model_id: str
    input_per_1m: float
    output_per_1m: float
    source: str          # "live" | "cached" | "hardcoded"
    source_display: str  # e.g. "live (42s ago)"
    provider_note: str
    fetched_at: Optional[str]


class PricingResponse(BaseModel):
    last_refresh: Optional[str]
    models: List[PricingModelInfo]

