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
    """Cost breakdown for one provider evaluated during routing."""
    model_config = ConfigDict(protected_namespaces=())
    provider: str
    provider_display: str
    model: str
    model_id: str
    est_input_tokens: int
    est_output_tokens: int
    compute_cost: float     # USD
    latency_ms: int         # typical latency baseline
    latency_cost: float     # USD  = weight * latency_s
    effective_cost: float   # compute_cost + latency_cost
    is_mock: bool
    selected: bool = False


class CompletionResponse(BaseModel):
    text: str
    provider: str
    provider_display: str
    model: str
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


class ProviderStatus(BaseModel):
    name: str
    display_name: str
    is_configured: bool
    default_model: str
    models: List[str]
    typical_latency_ms: int


class MetricsSummary(BaseModel):
    total_requests: int
    avg_effective_cost: float
    avg_latency_ms: float
    total_cost_saved: float
    provider_breakdown: Dict[str, Any]


class HistoryEntry(BaseModel):
    id: int
    timestamp: str
    prompt_snippet: str
    provider: str
    provider_display: str
    model: str
    effective_cost: float
    actual_latency_ms: float
    is_mock: bool
