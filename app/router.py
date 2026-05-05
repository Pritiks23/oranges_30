"""
Core routing engine.

Effective cost formula
──────────────────────
  effective_cost = compute_cost + latency_weight * (latency_ms / 1000)

  compute_cost  = (input_tokens  * input_cost_per_1M
                +  output_tokens * output_cost_per_1M) / 1_000_000
  latency_cost  = latency_weight_$/s * (typical_latency_ms / 1000)

The provider with the lowest effective_cost is selected.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from adapters.aws import AWSAdapter
from adapters.azure import AzureAdapter
from adapters.base import BaseAdapter
from adapters.gcp import GCPAdapter
from app.schema import CompletionResponse, ProviderCandidate
from config.config import PROVIDERS, router_config


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
    """Evaluate all providers and return ranked ProviderCandidate list."""
    est_input = estimate_tokens(prompt)
    est_output = min(max_tokens, 200)  # conservative: most responses fit within 200 tokens

    candidates: List[ProviderCandidate] = []

    for adapter in _ADAPTERS:
        cfg = PROVIDERS[adapter.name]
        model_key = cfg.default_model
        model_cfg = cfg.models[model_key]

        compute_cost = (
            est_input * model_cfg.input_cost_per_1m
            + est_output * model_cfg.output_cost_per_1m
        ) / 1_000_000

        latency_cost = latency_weight * (model_cfg.typical_latency_ms / 1000)
        effective_cost = compute_cost + latency_cost

        candidates.append(
            ProviderCandidate(
                provider=adapter.name,
                provider_display=cfg.display_name,
                model=model_key,
                model_id=model_cfg.model_id,
                est_input_tokens=est_input,
                est_output_tokens=est_output,
                compute_cost=compute_cost,
                latency_ms=model_cfg.typical_latency_ms,
                latency_cost=latency_cost,
                effective_cost=effective_cost,
                is_mock=not adapter.is_configured,
                selected=False,
            )
        )

    candidates.sort(key=lambda c: c.effective_cost)
    candidates[0].selected = True
    return candidates


def _get_adapter(name: str) -> BaseAdapter:
    for a in _ADAPTERS:
        if a.name == name:
            return a
    raise ValueError(f"No adapter for provider '{name}'")


async def route_and_complete(
    prompt: str,
    max_tokens: int = 256,
    latency_weight: float | None = None,
) -> CompletionResponse:
    """Select the cheapest provider and run the completion."""
    weight = latency_weight if latency_weight is not None else router_config.latency_weight

    candidates = build_candidates(prompt, max_tokens, weight)
    best = candidates[0]

    adapter = _get_adapter(best.provider)
    result = await adapter.complete(prompt, max_tokens, best.model)

    cfg = PROVIDERS[best.provider]
    model_cfg = cfg.models[best.model]

    actual_compute_cost = (
        result.input_tokens * model_cfg.input_cost_per_1m
        + result.output_tokens * model_cfg.output_cost_per_1m
    ) / 1_000_000
    actual_latency_cost = weight * (result.latency_ms / 1000)
    actual_effective_cost = actual_compute_cost + actual_latency_cost

    return CompletionResponse(
        text=result.text,
        provider=result.provider,
        provider_display=cfg.display_name,
        model=result.model,
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
