"""
Provider configurations: cost tables, latency baselines, routing parameters.
All costs are in USD. Prices are approximate 2024 on-demand rates.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    model_id: str  # provider-native identifier used in API calls
    input_cost_per_1m: float   # USD per 1 million input tokens
    output_cost_per_1m: float  # USD per 1 million output tokens
    typical_latency_ms: int    # p50 latency baseline in ms
    max_context_tokens: int


@dataclass
class ProviderConfig:
    name: str          # internal key  e.g. "aws"
    display_name: str  # e.g. "AWS Bedrock"
    models: Dict[str, ModelConfig]
    default_model: str  # key into models dict


# ---------------------------------------------------------------------------
# Provider catalogue
# ---------------------------------------------------------------------------
PROVIDERS: Dict[str, ProviderConfig] = {
    "aws": ProviderConfig(
        name="aws",
        display_name="AWS Bedrock",
        default_model="claude-3-haiku",
        models={
            "claude-3-haiku": ModelConfig(
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                input_cost_per_1m=0.25,
                output_cost_per_1m=1.25,
                typical_latency_ms=800,
                max_context_tokens=200_000,
            ),
            "claude-3-sonnet": ModelConfig(
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                input_cost_per_1m=3.0,
                output_cost_per_1m=15.0,
                typical_latency_ms=1_500,
                max_context_tokens=200_000,
            ),
        },
    ),
    "gcp": ProviderConfig(
        name="gcp",
        display_name="GCP Vertex AI",
        default_model="gemini-1.5-flash",
        models={
            "gemini-1.5-flash": ModelConfig(
                model_id="gemini-1.5-flash",
                input_cost_per_1m=0.075,
                output_cost_per_1m=0.30,
                typical_latency_ms=600,
                max_context_tokens=1_000_000,
            ),
            "gemini-1.5-pro": ModelConfig(
                model_id="gemini-1.5-pro",
                input_cost_per_1m=3.50,
                output_cost_per_1m=10.50,
                typical_latency_ms=1_200,
                max_context_tokens=1_000_000,
            ),
        },
    ),
    "azure": ProviderConfig(
        name="azure",
        display_name="Azure OpenAI",
        default_model="gpt-35-turbo",
        models={
            "gpt-35-turbo": ModelConfig(
                model_id="gpt-35-turbo",
                input_cost_per_1m=0.50,
                output_cost_per_1m=1.50,
                typical_latency_ms=950,
                max_context_tokens=16_385,
            ),
            "gpt-4": ModelConfig(
                model_id="gpt-4",
                input_cost_per_1m=30.0,
                output_cost_per_1m=60.0,
                typical_latency_ms=2_000,
                max_context_tokens=128_000,
            ),
        },
    ),
}


# ---------------------------------------------------------------------------
# Routing parameters
# ---------------------------------------------------------------------------
@dataclass
class RouterConfig:
    # Converts latency to a dollar penalty: cost = compute$ + weight * latency_s
    # Default: $0.001 per second → 1 s of extra latency ≈ $0.001
    latency_weight: float = field(
        default_factory=lambda: float(os.getenv("LATENCY_WEIGHT", "0.001"))
    )
    # When True every provider returns a realistic mock response (no cloud creds needed)
    mock_mode: bool = field(
        default_factory=lambda: os.getenv("MOCK_MODE", "true").lower() == "true"
    )


router_config = RouterConfig()
