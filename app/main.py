"""FastAPI application — API routes + frontend serving."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from adapters.aws import AWSAdapter
from adapters.azure import AzureAdapter
from adapters.gcp import GCPAdapter
from app.latency import latency_tracker
from app.metrics import metrics_store
from app.router import route_and_complete
from app.schema import (
    CompletionRequest,
    CompletionResponse,
    ClusterStatus,
    HistoryEntry,
    MetricsSummary,
    PricingModelInfo,
    PricingResponse,
)
from config.clusters import CLUSTERS
from config.config import PROVIDERS
from config.pricing import pricing_service

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Inference Router",
    description="Cost-optimal AI inference routing across AWS, GCP, and Azure.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Adapter registry ─────────────────────────────────────────────────────────
_ADAPTERS = {a.name: a for a in [AWSAdapter(), GCPAdapter(), AzureAdapter()]}

# ── Static frontend ───────────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Startup / background pricing refresh ─────────────────────────────────────

_PRICING_REFRESH_INTERVAL = int(os.getenv("PRICING_REFRESH_INTERVAL", "1800"))  # seconds


async def _pricing_refresh_loop() -> None:
    """Background task: refresh pricing every PRICING_REFRESH_INTERVAL seconds."""
    while True:
        await asyncio.sleep(_PRICING_REFRESH_INTERVAL)
        try:
            await pricing_service.refresh_all()
            logger.info("Pricing cache refreshed.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Background pricing refresh failed: %s", exc)


@app.on_event("startup")
async def startup_event() -> None:
    # Fetch live prices immediately at startup (non-blocking — network failures are logged)
    try:
        await pricing_service.refresh_all()
        logger.info("Initial pricing fetch complete.")
    except asyncio.CancelledError:
        raise  # don't swallow cancellation
    except Exception as exc:
        logger.warning("Initial pricing fetch failed: %s", exc)
    # Schedule periodic refresh
    asyncio.create_task(_pricing_refresh_loop())


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index))


# ── API ───────────────────────────────────────────────────────────────────────

@app.post("/api/complete", response_model=CompletionResponse, tags=["inference"])
async def complete(req: CompletionRequest) -> CompletionResponse:
    """Route the prompt to the lowest-cost cluster and return the completion."""
    response = await route_and_complete(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        latency_weight=req.latency_weight,
    )
    metrics_store.record(
        {
            "id": 0,  # overwritten by record()
            "timestamp": response.timestamp,
            "prompt_snippet": req.prompt[:80],
            "cluster_id": response.cluster_id,
            "cluster_provider": response.cluster_provider,
            "cluster_gpu": response.cluster_gpu,
            "model": response.model,
            "effective_cost": response.effective_cost,
            "actual_latency_ms": response.actual_latency_ms,
            "is_mock": response.is_mock,
            "candidates": [c.model_dump() for c in response.candidates],
        }
    )
    return response


@app.get("/api/clusters", response_model=List[ClusterStatus], tags=["system"])
async def list_clusters() -> List[ClusterStatus]:
    """Return status and configuration for each of the 10 inference clusters."""
    result = []
    for cluster_id, cfg in CLUSTERS.items():
        effective_cost_str = f"${cfg.effective_cost_min:.9f}–${cfg.effective_cost_max:.9f}"
        result.append(
            ClusterStatus(
                cluster_id=cluster_id,
                provider=cfg.provider,
                gpu_type=cfg.gpu_type,
                model_support=cfg.model_support,
                is_available=True,
                effective_cost_range=effective_cost_str,
                typical_latency_ms=cfg.latency_midpoint,
            )
        )
    return result


@app.get("/api/pricing", response_model=PricingResponse, tags=["system"])
async def get_pricing() -> PricingResponse:
    """
    Current token prices used for routing decisions.

    Each entry shows:
    - The price in USD per 1 million tokens (input and output)
    - Source: "live" (fetched from provider API this session),
              "cached" (fetched earlier, still within TTL),
              "hardcoded" (no live API available or fetch failed)
    - When the price was last fetched
    - Which API/source the price came from
    """
    all_entries = pricing_service.all_entries()
    models: List[PricingModelInfo] = []

    for provider_key, provider_cfg in PROVIDERS.items():
        for model_key in provider_cfg.models:
            entry_key = f"{provider_key}/{model_key}"
            entry = all_entries.get(entry_key)
            if not entry:
                continue
            model_cfg = provider_cfg.models[model_key]
            models.append(
                PricingModelInfo(
                    provider=provider_key,
                    provider_display=provider_cfg.display_name,
                    model_key=model_key,
                    model_id=model_cfg.model_id,
                    input_per_1m=entry.input_per_1m,
                    output_per_1m=entry.output_per_1m,
                    source=entry.source,
                    source_display=entry.display_source(),
                    provider_note=entry.provider_note,
                    fetched_at=entry.fetched_at.isoformat() if entry.fetched_at else None,
                )
            )

    last_refresh = pricing_service.last_refresh
    return PricingResponse(
        last_refresh=last_refresh.isoformat() if last_refresh else None,
        models=models,
    )


@app.post("/api/pricing/refresh", tags=["system"])
async def force_pricing_refresh() -> dict:
    """Manually trigger a pricing refresh from all provider APIs."""
    await pricing_service.refresh_all()
    return {"status": "ok", "refreshed_at": pricing_service.last_refresh.isoformat()}


@app.get("/api/metrics", response_model=MetricsSummary, tags=["system"])
async def get_metrics() -> MetricsSummary:
    """Aggregate metrics across all completed requests."""
    return MetricsSummary(**metrics_store.get_summary())


@app.get("/api/history", response_model=List[HistoryEntry], tags=["system"])
async def get_history(n: int = 20) -> List[HistoryEntry]:
    """Last N completed requests (newest first)."""
    rows = metrics_store.get_history(n)
    return [
        HistoryEntry(
            id=r["id"],
            timestamp=r["timestamp"],
            prompt_snippet=r["prompt_snippet"],
            cluster_id=r["cluster_id"],
            cluster_provider=r["cluster_provider"],
            cluster_gpu=r["cluster_gpu"],
            model=r["model"],
            effective_cost=r["effective_cost"],
            actual_latency_ms=r["actual_latency_ms"],
            is_mock=r["is_mock"],
        )
        for r in rows
    ]


# ── Dev server entrypoint ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

