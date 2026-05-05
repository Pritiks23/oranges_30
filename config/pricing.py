"""
Real-time pricing service.

Pricing sources
───────────────
• Azure   — Azure Retail Prices REST API (public, no auth required)
            https://prices.azure.com/api/retail/prices
            Refreshed every CACHE_TTL_SECONDS (default 30 min).

• AWS     — AWS Pricing API via boto3 (requires AWS credentials).
            Falls back to hardcoded rates if credentials are absent
            or the API call fails.

• GCP     — No public machine-readable pricing API exists for Vertex AI.
            Hardcoded rates are used and labeled as such.

All prices are USD per 1 million tokens.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# ── Fallback hardcoded prices (USD / 1M tokens) ──────────────────────────────
# These are used when live fetching is unavailable.  Source: public pricing
# pages as of mid-2024.  Updated in code whenever rates change materially.
_FALLBACK: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("aws",   "claude-3-haiku"):   (0.25,   1.25),
    ("aws",   "claude-3-sonnet"):  (3.00,  15.00),
    ("gcp",   "gemini-1.5-flash"): (0.075,  0.30),
    ("gcp",   "gemini-1.5-pro"):   (3.50,  10.50),
    ("azure", "gpt-35-turbo"):     (0.50,   1.50),
    ("azure", "gpt-4"):            (30.00, 60.00),
}

# TTL for cached live prices
_CACHE_TTL_SECONDS = 1_800  # 30 minutes


@dataclass
class PricingEntry:
    input_per_1m: float
    output_per_1m: float
    source: str           # "live" | "cached" | "hardcoded"
    fetched_at: Optional[datetime] = field(default=None)
    provider_note: str = ""  # e.g. "Azure Retail Prices API"

    def is_stale(self) -> bool:
        if self.source == "hardcoded" or self.fetched_at is None:
            return True
        age = (datetime.now(timezone.utc) - self.fetched_at).total_seconds()
        return age > _CACHE_TTL_SECONDS

    def age_seconds(self) -> Optional[float]:
        if self.fetched_at is None:
            return None
        return (datetime.now(timezone.utc) - self.fetched_at).total_seconds()

    def display_source(self) -> str:
        """Human-readable source with staleness info."""
        if self.source == "live":
            age = self.age_seconds() or 0
            return f"live ({int(age)}s ago)"
        if self.source == "cached":
            age = self.age_seconds() or 0
            return f"cached ({int(age)}s ago)"
        return "hardcoded"


class PricingService:
    """
    Thread-safe async service that maintains a live-refreshed price cache.

    Usage
    ─────
        svc = PricingService()
        await svc.refresh_all()          # fetch from APIs once at startup
        entry = svc.get("azure", "gpt-35-turbo")
        print(entry.input_per_1m, entry.source)

    A background task in main.py calls refresh_all() every CACHE_TTL_SECONDS.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], PricingEntry] = {}
        self._lock = asyncio.Lock()
        self._last_refresh: Optional[datetime] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, provider: str, model: str) -> PricingEntry:
        """Return the best available price entry for a provider+model pair."""
        key = (provider, model)
        entry = self._cache.get(key)
        if entry and not entry.is_stale():
            # Return as "cached" (still fresh)
            return PricingEntry(
                input_per_1m=entry.input_per_1m,
                output_per_1m=entry.output_per_1m,
                source="cached",
                fetched_at=entry.fetched_at,
                provider_note=entry.provider_note,
            )
        # Fall back to hardcoded
        fb = _FALLBACK.get(key)
        if fb:
            return PricingEntry(
                input_per_1m=fb[0],
                output_per_1m=fb[1],
                source="hardcoded",
                provider_note="fallback — see README for setup",
            )
        return PricingEntry(
            input_per_1m=0.0,
            output_per_1m=0.0,
            source="hardcoded",
            provider_note="unknown model",
        )

    def all_entries(self) -> Dict[str, PricingEntry]:
        """Return all cached entries keyed as 'provider/model'."""
        result: Dict[str, PricingEntry] = {}
        seen: set[Tuple[str, str]] = set()
        # First add live/cached entries
        for (provider, model), entry in self._cache.items():
            result[f"{provider}/{model}"] = entry
            seen.add((provider, model))
        # Fill in hardcoded fallbacks for anything not yet fetched
        for (provider, model), (inp, out) in _FALLBACK.items():
            if (provider, model) not in seen:
                result[f"{provider}/{model}"] = PricingEntry(
                    input_per_1m=inp,
                    output_per_1m=out,
                    source="hardcoded",
                    provider_note="fallback",
                )
        return result

    @property
    def last_refresh(self) -> Optional[datetime]:
        return self._last_refresh

    async def refresh_all(self) -> None:
        """Fetch live prices from all available APIs concurrently."""
        async with self._lock:
            results = await asyncio.gather(
                self._fetch_azure(),
                self._fetch_aws(),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("Pricing refresh error: %s", r)
            self._last_refresh = datetime.now(timezone.utc)

    # ── Azure Retail Prices API ───────────────────────────────────────────────
    # Public endpoint — no authentication required.
    # Docs: https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices

    async def _fetch_azure(self) -> None:
        url = "https://prices.azure.com/api/retail/prices"
        params = {
            "api-version": "2023-01-01-preview",
            "$filter": (
                "serviceName eq 'Azure OpenAI' "
                "and armRegionName eq 'eastus' "
                "and type eq 'Consumption'"
            ),
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                items = r.json().get("Items", [])
        except Exception as exc:
            logger.warning("Azure pricing fetch failed: %s", exc)
            return

        # Build {normalized_sku: {input: price, output: price}} in USD/1M tokens
        sku_prices: Dict[str, Dict[str, float]] = {}
        for item in items:
            sku: str = item.get("skuName", "").lower()
            meter: str = item.get("meterName", "").lower()
            uom: str = item.get("unitOfMeasure", "")
            retail: float = item.get("retailPrice", 0.0)

            # Convert to USD per 1M tokens
            if "1k" in uom.lower():
                price_per_1m = retail * 1_000
            elif "1m" in uom.lower():
                price_per_1m = retail
            else:
                continue

            sku_prices.setdefault(sku, {})
            if "input" in meter or "prompt" in meter:
                sku_prices[sku]["input"] = price_per_1m
            elif "output" in meter or "completion" in meter:
                sku_prices[sku]["output"] = price_per_1m
            else:
                # Undifferentiated pricing — apply to both
                sku_prices[sku].setdefault("input", price_per_1m)
                sku_prices[sku].setdefault("output", price_per_1m)

        now = datetime.now(timezone.utc)
        note = "Azure Retail Prices API (eastus)"

        # Map Azure SKU names → our model keys
        _azure_map = [
            ("gpt-3.5-turbo",  ("azure", "gpt-35-turbo")),
            ("gpt-35-turbo",   ("azure", "gpt-35-turbo")),
            ("gpt-4",          ("azure", "gpt-4")),
        ]
        matched: set[Tuple[str, str]] = set()
        for sku, prices in sku_prices.items():
            if "input" not in prices or "output" not in prices:
                continue
            for sku_substr, model_key in _azure_map:
                if sku_substr in sku and model_key not in matched:
                    self._cache[model_key] = PricingEntry(
                        input_per_1m=prices["input"],
                        output_per_1m=prices["output"],
                        source="live",
                        fetched_at=now,
                        provider_note=note,
                    )
                    matched.add(model_key)
                    logger.info(
                        "Azure price updated: %s  in=$%.4f/1M  out=$%.4f/1M",
                        model_key,
                        prices["input"],
                        prices["output"],
                    )

    # ── AWS Pricing API ───────────────────────────────────────────────────────
    # Requires AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in environment.
    # Uses the AWS Pricing service (always in us-east-1) to query Bedrock rates.

    async def _fetch_aws(self) -> None:
        if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
            logger.debug("AWS credentials absent — skipping live pricing fetch.")
            return

        try:
            import boto3  # noqa: PLC0415
        except ImportError:
            return

        region = os.getenv("AWS_REGION", "us-east-1")
        # AWS Pricing API is only available in us-east-1 and ap-south-1
        try:
            client = boto3.client("pricing", region_name="us-east-1")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.get_products(
                    ServiceCode="AmazonBedrock",
                    Filters=[
                        {
                            "Type": "TERM_MATCH",
                            "Field": "location",
                            "Value": "US East (N. Virginia)",
                        }
                    ],
                    MaxResults=100,
                ),
            )
        except Exception as exc:
            logger.warning("AWS pricing fetch failed: %s", exc)
            return

        now = datetime.now(timezone.utc)
        note = f"AWS Pricing API (region: {region})"

        # Map Bedrock model identifiers → our keys
        _aws_map = {
            "claude-3-haiku":   ("aws", "claude-3-haiku"),
            "claude-3-sonnet":  ("aws", "claude-3-sonnet"),
        }

        for pl_str in response.get("PriceList", []):
            try:
                pl = json.loads(pl_str)
            except (json.JSONDecodeError, TypeError):
                continue

            attrs = pl.get("product", {}).get("attributes", {})
            model_id: str = attrs.get("modelId", "").lower()
            token_type: str = attrs.get("tokenType", "").lower()  # "input" | "output"

            matched_key: Optional[Tuple[str, str]] = None
            for substr, key in _aws_map.items():
                if substr in model_id:
                    matched_key = key
                    break
            if not matched_key:
                continue

            # Extract USD price per unit from the nested pricing structure
            terms = pl.get("terms", {}).get("OnDemand", {})
            price_usd: Optional[float] = None
            uom_str = ""
            for term in terms.values():
                for dim in term.get("priceDimensions", {}).values():
                    usd = dim.get("pricePerUnit", {}).get("USD")
                    if usd is not None:
                        price_usd = float(usd)
                        uom_str = dim.get("unit", "")
                        break

            if price_usd is None:
                continue

            # Convert to USD/1M tokens
            if "1k" in uom_str.lower():
                price_per_1m = price_usd * 1_000
            elif "1m" in uom_str.lower():
                price_per_1m = price_usd
            else:
                # Assume per-token; multiply by 1M
                price_per_1m = price_usd * 1_000_000

            entry = self._cache.get(matched_key) or PricingEntry(
                input_per_1m=_FALLBACK.get(matched_key, (0.0, 0.0))[0],
                output_per_1m=_FALLBACK.get(matched_key, (0.0, 0.0))[1],
                source="live",
                fetched_at=now,
                provider_note=note,
            )
            if "input" in token_type or "prompt" in token_type:
                entry = PricingEntry(
                    input_per_1m=price_per_1m,
                    output_per_1m=entry.output_per_1m,
                    source="live",
                    fetched_at=now,
                    provider_note=note,
                )
            elif "output" in token_type or "completion" in token_type:
                entry = PricingEntry(
                    input_per_1m=entry.input_per_1m,
                    output_per_1m=price_per_1m,
                    source="live",
                    fetched_at=now,
                    provider_note=note,
                )
            self._cache[matched_key] = entry
            logger.info("AWS price updated: %s  token_type=%s  $%.4f/1M", matched_key, token_type, price_per_1m)


# ── Module-level singleton ────────────────────────────────────────────────────
pricing_service = PricingService()
