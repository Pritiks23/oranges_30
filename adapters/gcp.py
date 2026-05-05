"""
GCP Vertex AI adapter — supports Gemini models via the Vertex AI SDK.

──────────────────────────────────────────────────────────────────────────────
HOW TO CONNECT GCP VERTEX AI (minimal steps)
──────────────────────────────────────────────────────────────────────────────
1. Open https://console.cloud.google.com, create a project (or use existing).
2. Enable the Vertex AI API:
   APIs & Services → Enable APIs → search "Vertex AI API" → Enable.
3. Create a service account:
   IAM & Admin → Service Accounts → Create → grant role "Vertex AI User".
4. Generate a JSON key:
   Service Account → Keys → Add Key → JSON → download file.
5. Set the following environment variables (add to .env):
     GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/key.json
     GCP_PROJECT_ID=my-project-id
     GCP_REGION=us-central1          # region where Vertex AI is enabled
6. Set MOCK_MODE=false in .env.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import os
import random
import time

from adapters.base import BaseAdapter, CompletionResult
from config.config import PROVIDERS


def _mock_response(prompt: str, model: str) -> str:
    snippet = (prompt[:60] + "...") if len(prompt) > 60 else prompt
    return (
        f"[GCP Vertex AI / {model} - MOCK]\n\n"
        f"You asked: \"{snippet}\"\n\n"
        "This is a simulated response. The real Vertex AI model (Gemini) would "
        "analyse your prompt and return a detailed, grounded answer. "
        "Set GCP credentials in .env and MOCK_MODE=false to call the live API."
    )


class GCPAdapter(BaseAdapter):
    @property
    def name(self) -> str:
        return "gcp"

    @property
    def is_configured(self) -> bool:
        return bool(
            os.getenv("GCP_PROJECT_ID")
            and (
                os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                or os.getenv("GOOGLE_CLOUD_PROJECT")
            )
        )

    async def complete(
        self, prompt: str, max_tokens: int, model: str
    ) -> CompletionResult:
        cfg = PROVIDERS["gcp"]
        model_cfg = cfg.models[model]

        if not self.is_configured:
            # ── Mock path ─────────────────────────────────────────────────
            latency_ms = random.uniform(500, 700)
            await asyncio.sleep(latency_ms / 1000)
            output_tokens = random.randint(90, 150)
            return CompletionResult(
                text=_mock_response(prompt, model),
                input_tokens=max(1, len(prompt.split())),
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                provider=self.name,
                model=model,
                is_mock=True,
            )

        # ── Live path ─────────────────────────────────────────────────────
        import vertexai  # noqa: PLC0415
        from vertexai.generative_models import GenerativeModel, GenerationConfig  # noqa: PLC0415

        project = os.getenv("GCP_PROJECT_ID")
        region = os.getenv("GCP_REGION", "us-central1")
        vertexai.init(project=project, location=region)

        gemini = GenerativeModel(model_cfg.model_id)
        config = GenerationConfig(max_output_tokens=max_tokens)

        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini.generate_content(prompt, generation_config=config),
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        text = response.text
        usage = response.usage_metadata
        return CompletionResult(
            text=text,
            input_tokens=getattr(usage, "prompt_token_count", len(prompt.split())),
            output_tokens=getattr(usage, "candidates_token_count", len(text.split())),
            latency_ms=latency_ms,
            provider=self.name,
            model=model,
            is_mock=False,
        )
