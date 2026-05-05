"""
Azure OpenAI adapter - uses the official OpenAI Python SDK with an Azure endpoint.

──────────────────────────────────────────────────────────────────────────────
HOW TO CONNECT AZURE OPENAI (minimal steps)
──────────────────────────────────────────────────────────────────────────────
1. Open https://portal.azure.com → Create a resource → Azure OpenAI Service.
2. After provisioning (~2 min), go to the resource → Keys and Endpoint.
   Copy KEY 1 and the Endpoint URL.
3. Go to Azure OpenAI Studio (oai.azure.com) → Deployments → Create.
   Deploy a model e.g. "gpt-35-turbo", name the deployment "gpt-35-turbo".
4. Set the following environment variables (add to .env):
     AZURE_OPENAI_API_KEY=<KEY 1 from step 2>
     AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
     AZURE_OPENAI_DEPLOYMENT=gpt-35-turbo   # deployment name from step 3
     AZURE_OPENAI_API_VERSION=2024-02-01    # stable API version
5. Set MOCK_MODE=false in .env.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import os
import random
import time

from adapters.base import BaseAdapter, CompletionResult


def _mock_response(prompt: str, model: str) -> str:
    snippet = (prompt[:60] + "...") if len(prompt) > 60 else prompt
    return (
        f"[Azure OpenAI / {model} - MOCK]\n\n"
        f'You asked: "{snippet}"\n\n'
        "This is a simulated response. The real Azure OpenAI GPT model would "
        "generate a high-quality, enterprise-grade reply to your prompt. "
        "Set Azure credentials in .env and MOCK_MODE=false to call the live API."
    )


class AzureAdapter(BaseAdapter):
    @property
    def name(self) -> str:
        return "azure"

    @property
    def is_configured(self) -> bool:
        return bool(
            os.getenv("AZURE_OPENAI_API_KEY")
            and os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    async def complete(
        self, prompt: str, max_tokens: int, model: str
    ) -> CompletionResult:
        if not self.is_configured:
            # ── Mock path ─────────────────────────────────────────────────
            latency_ms = random.uniform(900, 1_100)
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
        from openai import AsyncAzureOpenAI  # noqa: PLC0415

        client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", model)

        t0 = time.perf_counter()
        response = await client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        text = response.choices[0].message.content or ""
        usage = response.usage
        return CompletionResult(
            text=text,
            input_tokens=usage.prompt_tokens if usage else len(prompt.split()),
            output_tokens=usage.completion_tokens if usage else len(text.split()),
            latency_ms=latency_ms,
            provider=self.name,
            model=model,
            is_mock=False,
        )
