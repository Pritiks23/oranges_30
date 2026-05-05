"""
AWS Bedrock adapter — supports Anthropic Claude models via the Converse API.

──────────────────────────────────────────────────────────────────────────────
HOW TO CONNECT AWS BEDROCK (minimal steps)
──────────────────────────────────────────────────────────────────────────────
1. Open https://console.aws.amazon.com and sign in.
2. Navigate to Bedrock → Model Access → request access for "Claude 3 Haiku".
   (Takes ~1 minute to approve for most regions.)
3. Create an IAM user or role with the managed policy:
     AmazonBedrockFullAccess  (or narrow to bedrock:InvokeModel on the ARN)
4. Generate an access key for the IAM user:
   IAM → Users → <user> → Security credentials → Create access key.
5. Set the following environment variables (add to .env):
     AWS_ACCESS_KEY_ID=AKIA...
     AWS_SECRET_ACCESS_KEY=...
     AWS_REGION=us-east-1        # must match region where you enabled the model
6. Set MOCK_MODE=false in .env.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time

from adapters.base import BaseAdapter, CompletionResult
from config.config import PROVIDERS


def _mock_response(prompt: str, model: str) -> str:
    snippet = (prompt[:60] + "...") if len(prompt) > 60 else prompt
    return (
        f"[AWS Bedrock / {model} - MOCK]\n\n"
        f"You asked: \"{snippet}\"\n\n"
        "This is a simulated response. The real Bedrock model would process "
        "your prompt using Claude's advanced reasoning. "
        "Set AWS credentials in .env and MOCK_MODE=false to call the live API."
    )


class AWSAdapter(BaseAdapter):
    @property
    def name(self) -> str:
        return "aws"

    @property
    def is_configured(self) -> bool:
        return bool(
            os.getenv("AWS_ACCESS_KEY_ID")
            and os.getenv("AWS_SECRET_ACCESS_KEY")
            and os.getenv("AWS_REGION")
        )

    async def complete(
        self, prompt: str, max_tokens: int, model: str
    ) -> CompletionResult:
        cfg = PROVIDERS["aws"]
        model_cfg = cfg.models[model]

        if not self.is_configured:
            # ── Mock path ─────────────────────────────────────────────────
            latency_ms = random.uniform(700, 900)
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
        import boto3  # noqa: PLC0415 — optional heavy dep

        client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.invoke_model(
                modelId=model_cfg.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            ),
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        data = json.loads(response["body"].read())
        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        return CompletionResult(
            text=text,
            input_tokens=usage.get("input_tokens", len(prompt.split())),
            output_tokens=usage.get("output_tokens", len(text.split())),
            latency_ms=latency_ms,
            provider=self.name,
            model=model,
            is_mock=False,
        )
