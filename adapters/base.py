"""Abstract base class that every cloud adapter must implement."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    provider: str    # internal key e.g. "aws"
    model: str       # model key e.g. "claude-3-haiku"
    is_mock: bool


class BaseAdapter(ABC):
    """Thin wrapper around a cloud provider's inference endpoint."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Internal provider key, must match key in config.PROVIDERS."""
        ...

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True when the required env-vars / credentials are present."""
        ...

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        max_tokens: int,
        model: str,
    ) -> CompletionResult:
        """Run inference and return a CompletionResult."""
        ...
