"""Explicit scientific settings shared by execution and analysis."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GreedyDecoding:
    max_new_tokens: int
    seed: int = 42

    def generation_options(self):
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        return {
            "do_sample": False,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
        }


@dataclass(frozen=True)
class SampledDecoding:
    max_new_tokens: int
    temperature: float
    seed: int
    top_p: float = 1.0
    top_k: int = 0

    def generation_options(self):
        if self.max_new_tokens < 1 or self.temperature <= 0:
            raise ValueError(
                "Sampled decoding needs positive token cap and temperature"
            )
        if not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("Invalid top_p or top_k")
        return {
            "do_sample": True,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "use_cache": True,
        }


@dataclass
class Direction:
    values: Any
    provenance: dict


@dataclass
class Condition:
    name: str
    method: str = "unchanged"
    direction: Direction | None = None
    layers: tuple[int, ...] = ()
    application_locations: tuple[str, ...] = ()
    strength: float = 0.0
    token_positions: str = "all"


@dataclass
class Comparison:
    counts: Any
    paired_changes: Any
    uncertainty: Any
    examples: Any
    completeness: Any = field(default_factory=dict)
