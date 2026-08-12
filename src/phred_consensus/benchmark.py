"""Deterministic synthetic comparison with unweighted majority voting."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .core import BASES, call_consensus, majority_consensus


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Mismatch counts for a seeded synthetic experiment."""

    seed: int
    bases: int
    reads: int
    bayesian_mismatches: int
    majority_mismatches: int

    @property
    def bayesian_rate(self) -> float:
        return self.bayesian_mismatches / self.bases

    @property
    def majority_rate(self) -> float:
        return self.majority_mismatches / self.bases


def _observe(rng: random.Random, base: str, quality: int) -> str:
    error_probability = 10.0 ** (-quality / 10.0)
    if rng.random() >= error_probability:
        return base
    alternatives = BASES.replace(base, "")
    return alternatives[rng.randrange(3)]


def run_benchmark(*, seed: int = 2026, bases: int = 2000) -> BenchmarkResult:
    """Compare callers on five reads with heterogeneous Q3/Q30 qualities."""

    if bases <= 0:
        raise ValueError("bases must be positive")
    rng = random.Random(seed)
    truth = "".join(rng.choice(BASES) for _ in range(bases))
    quality_levels = (3, 3, 3, 30, 30)
    reads = [
        "".join(_observe(rng, base, quality) for base in truth)
        for quality in quality_levels
    ]
    qualities = [tuple([quality] * bases) for quality in quality_levels]
    bayesian = call_consensus(reads, qualities).sequence
    majority = majority_consensus(reads)
    return BenchmarkResult(
        seed=seed,
        bases=bases,
        reads=len(reads),
        bayesian_mismatches=sum(a != b for a, b in zip(bayesian, truth, strict=True)),
        majority_mismatches=sum(a != b for a, b in zip(majority, truth, strict=True)),
    )
