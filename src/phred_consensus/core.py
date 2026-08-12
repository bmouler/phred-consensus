"""Numerically stable Phred-aware Bayesian consensus calling."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

BASES: Final = "ACGT"
_MAX_PHRED: Final = 60


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    """Consensus sequence and posterior-derived qualities."""

    sequence: str
    qualities: tuple[int, ...]
    posteriors: tuple[float, ...]


def _log_prior(prior: Mapping[str, float] | None) -> tuple[float, ...]:
    if prior is None:
        return (math.log(0.25),) * 4
    if set(prior) != set(BASES):
        raise ValueError("prior must contain exactly A, C, G, and T")
    values: tuple[float, ...] = tuple(float(prior[base]) for base in BASES)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("prior probabilities must be finite and positive")
    logs: tuple[float, ...] = tuple(math.log(value) for value in values)
    maximum = max(logs)
    log_total = maximum + math.log(sum(math.exp(value - maximum) for value in logs))
    return tuple(value - log_total for value in logs)


def _validate_reads(
    sequences: Sequence[str], qualities: Sequence[Sequence[int]]
) -> tuple[int, list[str], list[tuple[int, ...]]]:
    if not sequences:
        raise ValueError("at least one read is required")
    if len(sequences) != len(qualities):
        raise ValueError("sequence and quality read counts differ")
    normalised_sequences: list[str] = [sequence.upper() for sequence in sequences]
    length = len(normalised_sequences[0])
    if length == 0:
        raise ValueError("reads must not be empty")
    normalised_qualities: list[tuple[int, ...]] = []
    for index, (sequence, read_qualities) in enumerate(
        zip(normalised_sequences, qualities, strict=True), start=1
    ):
        if len(sequence) != length:
            raise ValueError(f"read {index} has a different sequence length")
        invalid = set(sequence) - set(BASES)
        if invalid:
            raise ValueError(
                f"read {index} contains unsupported base(s): {''.join(sorted(invalid))}"
            )
        quality_tuple: tuple[int, ...] = tuple(read_qualities)
        if len(quality_tuple) != length:
            raise ValueError(f"read {index} sequence and quality lengths differ")
        if any(
            isinstance(quality, bool)
            or not isinstance(quality, int)
            or quality < 0
            or quality > 93
            for quality in quality_tuple
        ):
            raise ValueError(f"read {index} qualities must be integers from 0 to 93")
        normalised_qualities.append(quality_tuple)
    ordered: list[tuple[str, tuple[int, ...]]] = sorted(
        zip(normalised_sequences, normalised_qualities, strict=True)
    )
    return (
        length,
        [sequence for sequence, _ in ordered],
        [quality for _, quality in ordered],
    )


def _posterior_quality(posterior: float) -> int:
    error_probability: float = max(1.0 - posterior, 10.0 ** (-_MAX_PHRED / 10.0))
    return min(_MAX_PHRED, round(-10.0 * math.log10(error_probability)))


def call_consensus(
    sequences: Sequence[str],
    qualities: Sequence[Sequence[int]],
    *,
    prior: Mapping[str, float] | None = None,
    min_posterior: float = 0.0,
) -> ConsensusResult:
    """Call a Bayesian consensus from aligned sequences and integer Phred scores.

    Each observation assigns probability ``1 - 10**(-Q/10)`` to its called base
    and distributes the error probability equally among the other three bases.
    Candidate likelihoods and priors are combined in natural-log space.
    """

    if not math.isfinite(min_posterior) or not 0.0 <= min_posterior <= 1.0:
        raise ValueError("min_posterior must be between 0 and 1")
    length, sequences_list, qualities_list = _validate_reads(sequences, qualities)
    log_prior: tuple[float, ...] = _log_prior(prior)
    consensus: list[str] = []
    consensus_qualities: list[int] = []
    posteriors: list[float] = []

    for position in range(length):
        scores: list[float] = list(log_prior)
        for sequence, read_qualities in zip(
            sequences_list, qualities_list, strict=True
        ):
            observed = sequence[position]
            error: float = 10.0 ** (-read_qualities[position] / 10.0)
            correct: float = 1.0 - error
            for candidate_index, candidate in enumerate(BASES):
                probability = correct if candidate == observed else error / 3.0
                if probability == 0.0:
                    scores[candidate_index] = -math.inf
                elif scores[candidate_index] != -math.inf:
                    scores[candidate_index] += math.log(probability)
        maximum = max(scores)
        if maximum == -math.inf:
            raise ValueError(
                f"position {position + 1} has zero probability for every candidate"
            )
        weights: list[float] = [math.exp(score - maximum) for score in scores]
        denominator: float = sum(weights)
        winner_index: int = max(range(4), key=weights.__getitem__)
        posterior: float = weights[winner_index] / denominator
        base = BASES[winner_index] if posterior >= min_posterior else "N"
        consensus.append(base)
        consensus_qualities.append(_posterior_quality(posterior))
        posteriors.append(posterior)

    return ConsensusResult(
        "".join(consensus), tuple(consensus_qualities), tuple(posteriors)
    )


def majority_consensus(sequences: Sequence[str]) -> str:
    """Return an unweighted majority consensus, resolving ties as A, C, G, T."""

    if not sequences:
        raise ValueError("at least one read is required")
    normalised: list[str] = [sequence.upper() for sequence in sequences]
    length = len(normalised[0])
    if length == 0:
        raise ValueError("reads must not be empty")
    for index, sequence in enumerate(normalised, start=1):
        if len(sequence) != length:
            raise ValueError(f"read {index} has a different sequence length")
        invalid = set(sequence) - set(BASES)
        if invalid:
            raise ValueError(
                f"read {index} contains unsupported base(s): {''.join(sorted(invalid))}"
            )
    return "".join(
        max(
            BASES,
            key=lambda base: (
                sum(read[pos] == base for read in normalised),
                -BASES.index(base),
            ),
        )
        for pos in range(length)
    )
