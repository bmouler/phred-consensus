"""Deterministic end-to-end benchmark for public Bayesian consensus calling."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections.abc import Mapping, Sequence

from phred_consensus import ConsensusResult, call_consensus

BASES = "ACGT"
DEFAULT_EXPECTED_CHECKSUM = (
    "21481dbcff097c4aaf679005f7433940f31343e1d48eac299ac7de7b8a20a841"
)

Group = tuple[list[str], list[tuple[int, ...]], Mapping[str, float] | None, float]


def make_groups(
    *, seed: int = 20260815, groups: int = 240, length: int = 180
) -> list[Group]:
    """Build fixed heterogeneous aligned-read families outside timed regions."""
    rng = random.Random(seed)
    result: list[Group] = []
    quality_levels = (3, 6, 10, 15, 20, 25, 30, 35, 40)
    for group_index in range(groups):
        depth = (5, 8, 12, 16, 20)[group_index % 5]
        truth = "".join(rng.choice(BASES) for _ in range(length))
        reads: list[str] = []
        qualities: list[tuple[int, ...]] = []
        for read_index in range(depth):
            read: list[str] = []
            read_qualities: list[int] = []
            for base in truth:
                quality = quality_levels[
                    (group_index + read_index + rng.randrange(5)) % len(quality_levels)
                ]
                error = 10.0 ** (-quality / 10.0)
                if rng.random() < error:
                    observed = BASES[(BASES.index(base) + 1 + rng.randrange(3)) % 4]
                else:
                    observed = base
                read.append(
                    observed.lower()
                    if (group_index + read_index) % 11 == 0
                    else observed
                )
                read_qualities.append(quality)
            reads.append("".join(read))
            qualities.append(tuple(read_qualities))
        prior = (
            {"A": 1.7, "C": 0.8, "G": 1.1, "T": 1.4} if group_index % 4 == 0 else None
        )
        threshold = (0.0, 0.8, 0.95)[group_index % 3]
        result.append((reads, qualities, prior, threshold))
    return result


def call_all(groups: Sequence[Group]) -> list[ConsensusResult]:
    """Materialize every result through the documented public API."""
    return [
        call_consensus(reads, qualities, prior=prior, min_posterior=threshold)
        for reads, qualities, prior, threshold in groups
    ]


def checksum(results: Sequence[ConsensusResult]) -> str:
    """Hash every discrete and binary-float result in stable group order."""
    digest = hashlib.sha256()
    for result in results:
        digest.update(result.sequence.encode("ascii"))
        digest.update(bytes(result.qualities))
        for posterior in result.posteriors:
            digest.update(posterior.hex().encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--groups", type=int, default=240)
    parser.add_argument("--length", type=int, default=180)
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 0:
        parser.error("samples must be positive and warmups nonnegative")

    groups = make_groups(groups=args.groups, length=args.length)
    reference = call_all(groups)
    reference_checksum = checksum(reference)
    if (
        args.groups == 240
        and args.length == 180
        and reference_checksum != DEFAULT_EXPECTED_CHECKSUM
    ):
        raise RuntimeError(
            f"expected default result digest {DEFAULT_EXPECTED_CHECKSUM}, "
            f"got {reference_checksum}"
        )
    for _ in range(args.warmups):
        if checksum(call_all(groups)) != reference_checksum:
            raise RuntimeError("consensus result changed during warmup")

    samples = []
    for _ in range(args.samples):
        started = time.perf_counter()
        measured = call_all(groups)
        samples.append(time.perf_counter() - started)
        if measured != reference:
            raise RuntimeError("consensus result changed between timed samples")

    depths = [len(reads) for reads, _, _, _ in groups]
    print(
        json.dumps(
            {
                "samples": samples,
                "median_seconds": statistics.median(samples),
                "min_seconds": min(samples),
                "max_seconds": max(samples),
                "groups": len(groups),
                "bases_per_read": args.length,
                "total_reads": sum(depths),
                "total_observations": sum(depths) * args.length,
                "checksum": reference_checksum,
                "repeat_exact": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
