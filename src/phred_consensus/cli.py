"""Command-line interface for grouped Bayesian consensus calling."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Literal, TextIO, cast

from .benchmark import run_benchmark
from .core import BASES, call_consensus
from .io import ReadGroup, parse_fastq, parse_tsv

InputFormat = Literal["fastq", "tsv"]
OutputFormat = Literal["fastq", "jsonl"]


class _Arguments(argparse.Namespace):
    input: Path | None
    output: Path | None
    input_format: InputFormat
    output_format: OutputFormat
    delimiter: str
    min_posterior: float
    prior: dict[str, float] | None
    benchmark: bool
    seed: int
    bases: int


def _prior(value: str) -> dict[str, float]:
    try:
        fields = value.split(",")
        parsed: dict[str, float] = {
            key.upper(): float(number)
            for key, number in (field.split("=", 1) for field in fields)
        }
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "prior must look like A=0.25,C=0.25,G=0.25,T=0.25"
        ) from error
    if set(parsed) != set(BASES):
        raise argparse.ArgumentTypeError("prior must specify A, C, G, and T once each")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phred-consensus",
        description="Call Bayesian consensus sequences from grouped aligned reads.",
    )
    parser.add_argument(
        "input", nargs="?", type=Path, help="input file (default: stdin)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="output file (default: stdout)"
    )
    parser.add_argument("--input-format", choices=("fastq", "tsv"), default="fastq")
    parser.add_argument("--output-format", choices=("fastq", "jsonl"), default="fastq")
    parser.add_argument(
        "--delimiter", default="/", help="FASTQ ID prefix delimiter (default: /)"
    )
    parser.add_argument("--min-posterior", type=float, default=0.0)
    parser.add_argument("--prior", type=_prior)
    parser.add_argument(
        "--benchmark", action="store_true", help="run the seeded synthetic benchmark"
    )
    parser.add_argument("--seed", type=int, default=2026, help="benchmark random seed")
    parser.add_argument(
        "--bases", type=int, default=2000, help="benchmark truth length"
    )
    return parser


def _quality_text(qualities: Sequence[int]) -> str:
    return "".join(chr(quality + 33) for quality in qualities)


def _write_groups(
    source: TextIO,
    destination: TextIO,
    *,
    input_format: InputFormat,
    output_format: OutputFormat,
    delimiter: str,
    prior: Mapping[str, float] | None,
    min_posterior: float,
) -> None:
    groups: Iterator[ReadGroup] = (
        parse_fastq(source, delimiter) if input_format == "fastq" else parse_tsv(source)
    )
    for group in groups:
        result = call_consensus(
            group.sequences,
            group.qualities,
            prior=prior,
            min_posterior=min_posterior,
        )
        if output_format == "fastq":
            destination.write(
                f"@{group.name}\n{result.sequence}\n+\n{_quality_text(result.qualities)}\n"
            )
        else:
            payload: dict[str, str | int | list[int] | list[float]] = {
                "group": group.name,
                "sequence": result.sequence,
                "quality": list(result.qualities),
                "posterior": [round(value, 12) for value in result.posteriors],
                "reads": len(group.sequences),
            }
            destination.write(json.dumps(payload, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""

    parser = build_parser()
    args = cast(_Arguments, parser.parse_args(argv))
    try:
        if args.benchmark:
            benchmark = run_benchmark(seed=args.seed, bases=args.bases)
            payload: dict[str, int | float] = {
                "seed": benchmark.seed,
                "bases": benchmark.bases,
                "reads": benchmark.reads,
                "bayesian_mismatches": benchmark.bayesian_mismatches,
                "bayesian_rate": benchmark.bayesian_rate,
                "majority_mismatches": benchmark.majority_mismatches,
                "majority_rate": benchmark.majority_rate,
            }
            print(json.dumps(payload, separators=(",", ":")))
            return 0
        input_context: AbstractContextManager[TextIO] = (
            args.input.open(encoding="ascii") if args.input else nullcontext(sys.stdin)
        )
        output_context: AbstractContextManager[TextIO] = (
            args.output.open("w", encoding="ascii")
            if args.output
            else nullcontext(sys.stdout)
        )
        with input_context as source, output_context as destination:
            _write_groups(
                source,
                destination,
                input_format=args.input_format,
                output_format=args.output_format,
                delimiter=args.delimiter,
                prior=args.prior,
                min_posterior=args.min_posterior,
            )
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(2, f"phred-consensus: error: {error}\n")
    return 0
