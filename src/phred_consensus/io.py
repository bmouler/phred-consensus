"""Strict FASTQ and TSV parsing helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReadGroup:
    """A named group of aligned reads."""

    name: str
    sequences: tuple[str, ...]
    qualities: tuple[tuple[int, ...], ...]


def decode_phred(text: str, *, context: str) -> tuple[int, ...]:
    """Decode Sanger FASTQ quality text (Phred+33)."""

    qualities = tuple(ord(character) - 33 for character in text)
    if any(quality < 0 or quality > 93 for quality in qualities):
        raise ValueError(f"{context}: quality contains a non-Phred+33 character")
    return qualities


def _group_records(
    records: Iterable[tuple[str, str, tuple[int, ...]]],
) -> Iterator[ReadGroup]:
    current_name: str | None = None
    sequences: list[str] = []
    qualities: list[tuple[int, ...]] = []
    seen: set[str] = set()
    for name, sequence, quality in records:
        if not name:
            raise ValueError("group name must not be empty")
        if name != current_name:
            if name in seen:
                raise ValueError(f"group {name!r} is not contiguous")
            if current_name is not None:
                yield ReadGroup(current_name, tuple(sequences), tuple(qualities))
            seen.add(name)
            current_name = name
            sequences = []
            qualities = []
        sequences.append(sequence)
        qualities.append(quality)
    if current_name is not None:
        yield ReadGroup(current_name, tuple(sequences), tuple(qualities))


def parse_fastq(lines: Iterable[str], delimiter: str) -> Iterator[ReadGroup]:
    """Parse FASTQ and group adjacent records by ID prefix."""

    if not delimiter:
        raise ValueError("ID delimiter must not be empty")

    def records() -> Iterator[tuple[str, str, tuple[int, ...]]]:
        iterator = iter(lines)
        record_number = 0
        while True:
            try:
                header = next(iterator).rstrip("\r\n")
            except StopIteration:
                if record_number == 0:
                    raise ValueError("input contains no records") from None
                return
            record_number += 1
            try:
                sequence = next(iterator).rstrip("\r\n")
                plus = next(iterator).rstrip("\r\n")
                quality_text = next(iterator).rstrip("\r\n")
            except StopIteration as error:
                raise ValueError(
                    f"FASTQ record {record_number} is incomplete"
                ) from error
            if not header.startswith("@") or len(header) == 1:
                raise ValueError(f"FASTQ record {record_number} has an invalid header")
            if not plus.startswith("+"):
                raise ValueError(f"FASTQ record {record_number} lacks a '+' separator")
            identifier = header[1:].split(maxsplit=1)[0]
            group = identifier.split(delimiter, maxsplit=1)[0]
            quality = decode_phred(
                quality_text, context=f"FASTQ record {record_number}"
            )
            if len(sequence) != len(quality):
                raise ValueError(
                    f"FASTQ record {record_number} sequence and quality lengths differ"
                )
            yield group, sequence, quality

    yield from _group_records(records())


def parse_tsv(lines: Iterable[str]) -> Iterator[ReadGroup]:
    """Parse headerless group, sequence, Phred+33-quality TSV rows."""

    def records() -> Iterator[tuple[str, str, tuple[int, ...]]]:
        saw_record = False
        for line_number, raw_line in enumerate(lines, start=1):
            saw_record = True
            line = raw_line.rstrip("\r\n")
            if not line:
                raise ValueError(f"TSV line {line_number} is blank")
            fields = line.split("\t")
            if len(fields) != 3:
                raise ValueError(f"TSV line {line_number} must contain three columns")
            group, sequence, quality_text = fields
            quality = decode_phred(quality_text, context=f"TSV line {line_number}")
            if len(sequence) != len(quality):
                raise ValueError(
                    f"TSV line {line_number} sequence and quality lengths differ"
                )
            yield group, sequence, quality
        if not saw_record:
            raise ValueError("input contains no records")

    yield from _group_records(records())
