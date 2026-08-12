from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from phred_consensus.benchmark import run_benchmark
from phred_consensus.cli import _prior, main


def test_cli_writes_jsonl_from_tsv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "reads.tsv"
    source.write_text("sample\tA\t$\nsample\tC\tI\n", encoding="ascii")

    assert main([str(source), "--input-format", "tsv", "--output-format", "jsonl"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["group"] == "sample"
    assert payload["sequence"] == "C"
    assert payload["quality"] == [38]
    assert payload["reads"] == 2


def test_cli_writes_fastq_file_with_threshold_and_prior(tmp_path: Path) -> None:
    source = tmp_path / "reads.tsv"
    output = tmp_path / "consensus.fastq"
    source.write_text("x\tA\t5\nx\tC\t5\n", encoding="ascii")

    assert (
        main(
            [
                str(source),
                "--input-format",
                "tsv",
                "--output",
                str(output),
                "--min-posterior",
                "0.9",
                "--prior",
                "A=1,C=2,G=1,T=1",
            ]
        )
        == 0
    )
    assert output.read_text(encoding="ascii").splitlines()[:3] == ["@x", "N", "+"]


def test_cli_reads_fastq_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "sys.stdin",
        type(
            "Input",
            (),
            {"__iter__": lambda self: iter(["@g/1\n", "A\n", "+\n", "I\n"])},
        )(),
    )

    assert main([]) == 0
    assert capsys.readouterr().out == "@g\nA\n+\nI\n"


def test_cli_benchmark_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--benchmark", "--seed", "7", "--bases", "20"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["seed"] == 7
    assert payload["bases"] == 20
    assert payload["reads"] == 5
    assert set(payload) == {
        "seed",
        "bases",
        "reads",
        "bayesian_mismatches",
        "bayesian_rate",
        "majority_mismatches",
        "majority_rate",
    }


def test_cli_reports_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "bad.fastq"
    source.write_text("broken\n", encoding="ascii")

    with pytest.raises(SystemExit) as error:
        main([str(source)])
    assert error.value.code == 2
    assert "incomplete" in capsys.readouterr().err


def test_cli_reports_file_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "missing.fastq")])
    assert error.value.code == 2
    assert "No such file" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["bad", "A=1,C=1,G=1", "A=x,C=1,G=1,T=1"])
def test_prior_argument_rejects_malformed_values(value: str) -> None:
    with pytest.raises(Exception, match="prior"):
        _prior(value)


def test_benchmark_is_deterministic_and_bayesian_wins() -> None:
    first = run_benchmark()
    second = run_benchmark()

    assert first == second
    assert first.bayesian_mismatches < first.majority_mismatches
    assert first.bayesian_rate == first.bayesian_mismatches / first.bases
    assert first.majority_rate == first.majority_mismatches / first.bases


def test_benchmark_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        run_benchmark(bases=0)


def test_installed_module_cli_end_to_end_from_clean_directory(tmp_path: Path) -> None:
    source = tmp_path / "reads.fastq"
    source.write_text("@family/1\nAC\n+\nII\n@family/2\nAT\n+\nI$\n", encoding="ascii")
    command = Path(sys.executable).with_name("phred-consensus")

    completed = subprocess.run(
        [
            str(command),
            str(source),
            "--output-format",
            "jsonl",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["group"] == "family"
    assert payload["sequence"] == "AC"
    assert payload["reads"] == 2


def test_module_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("phred_consensus.cli.main", lambda: 17)
    with pytest.raises(SystemExit) as error:
        runpy.run_module("phred_consensus.__main__", run_name="__main__")
    assert error.value.code == 17
