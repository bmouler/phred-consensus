from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from phred_consensus.benchmark import run_benchmark
from phred_consensus.cli import _prior, _quality_text, build_parser, main


def test_cli_writes_jsonl_from_tsv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "reads.tsv"
    source.write_text("sample\tA\t$\nsample\tC\tI\n", encoding="ascii")

    assert main([str(source), "--input-format", "tsv", "--output-format", "jsonl"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["group"] == "sample"
    assert payload["sequence"] == "C"
    assert payload["quality"] == [38]
    assert payload["reads"] == 2
    assert payload["posterior"] == [pytest.approx(0.999833818102)]
    assert output == (
        '{"group":"sample","sequence":"C","quality":[38],'
        '"posterior":[0.999833818102],"reads":2}\n'
    )


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
    assert output.read_text(encoding="ascii") == "@x\nN\n+\n&\n"


def test_cli_prior_changes_jsonl_posterior(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "reads.tsv"
    source.write_text("x\tA\t+\n", encoding="ascii")

    assert (
        main(
            [
                str(source),
                "--input-format",
                "tsv",
                "--output-format",
                "jsonl",
                "--prior",
                "A=1,C=2,G=3,T=4",
            ]
        )
        == 0
    )
    assert (
        capsys.readouterr().out
        == '{"group":"x","sequence":"A","quality":[6],"posterior":[0.75],"reads":1}\n'
    )


def test_quality_text_encodes_adjacent_values_without_separator() -> None:
    assert _quality_text((0, 40, 93)) == "!I~"


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
    assert capsys.readouterr().out == ""


def test_cli_benchmark_json_is_compact(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--benchmark", "--seed", "7", "--bases", "20"]) == 0

    assert capsys.readouterr().out == (
        '{"seed":7,"bases":20,"reads":5,"bayesian_mismatches":0,'
        '"bayesian_rate":0.0,"majority_mismatches":0,"majority_rate":0.0}\n'
    )


def test_parser_help_and_defaults_are_public_cli_contract() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    args = parser.parse_args([])

    assert help_text.startswith("usage: phred-consensus")
    assert "Call Bayesian consensus sequences from grouped aligned reads." in help_text
    assert "input file (default: stdin)" in help_text
    assert "output file (default: stdout)" in help_text
    assert "FASTQ ID prefix delimiter (default: /)" in help_text
    assert "run the seeded synthetic benchmark" in help_text
    assert "benchmark random seed" in help_text
    assert "benchmark truth length" in help_text
    assert args.seed == 2026
    assert args.bases == 2000
    assert "input file (default: stdin)" in help_text
    assert any(
        rendering in help_text
        for rendering in ("-o OUTPUT, --output OUTPUT", "-o, --output OUTPUT")
    )
    assert "FASTQ ID prefix delimiter (default: /)" in help_text
    assert "run the seeded synthetic benchmark" in help_text
    assert "benchmark random seed" in help_text
    assert "benchmark truth length" in help_text
    assert "input file (default: stdin)" in help_text
    assert "output file (default: stdout)" in help_text
    assert "FASTQ ID prefix delimiter (default: /)" in help_text
    assert "run the seeded synthetic benchmark" in help_text
    assert "benchmark random seed" in help_text
    assert "benchmark truth length" in help_text


@pytest.mark.parametrize(
    "needle",
    [
        "Call Bayesian consensus sequences from grouped aligned reads.",
        "input file (default: stdin)",
        "output file (default: stdout)",
        "FASTQ ID prefix delimiter (default: /)",
        "run the seeded synthetic benchmark",
        "benchmark random seed",
        "benchmark truth length",
    ],
)
def test_parser_help_text_is_exactly_cased(needle: str) -> None:
    assert needle in build_parser().format_help()


def test_parser_description_is_exact() -> None:
    assert (
        build_parser().description
        == "Call Bayesian consensus sequences from grouped aligned reads."
    )


@pytest.mark.parametrize(
    "argument, expected",
    [
        ("input", "input file (default: stdin)"),
        ("output", "output file (default: stdout)"),
        ("delimiter", "FASTQ ID prefix delimiter (default: /)"),
        ("benchmark", "run the seeded synthetic benchmark"),
        ("seed", "benchmark random seed"),
        ("bases", "benchmark truth length"),
    ],
)
def test_each_parser_action_exposes_its_documented_help(
    argument: str, expected: str
) -> None:
    action = next(
        action for action in build_parser()._actions if action.dest == argument
    )
    assert action.help == expected


def test_cli_opens_explicit_files_in_ascii_text_mode(tmp_path: Path) -> None:
    source = tmp_path / "reads.tsv"
    output = tmp_path / "out.fastq"
    source.write_text("g\tA\tI\n", encoding="ascii")

    original_open = Path.open
    calls: list[tuple[Path, tuple[object, ...], dict[str, object]]] = []

    def recording_open(path: Path, *args: object, **kwargs: object):
        calls.append((path, args, kwargs))
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", recording_open):
        assert (
            main(
                [
                    str(source),
                    "--input-format",
                    "tsv",
                    "--output",
                    str(output),
                ]
            )
            == 0
        )

    assert calls == [
        (source, (), {"encoding": "ascii"}),
        (output, ("w",), {"encoding": "ascii"}),
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--input-format", "bad"],
        ["--output-format", "bad"],
    ],
)
def test_parser_rejects_unknown_formats(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(arguments)

    assert error.value.code == 2


def test_parser_preserves_short_output_option_and_literal_format_choices() -> None:
    parser = build_parser()
    assert parser.parse_args(["-o", "x"]).output == Path("x")
    assert parser.parse_args(["--input-format", "fastq"]).input_format == "fastq"
    assert parser.parse_args(["--output-format", "fastq"]).output_format == "fastq"


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


def test_prior_argument_parses_lowercase_and_rejects_duplicate_or_extra_equals() -> (
    None
):
    assert _prior("a=1,c=2,g=3,t=4") == {"A": 1, "C": 2, "G": 3, "T": 4}

    with pytest.raises(Exception, match="look like"):
        _prior("A=1=2,C=1,G=1,T=1")

    with pytest.raises(Exception, match="A, C, G, and T once each"):
        _prior("A=1,A=2,C=1,G=1,T=1")


def test_prior_errors_are_exact_cli_contract() -> None:
    with pytest.raises(Exception) as malformed:
        _prior("bad")
    with pytest.raises(Exception) as missing:
        _prior("A=1,C=1,G=1")

    assert str(malformed.value) == "prior must look like A=0.25,C=0.25,G=0.25,T=0.25"
    assert str(missing.value) == "prior must specify A, C, G, and T once each"


def test_benchmark_is_deterministic_and_bayesian_wins() -> None:
    first = run_benchmark()
    second = run_benchmark()

    assert first == second
    assert first.bayesian_mismatches < first.majority_mismatches
    assert first.bayesian_rate == first.bayesian_mismatches / first.bases
    assert first.majority_rate == first.majority_mismatches / first.bases
    assert first == run_benchmark(seed=2026, bases=2000)
    assert first == type(first)(
        seed=2026,
        bases=2000,
        reads=5,
        bayesian_mismatches=1,
        majority_mismatches=128,
    )


def test_benchmark_accepts_single_base_boundary() -> None:
    assert run_benchmark(bases=1).bases == 1


def test_benchmark_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        run_benchmark(bases=0)


def test_benchmark_nonpositive_error_is_exact() -> None:
    with pytest.raises(ValueError) as error:
        run_benchmark(bases=0)

    assert str(error.value) == "bases must be positive"


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
