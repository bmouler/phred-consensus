import pytest

from phred_consensus.io import _group_records, decode_phred, parse_fastq, parse_tsv


def test_parse_fastq_groups_by_prefix_and_accepts_plus_annotation() -> None:
    groups = list(
        parse_fastq(
            [
                "@alpha/1 description\n",
                "AC\n",
                "+alpha/1\n",
                "II\n",
                "@alpha/2\n",
                "AT\n",
                "+\n",
                "I?\n",
                "@beta/1\n",
                "GG\n",
                "+\n",
                "!!\n",
            ],
            "/",
        )
    )

    assert [group.name for group in groups] == ["alpha", "beta"]
    assert groups[0].sequences == ("AC", "AT")
    assert groups[0].qualities == ((40, 40), (40, 30))
    assert groups[1].qualities == ((0, 0),)


def test_parse_tsv_groups_rows() -> None:
    groups = list(parse_tsv(["one\tAC\tII\n", "one\tAG\tI?\n", "two\tT\t5\n"]))

    assert [group.name for group in groups] == ["one", "two"]
    assert groups[1].qualities == ((20,),)


def test_fastq_yields_completed_group_before_consuming_the_file() -> None:
    consumed: list[str] = []

    def lines():
        for line in [
            "@alpha/1\n",
            "A\n",
            "+\n",
            "I\n",
            "@beta/1\n",
            "C\n",
            "+\n",
            "I\n",
        ]:
            consumed.append(line)
            yield line
        raise AssertionError("parser read past the first completed group")

    groups = parse_fastq(lines(), "/")
    assert next(groups).name == "alpha"
    assert consumed == [
        "@alpha/1\n",
        "A\n",
        "+\n",
        "I\n",
        "@beta/1\n",
        "C\n",
        "+\n",
        "I\n",
    ]


def test_tsv_yields_completed_group_before_consuming_the_file() -> None:
    consumed: list[str] = []

    def lines():
        for line in ["alpha\tA\tI\n", "beta\tC\tI\n"]:
            consumed.append(line)
            yield line
        raise AssertionError("parser read past the first completed group")

    groups = parse_tsv(lines())
    assert next(groups).name == "alpha"
    assert consumed == ["alpha\tA\tI\n", "beta\tC\tI\n"]


def test_noncontiguous_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="not contiguous"):
        list(parse_tsv(["a\tA\tI\n", "b\tA\tI\n", "a\tA\tI\n"]))


@pytest.mark.parametrize(
    ("lines", "delimiter", "message"),
    [
        ([], "/", "no records"),
        (["@a\n", "A\n"], "/", "incomplete"),
        (["a\n", "A\n", "+\n", "I\n"], "/", "invalid header"),
        (["@\n", "A\n", "+\n", "I\n"], "/", "invalid header"),
        (["@a\n", "A\n", "x\n", "I\n"], "/", "lacks a '\\+'"),
        (["@a\n", "AA\n", "+\n", "I\n"], "/", "lengths differ"),
        (["@a\n", "A\n", "+\n", " \n"], "/", "non-Phred"),
        (["@a\n", "A\n", "+\n", "\x7f\n"], "/", "non-Phred"),
        (["@a\n", "A\n", "+\n", "I\n"], "", "must not be empty"),
    ],
)
def test_fastq_errors(lines: list[str], delimiter: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        list(parse_fastq(lines, delimiter))


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([], "no records"),
        (["\n"], "is blank"),
        (["a\tA\n"], "three columns"),
        (["\tA\tI\n"], "group name"),
        (["a\tAA\tI\n"], "lengths differ"),
        (["a\tA\t \n"], "non-Phred"),
    ],
)
def test_tsv_errors(lines: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        list(parse_tsv(lines))


def test_decode_phred_boundaries() -> None:
    assert decode_phred("!~", context="quality") == (0, 93)


def test_empty_internal_record_stream_yields_nothing() -> None:
    assert list(_group_records([])) == []
