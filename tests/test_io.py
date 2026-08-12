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


def test_fastq_parses_crlf_without_stripping_sequence_whitespace() -> None:
    groups = list(parse_fastq(["@g/x/y note\r\n", " A\r\n", "+\r\n", "!I\r\n"], "/"))

    assert groups[0].name == "g"
    assert groups[0].sequences == (" A",)
    assert groups[0].qualities == ((0, 40),)


def test_fastq_uses_the_first_delimiter_in_an_identifier() -> None:
    groups = list(parse_fastq(["@family/child/read\n", "A\n", "+\n", "I\n"], "/"))

    assert groups[0].name == "family"


def test_fastq_strips_only_line_endings_from_all_record_lines() -> None:
    groups = list(
        parse_fastq(
            ["@g/1 \t\r\n", "A \t\r\n", "+ \t\r\n", "II?\r\n"],
            "/",
        )
    )

    assert groups[0].name == "g"
    assert groups[0].sequences == ("A \t",)
    assert groups[0].qualities == ((40, 40, 30),)


def test_fastq_line_endings_do_not_consume_valid_trailing_characters() -> None:
    groups = list(
        parse_fastq(
            ["@groupX\n", "AX\n", "+\n", "IX\n"],
            "/",
        )
    )

    assert groups[0].name == "groupX"
    assert groups[0].sequences == ("AX",)
    assert groups[0].qualities == ((40, 55),)


def test_fastq_identifier_ends_at_first_whitespace() -> None:
    groups = list(parse_fastq(["@g note more\n", "A\n", "+\n", "I\n"], "/"))

    assert groups[0].name == "g"


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


def test_group_record_error_preserves_group_name() -> None:
    with pytest.raises(ValueError, match="group 'a' is not contiguous"):
        list(
            _group_records(
                [
                    ("a", "A", (40,)),
                    ("b", "A", (40,)),
                    ("a", "A", (40,)),
                ]
            )
        )


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


def test_fastq_error_messages_include_exact_record_number_and_context() -> None:
    with pytest.raises(ValueError, match="FASTQ record 2 is incomplete"):
        list(
            parse_fastq(
                ["@a/1\n", "A\n", "+\n", "I\n", "@a/2\n", "A\n"],
                "/",
            )
        )

    with pytest.raises(
        ValueError, match="FASTQ record 2: quality contains a non-Phred"
    ):
        list(
            parse_fastq(
                [
                    "@a/1\n",
                    "A\n",
                    "+\n",
                    "I\n",
                    "@a/2\n",
                    "A\n",
                    "+\n",
                    " \n",
                ],
                "/",
            )
        )


def test_fastq_empty_delimiter_error_is_exact() -> None:
    with pytest.raises(ValueError) as error:
        list(parse_fastq([], ""))

    assert str(error.value) == "ID delimiter must not be empty"


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


def test_tsv_preserves_spaces_and_reports_exact_line_context() -> None:
    groups = list(parse_tsv(["g\t A\t!I\r\n"]))
    assert groups[0].sequences == (" A",)

    with pytest.raises(ValueError, match="TSV line 2: quality contains a non-Phred"):
        list(parse_tsv(["g\tA\tI\n", "g\tA\t \n"]))


def test_tsv_line_endings_do_not_consume_valid_trailing_characters() -> None:
    groups = list(parse_tsv(["g\tX\tX\n"]))

    assert groups[0].sequences == ("X",)
    assert groups[0].qualities == ((55,),)


def test_empty_input_and_group_name_messages_are_exact() -> None:
    with pytest.raises(ValueError) as fastq_error:
        list(parse_fastq([], "/"))
    with pytest.raises(ValueError) as tsv_error:
        list(parse_tsv([]))
    with pytest.raises(ValueError) as group_error:
        list(_group_records([("", "A", (40,))]))

    assert str(fastq_error.value) == "input contains no records"
    assert str(tsv_error.value) == "input contains no records"
    assert str(group_error.value) == "group name must not be empty"


def test_decode_phred_boundaries() -> None:
    assert decode_phred("!~", context="quality") == (0, 93)


def test_empty_internal_record_stream_yields_nothing() -> None:
    assert list(_group_records([])) == []
