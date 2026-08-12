from hypothesis import given
from hypothesis import strategies as st

from phred_consensus import call_consensus


@st.composite
def identical_read_piles(
    draw: st.DrawFn,
) -> tuple[list[str], list[list[int]]]:
    length = draw(st.integers(min_value=1, max_value=40))
    read_count = draw(st.integers(min_value=1, max_value=16))
    sequence = draw(st.text(alphabet="ACGT", min_size=length, max_size=length))
    # At Q1, P(correct) < P(each alternative), so identity is mathematically false.
    qualities = draw(
        st.lists(
            st.lists(
                st.integers(min_value=2, max_value=60),
                min_size=length,
                max_size=length,
            ),
            min_size=read_count,
            max_size=read_count,
        )
    )
    return [sequence] * read_count, qualities


@st.composite
def aligned_read_piles(
    draw: st.DrawFn,
) -> tuple[list[str], list[list[int]], tuple[int, ...]]:
    length = draw(st.integers(min_value=1, max_value=40))
    read_count = draw(st.integers(min_value=1, max_value=16))
    reads = draw(
        st.lists(
            st.text(alphabet="ACGT", min_size=length, max_size=length),
            min_size=read_count,
            max_size=read_count,
        )
    )
    qualities = draw(
        st.lists(
            st.lists(
                st.integers(min_value=1, max_value=60),
                min_size=length,
                max_size=length,
            ),
            min_size=read_count,
            max_size=read_count,
        )
    )
    permutation = draw(st.permutations(tuple(range(read_count))))
    return reads, qualities, permutation


@given(identical_read_piles())
def test_identical_reads_produce_identical_consensus(
    pile: tuple[list[str], list[list[int]]],
) -> None:
    reads, qualities = pile

    assert call_consensus(reads, qualities).sequence == reads[0]


@given(aligned_read_piles())
def test_read_order_does_not_change_consensus(
    pile: tuple[list[str], list[list[int]], tuple[int, ...]],
) -> None:
    reads, qualities, permutation = pile
    expected = call_consensus(reads, qualities)

    assert call_consensus(reads[::-1], qualities[::-1]) == expected
    assert (
        call_consensus(
            [reads[index] for index in permutation],
            [qualities[index] for index in permutation],
        )
        == expected
    )


@given(aligned_read_piles())
def test_posteriors_and_qualities_remain_in_reported_bounds(
    pile: tuple[list[str], list[list[int]], tuple[int, ...]],
) -> None:
    reads, qualities, _ = pile
    result = call_consensus(reads, qualities)

    assert all(0.0 <= posterior <= 1.0 for posterior in result.posteriors)
    assert all(0 <= quality <= 60 for quality in result.qualities)
