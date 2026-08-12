import math

import pytest

from phred_consensus import call_consensus, majority_consensus


def test_single_read_preserves_sequence_and_high_confidence() -> None:
    result = call_consensus(["acgt"], [[40, 40, 40, 40]])

    assert result.sequence == "ACGT"
    assert result.qualities == (40, 40, 40, 40)
    assert all(posterior > 0.9999 for posterior in result.posteriors)


def test_quality_weighting_overrides_unweighted_majority() -> None:
    sequences = ["A", "A", "C"]

    assert majority_consensus(sequences) == "A"
    assert call_consensus(sequences, [[3], [3], [40]]).sequence == "C"


def test_tie_resolves_in_base_order() -> None:
    result = call_consensus(["A", "C"], [[20], [20]])

    assert result.sequence == "A"
    assert result.posteriors[0] == pytest.approx(0.498322147651)


def test_threshold_emits_n_without_discarding_quality_or_posterior() -> None:
    result = call_consensus(["A", "C"], [[20], [20]], min_posterior=0.5)

    assert result.sequence == "N"
    assert result.qualities == (3,)
    assert result.posteriors[0] < 0.5


def test_quality_is_capped_at_sixty() -> None:
    result = call_consensus(["A", "A"], [[60], [60]])

    assert result.sequence == "A"
    assert result.qualities == (60,)


def test_nonuniform_prior_changes_ambiguous_call() -> None:
    result = call_consensus(
        ["A", "C"],
        [[20], [20]],
        prior={"A": 1, "C": 4, "G": 1, "T": 1},
    )

    assert result.sequence == "C"


def test_prior_is_normalised() -> None:
    first = call_consensus(["G"], [[10]], prior=dict.fromkeys("ACGT", 1.0))
    second = call_consensus(["G"], [[10]], prior=dict.fromkeys("ACGT", 8.0))

    assert first == second


@pytest.mark.parametrize(
    ("sequences", "qualities", "message"),
    [
        ([], [], "at least one read"),
        (["A"], [], "read counts differ"),
        ([""], [[]], "must not be empty"),
        (["A", "AA"], [[20], [20, 20]], "different sequence length"),
        (["X"], [[20]], "unsupported base"),
        (["AA"], [[20]], "sequence and quality lengths differ"),
        (["A"], [[-1]], "integers from 0 to 93"),
        (["A"], [[94]], "integers from 0 to 93"),
        (["A"], [[True]], "integers from 0 to 93"),
        (["A"], [[1.5]], "integers from 0 to 93"),
    ],
)
def test_call_consensus_rejects_bad_reads(
    sequences: list[str], qualities: list[list[object]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        call_consensus(sequences, qualities)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.inf, math.nan])
def test_call_consensus_rejects_bad_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        call_consensus(["A"], [[20]], min_posterior=threshold)


@pytest.mark.parametrize(
    "prior",
    [
        {"A": 1, "C": 1, "G": 1},
        {"A": 1, "C": 1, "G": 1, "T": 0},
        {"A": 1, "C": 1, "G": 1, "T": -1},
        {"A": 1, "C": 1, "G": 1, "T": math.inf},
        {"A": 1, "C": 1, "G": 1, "T": math.nan},
    ],
)
def test_call_consensus_rejects_bad_prior(prior: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="prior"):
        call_consensus(["A"], [[20]], prior=prior)


def test_extreme_finite_prior_weights_normalise_in_log_space() -> None:
    equal = call_consensus(["A"], [[20]], prior=dict.fromkeys("ACGT", 1e308))
    skewed = call_consensus(
        ["A"],
        [[20]],
        prior={"A": 1e308, "C": 5e-324, "G": 5e-324, "T": 5e-324},
    )
    assert equal.sequence == "A"
    assert skewed.sequence == "A"
    assert skewed.posteriors == (1.0,)


def test_mutually_impossible_q_zero_observations_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero probability"):
        call_consensus(list("ACGT"), [[0], [0], [0], [0]])


def test_majority_tie_and_lowercase() -> None:
    assert majority_consensus(["ac", "ca"]) == "AA"


@pytest.mark.parametrize(
    ("sequences", "message"),
    [
        ([], "at least one read"),
        ([""], "must not be empty"),
        (["A", "AA"], "different sequence length"),
        (["N"], "unsupported base"),
    ],
)
def test_majority_rejects_bad_reads(sequences: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        majority_consensus(sequences)
