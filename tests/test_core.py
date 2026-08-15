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


def test_default_prior_is_a_normalized_uniform_distribution() -> None:
    result = call_consensus(["A"], [[0]])

    assert result.posteriors == pytest.approx((1 / 3,))


def test_nonuniform_prior_posterior_is_numerically_exact() -> None:
    result = call_consensus(
        ["A"],
        [[10]],
        prior={"A": 1, "C": 2, "G": 3, "T": 4},
    )

    assert result.sequence == "A"
    assert result.posteriors == pytest.approx((0.75,))
    assert result.qualities == (6,)


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


def test_read_validation_errors_report_one_based_indices_and_invalid_bases() -> None:
    with pytest.raises(ValueError) as error:
        call_consensus(["AA", "XN"], [[20, 20], [20, 20]])
    assert str(error.value) == "read 2 contains unsupported base(s): NX"

    with pytest.raises(ValueError, match="read 2 has a different sequence length"):
        call_consensus(["AA", "A"], [[20, 20], [20]])


def test_phred_93_is_accepted() -> None:
    result = call_consensus(["A"], [[93]])

    assert result.sequence == "A"
    assert result.qualities == (60,)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.inf, math.nan])
def test_call_consensus_rejects_bad_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        call_consensus(["A"], [[20]], min_posterior=threshold)


def test_threshold_one_is_valid_and_equality_is_inclusive() -> None:
    certain = call_consensus(
        ["A"],
        [[20]],
        prior={"A": 1e308, "C": 5e-324, "G": 5e-324, "T": 5e-324},
        min_posterior=1.0,
    )
    exact = call_consensus(["A", "C"], [[20], [20]])
    inclusive = call_consensus(
        ["A", "C"], [[20], [20]], min_posterior=exact.posteriors[0]
    )

    assert certain.sequence == "A"
    assert certain.posteriors == (1.0,)
    assert inclusive.sequence == "A"


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


def test_validation_error_messages_are_exact() -> None:
    cases = [
        (([], []), "at least one read is required"),
        ((["A"], []), "sequence and quality read counts differ"),
        (([""], [[]]), "reads must not be empty"),
    ]
    for arguments, message in cases:
        with pytest.raises(ValueError) as error:
            call_consensus(*arguments)
        assert str(error.value) == message

    with pytest.raises(ValueError) as prior_keys:
        call_consensus(["A"], [[20]], prior={"A": 1})
    assert str(prior_keys.value) == "prior must contain exactly A, C, G, and T"

    with pytest.raises(ValueError) as prior_values:
        call_consensus(["A"], [[20]], prior=dict.fromkeys("ACGT", 0))
    assert str(prior_values.value) == "prior probabilities must be finite and positive"

    with pytest.raises(ValueError) as threshold:
        call_consensus(["A"], [[20]], min_posterior=-1)
    assert str(threshold.value) == "min_posterior must be between 0 and 1"


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


def test_integer_subclass_qualities_preserve_q_zero_bayesian_behavior() -> None:
    class QualityInt(int):
        pass

    one_observation = call_consensus(["A"], [[QualityInt(0)]])
    conflicting_observations = call_consensus(
        ["A", "C"], [[QualityInt(0)], [QualityInt(0)]]
    )

    assert one_observation.sequence == "C"
    assert one_observation.qualities == (2,)
    assert one_observation.posteriors == (1 / 3,)
    assert conflicting_observations.sequence == "G"
    assert conflicting_observations.qualities == (3,)
    assert conflicting_observations.posteriors == (0.5,)


def test_zero_probability_error_reports_one_based_position() -> None:
    with pytest.raises(ValueError, match="position 2 has zero probability"):
        call_consensus(
            ["AA", "AC", "AG", "AT"],
            [[20, 0], [20, 0], [20, 0], [20, 0]],
        )


def test_majority_tie_and_lowercase() -> None:
    assert majority_consensus(["ac", "ca"]) == "AA"


def test_majority_errors_report_one_based_indices_and_invalid_bases() -> None:
    with pytest.raises(ValueError, match=r"read 2 contains unsupported base\(s\): NX"):
        majority_consensus(["AA", "XN"])

    with pytest.raises(ValueError, match="read 2 has a different sequence length"):
        majority_consensus(["AA", "A"])


def test_majority_validation_messages_are_exact() -> None:
    with pytest.raises(ValueError) as missing:
        majority_consensus([])
    with pytest.raises(ValueError) as empty:
        majority_consensus([""])

    assert str(missing.value) == "at least one read is required"
    assert str(empty.value) == "reads must not be empty"


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


def test_majority_invalid_bases_message_is_exact() -> None:
    with pytest.raises(ValueError) as error:
        majority_consensus(["XN"])

    assert str(error.value) == "read 1 contains unsupported base(s): NX"
