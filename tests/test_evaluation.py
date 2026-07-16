from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime.evaluation import (
    chronological_split,
    compare_log_likelihoods,
    compare_models,
    conditional_log_likelihood,
    name_regimes,
    summarize_regimes,
)


def test_chronological_split_preserves_order_and_gap() -> None:
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    frame = pd.DataFrame({"value": np.arange(20)}, index=index)

    split = chronological_split(frame, train_fraction=0.5, validation_fraction=0.2, gap=1)

    assert split.train_indices == (0, 9)
    assert split.validation_indices == (10, 13)
    assert split.test_indices == (14, 20)
    assert split.train["value"].tolist() == list(range(9))
    assert split.validation["value"].tolist() == [10, 11, 12]
    assert split.test["value"].tolist() == list(range(14, 20))
    train, validation, test = split
    assert train.index.max() < validation.index.min() < test.index.min()


def test_chronological_split_supports_plain_sequences() -> None:
    split = chronological_split(list(range(10)), train_fraction=0.5, validation_fraction=0.2)
    assert split.train == [0, 1, 2, 3, 4]
    assert split.validation == [5, 6]
    assert split.test == [7, 8, 9]


def test_chronological_split_rejects_unsorted_time_index() -> None:
    frame = pd.DataFrame(
        {"value": [1, 2]},
        index=pd.to_datetime(["2022-01-02", "2022-01-01"]),
    )
    with pytest.raises(ValueError, match="sorted"):
        chronological_split(frame)


@pytest.mark.parametrize(
    "train_fraction, validation_fraction",
    [(0.0, 0.2), (0.8, 0.2), (0.6, 0.5)],
)
def test_chronological_split_rejects_invalid_fractions(
    train_fraction: float, validation_fraction: float
) -> None:
    with pytest.raises(ValueError):
        chronological_split(
            list(range(20)),
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )


def test_compare_log_likelihoods_reports_candidate_gain() -> None:
    comparison = compare_log_likelihoods(-120.0, -90.0, n_observations=30)

    assert comparison.baseline_per_observation == pytest.approx(-4.0)
    assert comparison.candidate_per_observation == pytest.approx(-3.0)
    assert comparison.delta_total == pytest.approx(30.0)
    assert comparison.delta_per_observation == pytest.approx(1.0)
    assert comparison.winner == "candidate"
    assert comparison.as_dict()["winner"] == "candidate"


def test_compare_models_calls_total_score_on_same_data() -> None:
    class ConstantModel:
        def __init__(self, result: float) -> None:
            self.result = result

        def score(self, X: np.ndarray) -> float:
            return self.result

    X = np.ones((10, 2))
    comparison = compare_models(ConstantModel(-50.0), ConstantModel(-45.0), X)
    assert comparison.n_observations == 10
    assert comparison.winner == "candidate"


def test_conditional_log_likelihood_uses_prefix_difference() -> None:
    class AdditiveModel:
        def score(self, X: np.ndarray) -> float:
            return float(X[:, 0].sum())

    observations = np.arange(1, 6, dtype=float).reshape(-1, 1)

    score = conditional_log_likelihood(AdditiveModel(), observations, start=3, stop=5)

    assert score == pytest.approx(4.0 + 5.0)


def test_summarize_three_regimes_and_assign_names() -> None:
    states = np.repeat([0, 1, 2], 4)
    returns = np.array(
        [0.001, 0.002, -0.001, 0.001, 0.01, 0.012, 0.008, 0.011, -0.04, -0.03, 0.02, -0.05]
    )
    volatility = np.repeat([0.01, 0.025, 0.09], 4)

    summary = summarize_regimes(states, returns, volatility)
    names = dict(zip(summary["state"], summary["regime"], strict=True))

    assert names == {0: "Calm", 1: "Trending", 2: "Crisis"}
    assert summary["observations"].sum() == len(states)
    assert summary["frequency"].sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(summary["annualized_return"]))


def test_name_four_regimes_uses_unique_descriptions() -> None:
    summary = pd.DataFrame(
        {
            "state": [0, 1, 2, 3],
            "mean_return": [0.001, 0.01, -0.006, -0.03],
            "mean_volatility": [0.01, 0.025, 0.04, 0.10],
        }
    )
    labels = name_regimes(summary)

    assert labels == {3: "Crisis", 0: "Calm", 1: "Bull", 2: "Bear"}


@pytest.mark.parametrize(
    "states, returns, volatility, message",
    [
        ([0, 1], [0.1], [0.1, 0.2], "equal lengths"),
        ([0, -1], [0.1, 0.2], [0.1, 0.2], "non-negative"),
        ([0, 1], [0.1, np.nan], [0.1, 0.2], "NaN"),
        ([0, 1], [0.1, 0.2], [0.1, -0.2], "cannot be negative"),
    ],
)
def test_summarize_regimes_validates_inputs(
    states: list[int],
    returns: list[float],
    volatility: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_regimes(states, returns, volatility)
