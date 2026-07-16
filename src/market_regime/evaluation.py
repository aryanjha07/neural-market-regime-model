"""Leakage-aware evaluation helpers for market-regime models."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import floor, sqrt
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

T = TypeVar("T")


@dataclass(frozen=True)
class ChronologicalSplit(Generic[T]):
    """Three time-ordered, non-overlapping partitions."""

    train: T
    validation: T
    test: T
    train_indices: tuple[int, int]
    validation_indices: tuple[int, int]
    test_indices: tuple[int, int]

    def __iter__(self) -> Iterator[T]:
        """Allow ``train, validation, test = chronological_split(...)``."""

        yield self.train
        yield self.validation
        yield self.test


@dataclass(frozen=True)
class LikelihoodComparison:
    """Held-out likelihood comparison, with totals and normalized values."""

    baseline_total: float
    candidate_total: float
    n_observations: int
    baseline_per_observation: float
    candidate_per_observation: float
    delta_total: float
    delta_per_observation: float
    winner: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "baseline_total": self.baseline_total,
            "candidate_total": self.candidate_total,
            "n_observations": self.n_observations,
            "baseline_per_observation": self.baseline_per_observation,
            "candidate_per_observation": self.candidate_per_observation,
            "delta_total": self.delta_total,
            "delta_per_observation": self.delta_per_observation,
            "winner": self.winner,
        }


def chronological_split(
    data: T,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    gap: int = 0,
    require_sorted_index: bool = True,
) -> ChronologicalSplit[T]:
    """Split ordered data without shuffling or overlapping observations.

    ``gap`` observations are omitted between adjacent partitions. A gap is
    useful when features or targets span multiple dates. Fractions are applied
    to the rows left after those gaps, so every returned partition is nonempty.
    Pandas objects retain their original index and columns.
    """

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    if isinstance(gap, bool) or not isinstance(gap, (int, np.integer)) or gap < 0:
        raise ValueError("gap must be a non-negative integer")

    try:
        n_observations = len(data)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("data must be a sized, sliceable object") from exc

    if require_sorted_index and isinstance(data, (pd.DataFrame, pd.Series)):
        if not data.index.is_monotonic_increasing:
            raise ValueError("data index must be sorted in increasing chronological order")

    usable = n_observations - 2 * int(gap)
    if usable < 3:
        raise ValueError("data is too short for three nonempty partitions and the gap")

    train_count = floor(usable * train_fraction)
    validation_count = floor(usable * validation_fraction)
    test_count = usable - train_count - validation_count
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("split fractions produce an empty partition")

    train_start, train_stop = 0, train_count
    validation_start = train_stop + int(gap)
    validation_stop = validation_start + validation_count
    test_start = validation_stop + int(gap)
    test_stop = n_observations

    return ChronologicalSplit(
        train=_slice_rows(data, train_start, train_stop),
        validation=_slice_rows(data, validation_start, validation_stop),
        test=_slice_rows(data, test_start, test_stop),
        train_indices=(train_start, train_stop),
        validation_indices=(validation_start, validation_stop),
        test_indices=(test_start, test_stop),
    )


def compare_log_likelihoods(
    baseline_log_likelihood: float,
    candidate_log_likelihood: float,
    n_observations: int,
) -> LikelihoodComparison:
    """Compare total held-out log-likelihoods on the exact same observations.

    Per-observation delta is the portable headline metric because raw totals
    depend on sequence length. A percentage of log-likelihood is intentionally
    not reported because it changes under harmless density-unit changes.
    """

    if isinstance(n_observations, bool) or not isinstance(n_observations, (int, np.integer)):
        raise ValueError("n_observations must be a positive integer")
    if n_observations <= 0:
        raise ValueError("n_observations must be a positive integer")

    baseline = _finite_scalar(baseline_log_likelihood, "baseline_log_likelihood")
    candidate = _finite_scalar(candidate_log_likelihood, "candidate_log_likelihood")
    delta = candidate - baseline
    tolerance = 1e-12
    if delta > tolerance:
        winner = "candidate"
    elif delta < -tolerance:
        winner = "baseline"
    else:
        winner = "tie"

    return LikelihoodComparison(
        baseline_total=baseline,
        candidate_total=candidate,
        n_observations=int(n_observations),
        baseline_per_observation=baseline / n_observations,
        candidate_per_observation=candidate / n_observations,
        delta_total=delta,
        delta_per_observation=delta / n_observations,
        winner=winner,
    )


def compare_models(
    baseline_model: Any,
    candidate_model: Any,
    X_test: ArrayLike,
) -> LikelihoodComparison:
    """Score two fitted models on one held-out matrix and compare them."""

    try:
        n_observations = len(X_test)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("X_test must be a sized object") from exc
    if n_observations == 0:
        raise ValueError("X_test cannot be empty")
    if not callable(getattr(baseline_model, "score", None)):
        raise TypeError("baseline_model must provide score(X)")
    if not callable(getattr(candidate_model, "score", None)):
        raise TypeError("candidate_model must provide score(X)")

    baseline_score = _to_scalar(baseline_model.score(X_test), "baseline score")
    candidate_score = _to_scalar(candidate_model.score(X_test), "candidate score")
    return compare_log_likelihoods(baseline_score, candidate_score, n_observations=n_observations)


def conditional_log_likelihood(
    model: Any,
    observations: ArrayLike,
    *,
    start: int,
    stop: int | None = None,
) -> float:
    """Score a held-out suffix conditional on all observations before it.

    Scoring a sliced HMM sequence restarts from its learned initial-state
    distribution. The difference between two prefix scores instead preserves
    the filtered state at the evaluation boundary.
    """

    values = np.asarray(observations)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("observations must be a two-dimensional sequence")
    resolved_stop = len(values) if stop is None else stop
    if isinstance(start, bool) or not isinstance(start, (int, np.integer)):
        raise ValueError("start must be an integer")
    if isinstance(resolved_stop, bool) or not isinstance(resolved_stop, (int, np.integer)):
        raise ValueError("stop must be an integer")
    if not 1 <= int(start) < int(resolved_stop) <= len(values):
        raise ValueError("require 1 <= start < stop <= len(observations)")
    if not callable(getattr(model, "score", None)):
        raise TypeError("model must provide score(X)")

    prefix_score = _to_scalar(model.score(values[: int(start)]), "prefix score")
    full_score = _to_scalar(model.score(values[: int(resolved_stop)]), "full score")
    conditional_score = full_score - prefix_score
    if not np.isfinite(conditional_score):
        raise ValueError("conditional log-likelihood must be finite")
    return conditional_score


def summarize_regimes(
    states: ArrayLike,
    returns: ArrayLike,
    volatility: ArrayLike,
    *,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """Characterize observed states and assign deterministic descriptive names.

    Names are interpretation aids, not labels learned by the HMM. For three
    states the heuristic produces ``Calm``, ``Trending``, and ``Crisis`` from
    mean return and volatility. Four states produce ``Calm``, ``Bull``,
    ``Bear``, and ``Crisis``.
    """

    if (
        isinstance(annualization_factor, bool)
        or not isinstance(annualization_factor, (int, np.integer))
        or annualization_factor <= 0
    ):
        raise ValueError("annualization_factor must be a positive integer")

    state_values = np.asarray(states)
    return_values = _one_dimensional_finite(returns, "returns")
    volatility_values = _one_dimensional_finite(volatility, "volatility")
    if state_values.ndim != 1:
        raise ValueError("states must be one-dimensional")
    if len(state_values) == 0:
        raise ValueError("states cannot be empty")
    if not (len(state_values) == len(return_values) == len(volatility_values)):
        raise ValueError("states, returns, and volatility must have equal lengths")
    if not np.all(np.isfinite(state_values)):
        raise ValueError("states contains NaN or infinite values")
    if not np.all(np.equal(state_values, np.floor(state_values))) or np.any(state_values < 0):
        raise ValueError("states must contain non-negative integers")
    if np.any(volatility_values < 0):
        raise ValueError("volatility cannot be negative")

    frame = pd.DataFrame(
        {
            "state": state_values.astype(np.int64),
            "return": return_values,
            "volatility": volatility_values,
        }
    )
    summary = (
        frame.groupby("state", sort=True, as_index=False)
        .agg(
            observations=("state", "size"),
            mean_return=("return", "mean"),
            median_return=("return", "median"),
            return_std=("return", "std"),
            mean_volatility=("volatility", "mean"),
        )
        .fillna({"return_std": 0.0})
    )
    summary["frequency"] = summary["observations"] / len(frame)
    summary["annualized_return"] = summary["mean_return"] * annualization_factor
    summary["annualized_volatility"] = summary["mean_volatility"] * sqrt(annualization_factor)

    labels = name_regimes(summary)
    summary["regime"] = summary["state"].map(labels)
    return summary[
        [
            "state",
            "regime",
            "observations",
            "frequency",
            "mean_return",
            "median_return",
            "return_std",
            "mean_volatility",
            "annualized_return",
            "annualized_volatility",
        ]
    ]


def name_regimes(summary: pd.DataFrame) -> dict[int, str]:
    """Map numeric states to descriptive names using return and risk profiles."""

    required = {"state", "mean_return", "mean_volatility"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing required columns: {sorted(missing)}")
    if summary.empty:
        raise ValueError("summary cannot be empty")
    if summary["state"].duplicated().any():
        raise ValueError("summary must contain exactly one row per state")
    for column in ("state", "mean_return", "mean_volatility"):
        if not np.all(np.isfinite(np.asarray(summary[column], dtype=np.float64))):
            raise ValueError(f"summary column {column!r} must contain finite values")

    working = summary.loc[:, ["state", "mean_return", "mean_volatility"]].copy()
    working["state"] = working["state"].astype(int)
    states = working["state"].tolist()
    if len(states) == 1:
        return {states[0]: "Calm"}

    return_scale = float(working["mean_return"].std(ddof=0)) or 1.0
    volatility_scale = float(working["mean_volatility"].std(ddof=0)) or 1.0
    return_z = (working["mean_return"] - working["mean_return"].mean()) / return_scale
    volatility_z = (
        working["mean_volatility"] - working["mean_volatility"].mean()
    ) / volatility_scale
    working["crisis_score"] = volatility_z - return_z

    labels: dict[int, str] = {}
    crisis_index = working.sort_values(
        ["crisis_score", "mean_volatility", "state"],
        ascending=[False, False, True],
    ).index[0]
    labels[int(working.loc[crisis_index, "state"])] = "Crisis"
    remaining = working.drop(index=crisis_index)

    if not remaining.empty:
        calm_index = remaining.sort_values(
            ["mean_volatility", "mean_return", "state"],
            ascending=[True, False, True],
        ).index[0]
        labels[int(remaining.loc[calm_index, "state"])] = "Calm"
        remaining = remaining.drop(index=calm_index)

    if len(states) == 3 and len(remaining) == 1:
        labels[int(remaining.iloc[0]["state"])] = "Trending"
        return labels

    if not remaining.empty:
        bull_index = remaining.sort_values(
            ["mean_return", "mean_volatility", "state"],
            ascending=[False, True, True],
        ).index[0]
        labels[int(remaining.loc[bull_index, "state"])] = "Bull"
        remaining = remaining.drop(index=bull_index)

    if not remaining.empty:
        bear_index = remaining.sort_values(
            ["mean_return", "mean_volatility", "state"],
            ascending=[True, False, True],
        ).index[0]
        labels[int(remaining.loc[bear_index, "state"])] = "Bear"
        remaining = remaining.drop(index=bear_index)

    for position, (_, row) in enumerate(remaining.sort_values("mean_return").iterrows(), start=1):
        labels[int(row["state"])] = f"Transitional {position}"
    return labels


def _slice_rows(data: T, start: int, stop: int) -> T:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[start:stop]  # type: ignore[return-value]
    try:
        return data[start:stop]  # type: ignore[index, return-value]
    except (TypeError, KeyError) as exc:
        raise ValueError("data must support positional slicing") from exc


def _finite_scalar(value: Any, name: str) -> float:
    scalar = _to_scalar(value, name)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _to_scalar(value: Any, name: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError as exc:
            raise ValueError(f"{name} must be scalar") from exc
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    try:
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _one_dimensional_finite(values: ArrayLike, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


__all__ = [
    "ChronologicalSplit",
    "LikelihoodComparison",
    "chronological_split",
    "conditional_log_likelihood",
    "compare_log_likelihoods",
    "compare_models",
    "name_regimes",
    "summarize_regimes",
]
