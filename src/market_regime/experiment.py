"""End-to-end, leakage-aware model comparison and allocation experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from market_regime.backtest import BacktestResult, run_regime_backtest
from market_regime.baseline import GaussianHMMBaseline, GaussianMixtureHMMBaseline
from market_regime.config import ExperimentConfig
from market_regime.evaluation import (
    LikelihoodComparison,
    chronological_split,
    compare_log_likelihoods,
    conditional_log_likelihood,
    summarize_regimes,
)
from market_regime.features import (
    FEATURE_COLUMNS,
    MarketDataset,
    fit_feature_scaler,
    transform_features,
)
from market_regime.neural_hmm import NeuralEmissionHMM


@dataclass(slots=True)
class ExperimentResult:
    """In-memory results from one train/validation/test experiment."""

    output_dir: Path
    likelihood: LikelihoodComparison
    mixture_likelihood: LikelihoodComparison
    validation_log_likelihoods: dict[str, float]
    neural_regimes: pd.DataFrame
    baseline_regimes: pd.DataFrame
    mixture_regimes: pd.DataFrame
    neural_backtest: BacktestResult
    baseline_backtest: BacktestResult
    latest_regime_probabilities: pd.Series
    latest_next_regime_probabilities: pd.Series


def _state_labels(
    summary: pd.DataFrame,
    n_states: int,
) -> dict[int, str]:
    labels = {
        int(row.state): str(row.regime)
        for row in summary.loc[:, ["state", "regime"]].itertuples(index=False)
    }
    for state in range(n_states):
        labels.setdefault(state, "Trending")
    return labels


def _state_allocations(
    labels: dict[int, str],
    config: ExperimentConfig,
) -> dict[int, float]:
    allocations: dict[int, float] = {}
    for state, label in labels.items():
        normalized = label.lower()
        if normalized == "calm":
            weight = config.backtest.calm_equity_weight
        elif normalized == "crisis":
            weight = config.backtest.crisis_equity_weight
        else:
            weight = config.backtest.neutral_equity_weight
        allocations[state] = weight
    return allocations


def _regime_summary(
    probabilities: np.ndarray,
    features: pd.DataFrame,
) -> pd.DataFrame:
    states = probabilities.argmax(axis=1)
    return summarize_regimes(
        states,
        features["equity_return"].to_numpy(),
        features["realized_volatility"].to_numpy(),
    )


def _probability_frame(
    values: np.ndarray,
    index: pd.Index,
    n_states: int,
) -> pd.DataFrame:
    return pd.DataFrame(values, index=index, columns=list(range(n_states)))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _save_artifacts(
    *,
    output_dir: Path,
    config: ExperimentConfig,
    baseline: GaussianHMMBaseline,
    mixture: GaussianMixtureHMMBaseline,
    neural: NeuralEmissionHMM,
    scaler: Any,
    likelihood: LikelihoodComparison,
    mixture_likelihood: LikelihoodComparison,
    validation_log_likelihoods: dict[str, float],
    baseline_summary: pd.DataFrame,
    mixture_summary: pd.DataFrame,
    neural_summary: pd.DataFrame,
    baseline_backtest: BacktestResult,
    neural_backtest: BacktestResult,
    baseline_filtered_probabilities: pd.DataFrame,
    neural_filtered_probabilities: pd.DataFrame,
    baseline_probabilities: pd.DataFrame,
    neural_probabilities: pd.DataFrame,
    baseline_allocation_probabilities: pd.DataFrame,
    neural_allocation_probabilities: pd.DataFrame,
    labels: dict[int, str],
    features: pd.DataFrame,
    returns: pd.DataFrame,
    split_dates: dict[str, Any],
) -> None:
    # Plotting is imported lazily so data-only CLI commands do not initialize
    # Matplotlib or build its font cache.
    from market_regime.plotting import plot_backtest, plot_regimes

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

    report = {
        "likelihood": likelihood.as_dict(),
        "mixture_ablation": mixture_likelihood.as_dict(),
        "validation_log_likelihood_per_observation": validation_log_likelihoods,
        "selected_seeds": {
            "gaussian": baseline.random_state,
            "mixture": mixture.random_state,
            "neural": neural.random_state,
        },
        "split_dates": split_dates,
        "latest": {
            "date": features.index[-1],
            "filtered_regime_probabilities": (neural_filtered_probabilities.iloc[-1].to_dict()),
            "next_session_regime_probabilities": (neural_probabilities.iloc[-1].to_dict()),
            "regime_labels": labels,
        },
        "backtest": {
            "signal_horizon_sessions": config.backtest.execution_lag,
            "neural": neural_backtest.metrics.to_dict(orient="index"),
            "gaussian_baseline": baseline_backtest.metrics.to_dict(orient="index"),
        },
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(report), handle, indent=2, allow_nan=False)

    baseline_summary.to_csv(output_dir / "baseline_regimes.csv", index=False)
    mixture_summary.to_csv(output_dir / "mixture_regimes.csv", index=False)
    neural_summary.to_csv(output_dir / "neural_regimes.csv", index=False)
    baseline_backtest.metrics.to_csv(output_dir / "baseline_backtest_metrics.csv")
    neural_backtest.metrics.to_csv(output_dir / "neural_backtest_metrics.csv")
    neural_backtest.daily.to_csv(output_dir / "neural_backtest_daily.csv")
    pd.concat(
        {
            "gaussian_filtered": baseline_filtered_probabilities,
            "gaussian_next": baseline_probabilities,
            "neural_filtered": neural_filtered_probabilities,
            "neural_next": neural_probabilities,
            "gaussian_allocation_horizon": baseline_allocation_probabilities,
            "neural_allocation_horizon": neural_allocation_probabilities,
        },
        axis="columns",
    ).to_csv(output_dir / "regime_probabilities.csv")
    pd.DataFrame(
        baseline.transition_matrix_,
        index=range(config.model.n_states),
        columns=range(config.model.n_states),
    ).to_csv(output_dir / "baseline_transition_matrix.csv")
    pd.DataFrame(
        mixture.transition_matrix_,
        index=range(config.model.n_states),
        columns=range(config.model.n_states),
    ).to_csv(output_dir / "mixture_transition_matrix.csv")
    pd.DataFrame(
        neural.transition_matrix_.cpu().numpy(),
        index=range(config.model.n_states),
        columns=range(config.model.n_states),
    ).to_csv(output_dir / "neural_transition_matrix.csv")

    joblib.dump(scaler, output_dir / "feature_scaler.joblib")
    joblib.dump(baseline, output_dir / "gaussian_hmm.joblib")
    joblib.dump(mixture, output_dir / "gaussian_mixture_hmm.joblib")
    neural.save_checkpoint(output_dir / "neural_hmm.pt")

    plot_start = pd.Timestamp(split_dates["validation"][0])
    plot_index = features.index[features.index >= plot_start]
    plot_states = neural_filtered_probabilities.loc[plot_index].to_numpy().argmax(axis=1)
    plot_regimes(
        returns.loc[plot_index, "equity_return"],
        pd.Series(plot_states, index=plot_index),
        {state: label.lower() for state, label in labels.items()},
        output_dir / "regimes.png",
        events={
            "COVID crash": "2020-02-19",
            "Fed hikes begin": "2022-03-16",
        },
    )
    plot_backtest(
        neural_backtest.daily["adaptive_return"],
        neural_backtest.daily["static_60_40_return"],
        output_dir / "backtest.png",
    )


def run_dataset_experiment(
    dataset: MarketDataset,
    config: ExperimentConfig,
    output_dir: str | Path,
    *,
    verbose: bool = False,
) -> ExperimentResult:
    """Train both HMMs and evaluate only on the untouched chronological test set."""

    config.validate()
    features = dataset.features.loc[:, FEATURE_COLUMNS].copy()
    returns = dataset.returns.loc[features.index].copy()
    if not features.index.equals(returns.index):
        raise ValueError("features and returns must have identical chronological indices")
    if not features.index.is_monotonic_increasing or features.index.duplicated().any():
        raise ValueError("dataset index must be unique and sorted chronologically")

    split = chronological_split(
        features,
        train_fraction=config.split.train_fraction,
        validation_fraction=config.split.validation_fraction,
    )
    scaler = fit_feature_scaler(split.train)
    scaled = transform_features(features, scaler)
    train_start, train_stop = split.train_indices
    test_start, test_stop = split.test_indices
    x_train = scaled.iloc[train_start:train_stop].to_numpy(dtype=np.float64)
    x_all = scaled.to_numpy(dtype=np.float64)

    validation_count = test_start - train_stop
    selected: dict[str, tuple[Any, float]] = {}
    for restart in range(config.model.n_restarts):
        seed = config.seed + restart
        candidates: dict[str, Any] = {
            "gaussian": GaussianHMMBaseline(
                n_states=config.model.n_states,
                covariance_type="full",
                min_covar=config.model.min_covar,
                random_state=seed,
            ).fit(x_train),
            "mixture": GaussianMixtureHMMBaseline(
                n_states=config.model.n_states,
                n_mixtures=config.model.n_mixtures,
                covariance_type="diag",
                min_covar=config.model.min_covar,
                random_state=seed,
            ).fit(x_train),
        }
        torch.manual_seed(seed)
        candidates["neural"] = NeuralEmissionHMM(
            n_states=config.model.n_states,
            n_features=x_train.shape[1],
            n_components=config.model.n_mixtures,
            hidden_dim=16,
            min_scale=config.model.min_scale,
            random_state=seed,
        ).fit(
            x_train,
            n_iter=config.model.epochs,
            learning_rate=config.model.learning_rate,
            emission_steps=config.model.emission_steps,
            verbose=verbose,
        )

        for name, candidate in candidates.items():
            validation_score = conditional_log_likelihood(
                candidate,
                x_all,
                start=train_stop,
                stop=test_start,
            )
            if verbose:
                print(
                    f"restart={restart + 1} model={name} "
                    f"validation_nats_per_day={validation_score / validation_count:.6f}"
                )
            if name not in selected or validation_score > selected[name][1]:
                selected[name] = (candidate, validation_score)

    baseline = selected["gaussian"][0]
    mixture = selected["mixture"][0]
    neural = selected["neural"][0]
    validation_log_likelihoods = {
        name: score / validation_count for name, (_, score) in selected.items()
    }

    baseline_test_score = conditional_log_likelihood(
        baseline,
        x_all,
        start=test_start,
        stop=test_stop,
    )
    neural_test_score = conditional_log_likelihood(
        neural,
        x_all,
        start=test_start,
        stop=test_stop,
    )
    mixture_test_score = conditional_log_likelihood(
        mixture,
        x_all,
        start=test_start,
        stop=test_stop,
    )
    likelihood = compare_log_likelihoods(
        baseline_test_score,
        neural_test_score,
        n_observations=test_stop - test_start,
    )
    mixture_likelihood = compare_log_likelihoods(
        mixture_test_score,
        neural_test_score,
        n_observations=test_stop - test_start,
    )

    baseline_filtered = np.asarray(baseline.filter_proba(x_all))
    neural_filtered = neural.filter(x_all).cpu().numpy()
    mixture_filtered = np.asarray(mixture.filter_proba(x_all))
    baseline_summary = _regime_summary(
        baseline_filtered[train_start:train_stop],
        features.iloc[train_start:train_stop],
    )
    neural_summary = _regime_summary(
        neural_filtered[train_start:train_stop],
        features.iloc[train_start:train_stop],
    )
    mixture_summary = _regime_summary(
        mixture_filtered[train_start:train_stop],
        features.iloc[train_start:train_stop],
    )
    baseline_labels = _state_labels(baseline_summary, config.model.n_states)
    neural_labels = _state_labels(neural_summary, config.model.n_states)

    baseline_next = baseline_filtered @ baseline.transition_matrix_
    neural_next = neural_filtered @ neural.transition_matrix_.cpu().numpy()
    baseline_allocation = baseline_filtered @ np.linalg.matrix_power(
        baseline.transition_matrix_, config.backtest.execution_lag
    )
    neural_allocation = neural_filtered @ np.linalg.matrix_power(
        neural.transition_matrix_.cpu().numpy(), config.backtest.execution_lag
    )
    baseline_filtered_probabilities = _probability_frame(
        baseline_filtered, features.index, config.model.n_states
    )
    neural_filtered_probabilities = _probability_frame(
        neural_filtered, features.index, config.model.n_states
    )
    baseline_probabilities = _probability_frame(
        baseline_next, features.index, config.model.n_states
    )
    neural_probabilities = _probability_frame(neural_next, features.index, config.model.n_states)
    baseline_allocation_probabilities = _probability_frame(
        baseline_allocation, features.index, config.model.n_states
    )
    neural_allocation_probabilities = _probability_frame(
        neural_allocation, features.index, config.model.n_states
    )
    test_index = features.index[test_start:test_stop]
    test_returns = returns.loc[test_index]

    common_backtest_options = {
        "transaction_cost_bps": config.backtest.transaction_cost_bps,
        "confidence_threshold": config.backtest.confidence_threshold,
        "fallback_weights": config.backtest.neutral_equity_weight,
        "rebalance_frequency": config.backtest.rebalance_frequency,
        "execution_lag": config.backtest.execution_lag,
    }
    baseline_backtest = run_regime_backtest(
        test_returns,
        baseline_allocation_probabilities.loc[test_index],
        _state_allocations(baseline_labels, config),
        **common_backtest_options,
    )
    neural_backtest = run_regime_backtest(
        test_returns,
        neural_allocation_probabilities.loc[test_index],
        _state_allocations(neural_labels, config),
        **common_backtest_options,
    )

    output = Path(output_dir)
    split_dates = {
        "train": [features.index[train_start], features.index[train_stop - 1]],
        "validation": [split.validation.index[0], split.validation.index[-1]],
        "test": [features.index[test_start], features.index[test_stop - 1]],
    }
    _save_artifacts(
        output_dir=output,
        config=config,
        baseline=baseline,
        mixture=mixture,
        neural=neural,
        scaler=scaler,
        likelihood=likelihood,
        mixture_likelihood=mixture_likelihood,
        validation_log_likelihoods=validation_log_likelihoods,
        baseline_summary=baseline_summary,
        mixture_summary=mixture_summary,
        neural_summary=neural_summary,
        baseline_backtest=baseline_backtest,
        neural_backtest=neural_backtest,
        baseline_filtered_probabilities=baseline_filtered_probabilities,
        neural_filtered_probabilities=neural_filtered_probabilities,
        baseline_probabilities=baseline_probabilities,
        neural_probabilities=neural_probabilities,
        baseline_allocation_probabilities=baseline_allocation_probabilities,
        neural_allocation_probabilities=neural_allocation_probabilities,
        labels=neural_labels,
        features=features,
        returns=returns,
        split_dates=split_dates,
    )

    latest_filtered = pd.Series(
        neural_filtered[-1],
        index=[neural_labels[state] for state in range(config.model.n_states)],
        name=features.index[-1],
    )
    latest_next = pd.Series(
        neural_next[-1],
        index=[neural_labels[state] for state in range(config.model.n_states)],
        name=features.index[-1],
    )
    return ExperimentResult(
        output_dir=output,
        likelihood=likelihood,
        mixture_likelihood=mixture_likelihood,
        validation_log_likelihoods=validation_log_likelihoods,
        neural_regimes=neural_summary,
        baseline_regimes=baseline_summary,
        mixture_regimes=mixture_summary,
        neural_backtest=neural_backtest,
        baseline_backtest=baseline_backtest,
        latest_regime_probabilities=latest_filtered,
        latest_next_regime_probabilities=latest_next,
    )


__all__ = ["ExperimentResult", "run_dataset_experiment"]
