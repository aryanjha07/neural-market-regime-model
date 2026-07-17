"""Expanding-window evaluation across consecutive unseen market periods."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from market_regime.backtest import (
    BacktestResult,
    apply_rebalance_frequency,
    backtest_target_weights,
    build_regime_weights,
)
from market_regime.config import ExperimentConfig
from market_regime.evaluation import (
    conditional_log_likelihood,
    expanding_window_splits,
    summarize_regimes,
)
from market_regime.features import (
    FEATURE_COLUMNS,
    MarketDataset,
    fit_feature_scaler,
    transform_features,
)
from market_regime.training import (
    MODEL_NAMES,
    fit_regime_model,
    select_models_by_validation,
)


@dataclass(slots=True)
class WalkForwardResult:
    """Auditable fold-level scores and one continuous out-of-sample backtest."""

    output_dir: Path
    folds: pd.DataFrame
    likelihood_summary: pd.DataFrame
    probabilities: pd.DataFrame
    decisions: pd.DataFrame
    backtests: dict[str, BacktestResult]


def _filter_probabilities(name: str, model: Any, observations: np.ndarray) -> np.ndarray:
    if name == "neural":
        return model.filter(observations).cpu().numpy()
    return np.asarray(model.filter_proba(observations), dtype=np.float64)


def _transition_matrix(name: str, model: Any) -> np.ndarray:
    if name == "neural":
        return model.transition_matrix_.cpu().numpy()
    return np.asarray(model.transition_matrix_, dtype=np.float64)


def _state_labels(
    probabilities: np.ndarray,
    features: pd.DataFrame,
    n_states: int,
) -> dict[int, str]:
    summary = summarize_regimes(
        probabilities.argmax(axis=1),
        features["equity_return"].to_numpy(),
        features["realized_volatility"].to_numpy(),
    )
    labels = {
        int(row.state): str(row.regime)
        for row in summary.loc[:, ["state", "regime"]].itertuples(index=False)
    }
    for state in range(n_states):
        labels.setdefault(state, "Trending")
    return labels


def _state_allocations(labels: dict[int, str], config: ExperimentConfig) -> dict[int, float]:
    weights: dict[int, float] = {}
    for state, label in labels.items():
        if label.lower() == "calm":
            weights[state] = config.backtest.calm_equity_weight
        elif label.lower() == "crisis":
            weights[state] = config.backtest.crisis_equity_weight
        else:
            weights[state] = config.backtest.neutral_equity_weight
    return weights


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
    result: WalkForwardResult,
    config: ExperimentConfig,
) -> None:
    from market_regime.plotting import plot_backtest, plot_walk_forward_likelihood

    output = result.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

    result.folds.to_csv(output / "folds.csv", index=False)
    result.likelihood_summary.to_csv(output / "likelihood_summary.csv")
    result.probabilities.to_csv(output / "probabilities.csv", index=False)
    result.decisions.to_csv(output / "decision_weights.csv", index=False)

    metrics = pd.concat(
        {name: backtest.metrics for name, backtest in result.backtests.items()},
        names=["model", "strategy"],
    )
    metrics.to_csv(output / "backtest_metrics.csv")
    daily_parts = []
    for name, backtest in result.backtests.items():
        daily = backtest.daily.reset_index()
        daily.insert(1, "model", name)
        daily_parts.append(daily)
    daily_frame = pd.concat(daily_parts, ignore_index=True)
    daily_frame.to_csv(output / "backtest_daily.csv", index=False)

    plot_walk_forward_likelihood(result.folds, output / "likelihood_by_fold.png")
    neural_daily = result.backtests["neural"].daily
    plot_backtest(
        neural_daily["adaptive_return"],
        neural_daily["static_60_40_return"],
        output / "backtest.png",
    )

    summary = {
        "method": (
            "Expanding training history, fixed trailing validation window, "
            "validation-selected restart, full pre-test refit, untouched test fold"
        ),
        "fold_count": len(result.folds),
        "out_of_sample": {
            "start": result.folds.iloc[0]["test_start"],
            "end": result.folds.iloc[-1]["test_end"],
            "observations": int(result.folds["test_observations"].sum()),
        },
        "settings": asdict(config.walk_forward),
        "aggregate_likelihood": result.likelihood_summary.to_dict(orient="index"),
        "backtest": {
            name: backtest.metrics.to_dict(orient="index")
            for name, backtest in result.backtests.items()
        },
    }
    with (output / "walk_forward_report.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2, allow_nan=False)


def run_walk_forward_evaluation(
    dataset: MarketDataset,
    config: ExperimentConfig,
    output_dir: str | Path = "artifacts/walk_forward",
    *,
    verbose: bool = False,
) -> WalkForwardResult:
    """Train, select, refit, and test all models across expanding folds."""

    config.validate()
    features = dataset.features.loc[:, FEATURE_COLUMNS].copy()
    returns = dataset.returns.loc[features.index].copy()
    if not features.index.equals(returns.index):
        raise ValueError("features and returns must have identical chronological indices")
    if features.index.duplicated().any() or not features.index.is_monotonic_increasing:
        raise ValueError("dataset index must be unique and sorted chronologically")

    fold_splits = expanding_window_splits(
        features,
        initial_train_size=config.walk_forward.initial_train_size,
        validation_size=config.walk_forward.validation_size,
        test_size=config.walk_forward.test_size,
        max_folds=config.walk_forward.max_folds,
    )
    fold_records: list[dict[str, Any]] = []
    probability_parts: list[pd.DataFrame] = []
    decision_parts: dict[str, list[pd.DataFrame]] = {name: [] for name in MODEL_NAMES}
    decision_audit_parts: list[pd.DataFrame] = []

    for fold in fold_splits:
        train_start, train_stop = fold.train_indices
        validation_start, validation_stop = fold.validation_indices
        test_start, test_stop = fold.test_indices

        selection_scaler = fit_feature_scaler(features.iloc[train_start:train_stop])
        selection_scaled = transform_features(
            features.iloc[:validation_stop], selection_scaler
        ).to_numpy(dtype=np.float64)
        selection = select_models_by_validation(
            selection_scaled[train_start:train_stop],
            selection_scaled,
            validation_start=validation_start,
            validation_stop=validation_stop,
            config=config,
            verbose=verbose,
            progress_prefix=f"fold={fold.fold} ",
        )

        final_scaler = fit_feature_scaler(features.iloc[:test_start])
        final_scaled = transform_features(features.iloc[:test_stop], final_scaler).to_numpy(
            dtype=np.float64
        )
        final_models = {
            name: fit_regime_model(
                name,
                final_scaled[:test_start],
                config,
                seed=selection.selected_seeds[name],
                verbose=verbose,
            )
            for name in MODEL_NAMES
        }

        fold_record: dict[str, Any] = {
            "fold": fold.fold,
            "train_start": features.index[train_start],
            "train_end": features.index[train_stop - 1],
            "validation_start": features.index[validation_start],
            "validation_end": features.index[validation_stop - 1],
            "refit_start": features.index[0],
            "refit_end": features.index[test_start - 1],
            "test_start": features.index[test_start],
            "test_end": features.index[test_stop - 1],
            "train_observations": train_stop - train_start,
            "validation_observations": validation_stop - validation_start,
            "refit_observations": test_start,
            "test_observations": test_stop - test_start,
        }
        model_test_scores: dict[str, float] = {}
        test_index = features.index[test_start:test_stop]
        for name in MODEL_NAMES:
            model = final_models[name]
            test_score = conditional_log_likelihood(
                model,
                final_scaled,
                start=test_start,
                stop=test_stop,
            )
            model_test_scores[name] = test_score
            fold_record[f"{name}_selected_seed"] = selection.selected_seeds[name]
            fold_record[f"{name}_validation_log_likelihood_per_observation"] = (
                selection.validation_scores[name] / (validation_stop - validation_start)
            )
            fold_record[f"{name}_test_log_likelihood"] = test_score
            fold_record[f"{name}_test_log_likelihood_per_observation"] = test_score / (
                test_stop - test_start
            )

            filtered = _filter_probabilities(name, model, final_scaled)
            transition = _transition_matrix(name, model)
            allocation_horizon = filtered @ np.linalg.matrix_power(
                transition, config.backtest.execution_lag
            )
            labels = _state_labels(
                filtered[:test_start],
                features.iloc[:test_start],
                config.model.n_states,
            )
            horizon_test = pd.DataFrame(
                allocation_horizon[test_start:test_stop],
                index=test_index,
                columns=range(config.model.n_states),
            )
            decisions = build_regime_weights(
                horizon_test,
                _state_allocations(labels, config),
                confidence_threshold=config.backtest.confidence_threshold,
                fallback_weights=config.backtest.neutral_equity_weight,
            )
            decision_parts[name].append(decisions)

            audit = decisions.reset_index()
            audit.insert(1, "fold", fold.fold)
            audit.insert(2, "model", name)
            decision_audit_parts.append(audit)
            for state in range(config.model.n_states):
                probability_parts.append(
                    pd.DataFrame(
                        {
                            "date": test_index,
                            "fold": fold.fold,
                            "model": name,
                            "state": state,
                            "regime": labels[state],
                            "filtered_probability": filtered[test_start:test_stop, state],
                            "allocation_horizon_probability": allocation_horizon[
                                test_start:test_stop, state
                            ],
                        }
                    )
                )

        fold_record["winner"] = max(model_test_scores, key=model_test_scores.get)
        fold_records.append(fold_record)
        if verbose:
            scores = " ".join(
                f"{name}={model_test_scores[name] / (test_stop - test_start):.6f}"
                for name in MODEL_NAMES
            )
            print(f"fold={fold.fold} test {scores} winner={fold_record['winner']}")

    folds = pd.DataFrame(fold_records)
    total_observations = int(folds["test_observations"].sum())
    likelihood_rows = []
    for name in MODEL_NAMES:
        total = float(folds[f"{name}_test_log_likelihood"].sum())
        likelihood_rows.append(
            {
                "model": name,
                "total_log_likelihood": total,
                "observations": total_observations,
                "log_likelihood_per_observation": total / total_observations,
                "fold_wins": int(folds["winner"].eq(name).sum()),
            }
        )
    likelihood_summary = pd.DataFrame(likelihood_rows).set_index("model")

    probabilities = pd.concat(probability_parts, ignore_index=True)
    decisions_audit = pd.concat(decision_audit_parts, ignore_index=True)
    asset_returns = returns.loc[
        pd.concat(decision_parts["neural"]).index,
        ["equity_return", "bond_return"],
    ].rename(columns={"equity_return": "equity", "bond_return": "bond"})
    backtests: dict[str, BacktestResult] = {}
    for name in MODEL_NAMES:
        decisions = pd.concat(decision_parts[name]).sort_index()
        if decisions.index.duplicated().any():
            raise RuntimeError("walk-forward test folds produced duplicate decision dates")
        scheduled = apply_rebalance_frequency(
            decisions,
            config.backtest.rebalance_frequency,
        )
        backtests[name] = backtest_target_weights(
            asset_returns,
            scheduled,
            static_weights=0.60,
            transaction_cost_bps=config.backtest.transaction_cost_bps,
            execution_lag=config.backtest.execution_lag,
        )

    result = WalkForwardResult(
        output_dir=Path(output_dir),
        folds=folds,
        likelihood_summary=likelihood_summary,
        probabilities=probabilities,
        decisions=decisions_audit,
        backtests=backtests,
    )
    _save_artifacts(result, config)
    return result


__all__ = ["WalkForwardResult", "run_walk_forward_evaluation"]
