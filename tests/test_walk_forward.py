from __future__ import annotations

import json

import numpy as np
import pandas as pd

from market_regime.config import (
    BacktestConfig,
    ExperimentConfig,
    ModelConfig,
    WalkForwardConfig,
)
from market_regime.features import MarketDataset
from market_regime.synthetic import generate_synthetic_market
from market_regime.walk_forward import run_walk_forward_evaluation


def test_walk_forward_writes_auditable_non_overlapping_results(tmp_path) -> None:
    market = generate_synthetic_market(n_days=230, seed=31)
    config = ExperimentConfig(
        seed=31,
        walk_forward=WalkForwardConfig(
            initial_train_size=100,
            validation_size=30,
            test_size=40,
            max_folds=2,
        ),
        model=ModelConfig(
            n_states=3,
            n_mixtures=2,
            n_restarts=1,
            epochs=1,
            emission_steps=1,
            learning_rate=0.01,
        ),
        backtest=BacktestConfig(rebalance_frequency="daily", execution_lag=2),
    )

    result = run_walk_forward_evaluation(
        MarketDataset(market.features, market.returns),
        config,
        tmp_path,
    )

    assert len(result.folds) == 2
    assert result.folds["train_observations"].tolist() == [100, 140]
    assert result.folds["refit_observations"].tolist() == [130, 170]
    assert result.folds["test_observations"].sum() == 80
    assert set(result.folds["winner"]).issubset({"gaussian", "mixture", "neural"})
    assert np.isfinite(result.likelihood_summary["log_likelihood_per_observation"].to_numpy()).all()
    assert result.likelihood_summary["observations"].eq(80).all()

    assert len(result.probabilities) == 80 * 3 * 3
    probability_sums = result.probabilities.groupby(["model", "date"])["filtered_probability"].sum()
    np.testing.assert_allclose(
        probability_sums.to_numpy(),
        1.0,
    )
    assert not result.decisions.duplicated(["model", "date"]).any()
    assert len(result.decisions) == 80 * 3
    for backtest in result.backtests.values():
        assert len(backtest.daily) == 78
        assert np.isfinite(backtest.daily.to_numpy()).all()

    expected_files = {
        "walk_forward_report.json",
        "folds.csv",
        "likelihood_summary.csv",
        "probabilities.csv",
        "decision_weights.csv",
        "backtest_daily.csv",
        "backtest_metrics.csv",
        "likelihood_by_fold.png",
        "backtest.png",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})

    report = json.loads((tmp_path / "walk_forward_report.json").read_text(encoding="utf-8"))
    assert report["fold_count"] == 2
    assert report["out_of_sample"]["observations"] == 80
    assert set(report["aggregate_likelihood"]) == {"gaussian", "mixture", "neural"}


def test_walk_forward_first_fold_is_unchanged_by_later_future_data(tmp_path) -> None:
    market = generate_synthetic_market(n_days=170, seed=37)
    config = ExperimentConfig(
        seed=37,
        walk_forward=WalkForwardConfig(
            initial_train_size=80,
            validation_size=20,
            test_size=30,
            max_folds=1,
        ),
        model=ModelConfig(
            n_states=3,
            n_mixtures=2,
            n_restarts=1,
            epochs=1,
            emission_steps=1,
        ),
        backtest=BacktestConfig(rebalance_frequency="daily", execution_lag=2),
    )
    original = run_walk_forward_evaluation(
        MarketDataset(market.features, market.returns),
        config,
        tmp_path / "original",
    )

    changed_features = market.features.copy()
    changed_features.iloc[130:] = changed_features.iloc[130:] + 1_000.0
    changed_returns = market.returns.copy()
    changed_returns.iloc[130:] = 0.0
    changed = run_walk_forward_evaluation(
        MarketDataset(changed_features, changed_returns),
        config,
        tmp_path / "changed",
    )

    pd.testing.assert_frame_equal(original.folds, changed.folds)
    pd.testing.assert_frame_equal(original.likelihood_summary, changed.likelihood_summary)
    pd.testing.assert_frame_equal(original.probabilities, changed.probabilities)
    pd.testing.assert_frame_equal(original.decisions, changed.decisions)
