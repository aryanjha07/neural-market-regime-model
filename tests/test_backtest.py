from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime.backtest import (
    apply_rebalance_frequency,
    backtest_target_weights,
    build_regime_weights,
    performance_metrics,
    run_regime_backtest,
)


def test_close_t_decision_only_earns_t_plus_one_return() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4, name="date")
    returns = pd.DataFrame(
        {"equity": [0.10, 0.20, 0.30, 0.40], "bond": [0.0, 0.0, 0.0, 0.0]},
        index=dates,
    )
    decisions = pd.DataFrame(
        {"equity": [0.0, 1.0, 0.0, 1.0], "bond": [1.0, 0.0, 1.0, 0.0]},
        index=dates,
    )

    result = backtest_target_weights(returns, decisions, transaction_cost_bps=0)

    assert result.daily.index.tolist() == dates[1:].tolist()
    assert result.daily["adaptive_equity_weight"].tolist() == [0.0, 1.0, 0.0]
    assert result.daily["adaptive_return"].tolist() == pytest.approx([0.0, 0.30, 0.0])
    # In particular, the equity decision formed on the second date cannot earn
    # that date's already-realized 20% equity return.
    assert result.daily.loc[dates[1], "adaptive_return"] == pytest.approx(0.0)


def test_zero_lag_is_rejected_as_lookahead() -> None:
    dates = pd.bdate_range("2024-01-01", periods=2)
    returns = pd.DataFrame({"equity": [0.0, 0.0], "bond": [0.0, 0.0]}, index=dates)
    weights = pd.DataFrame({"equity": [0.6, 0.6], "bond": [0.4, 0.4]}, index=dates)

    with pytest.raises(ValueError, match="look-ahead"):
        backtest_target_weights(returns, weights, execution_lag=0)


def test_transaction_costs_charge_traded_notional() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4)
    returns = pd.DataFrame({"equity": np.zeros(4), "bond": np.zeros(4)}, index=dates)
    decisions = pd.DataFrame(
        {"equity": [1.0, 0.0, 1.0, 0.0], "bond": [0.0, 1.0, 0.0, 1.0]},
        index=dates,
    )

    result = backtest_target_weights(returns, decisions, transaction_cost_bps=100)

    assert result.daily["adaptive_turnover"].tolist() == pytest.approx([0.0, 2.0, 2.0])
    assert result.daily["adaptive_cost"].tolist() == pytest.approx([0.0, 0.02, 0.02])
    assert result.daily["adaptive_return"].tolist() == pytest.approx([0.0, -0.02, -0.02])


def test_probability_weighted_regime_allocations() -> None:
    dates = pd.bdate_range("2024-01-01", periods=2)
    probabilities = pd.DataFrame({0: [0.25, 0.9], 1: [0.75, 0.1]}, index=dates)

    weights = build_regime_weights(probabilities, {0: 0.8, 1: 0.2})

    assert weights.iloc[0]["equity"] == pytest.approx(0.35)
    assert weights.iloc[0]["bond"] == pytest.approx(0.65)
    assert weights.sum(axis=1).to_numpy() == pytest.approx(np.ones(2))


def test_weekly_rebalancing_holds_weights_between_review_dates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    decisions = pd.DataFrame(
        {
            "equity": np.arange(8, dtype=float) / 10,
            "bond": 1 - np.arange(8, dtype=float) / 10,
        },
        index=dates,
    )

    held = apply_rebalance_frequency(decisions, "weekly")

    assert held["equity"].dropna().tolist() == pytest.approx([0.0, 0.4, 0.7])
    assert held["equity"].isna().tolist() == [False, True, True, True, False, True, True, False]


def test_sparse_review_schedule_allows_weights_to_drift_without_trading() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4)
    returns = pd.DataFrame(
        {"equity": [0.0, 0.10, 0.10, 0.0], "bond": [0.0, 0.0, 0.0, 0.0]},
        index=dates,
    )
    decisions = pd.DataFrame(
        {"equity": [0.5, np.nan, np.nan, 0.5], "bond": [0.5, np.nan, np.nan, 0.5]},
        index=dates,
    )

    result = backtest_target_weights(returns, decisions, transaction_cost_bps=100)

    assert result.daily["adaptive_turnover"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert result.daily["adaptive_equity_weight"].iloc[1] > 0.5


def test_performance_metrics_reports_peak_to_trough_loss() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])

    metrics = performance_metrics(returns)

    assert metrics["total_return"] == pytest.approx(1.1 * 0.8 * 1.05 - 1)
    assert metrics["max_drawdown"] == pytest.approx(0.20)
    assert metrics["observations"] == 3


def test_regime_backtest_accepts_feature_module_return_names() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4)
    returns = pd.DataFrame(
        {"equity_return": [0.01, 0.02, -0.01, 0.03], "bond_return": [0.0] * 4},
        index=dates,
    )
    states = pd.Series(["calm", "crisis", "calm", "calm"], index=dates)

    result = run_regime_backtest(
        returns,
        states,
        {"calm": 0.8, "crisis": 0.2},
        transaction_cost_bps=0,
    )

    assert result.daily["adaptive_equity_weight"].tolist() == pytest.approx([0.8, 0.2, 0.8])
    assert set(result.metrics.index) == {"adaptive", "static_60_40"}
