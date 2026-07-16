"""Transparent regime-allocation backtests with next-row execution."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

ASSET_COLUMNS = ("equity", "bond")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Daily audit trail and one-row-per-strategy performance metrics."""

    daily: pd.DataFrame
    metrics: pd.DataFrame


def _coerce_allocation(
    allocation: float | Sequence[float] | Mapping[str, float],
) -> np.ndarray:
    if isinstance(allocation, Mapping):
        missing = set(ASSET_COLUMNS) - set(allocation)
        if missing:
            raise ValueError(f"allocation is missing weights for: {sorted(missing)}")
        values = np.array([allocation[column] for column in ASSET_COLUMNS], dtype=float)
    elif np.isscalar(allocation):
        equity_weight = float(allocation)
        values = np.array([equity_weight, 1.0 - equity_weight], dtype=float)
    else:
        values = np.asarray(allocation, dtype=float)
        if values.shape != (2,):
            raise ValueError("allocation sequences must contain equity and bond weights")
    if not np.isfinite(values).all():
        raise ValueError("allocation weights must be finite")
    if (values < 0).any() or (values > 1).any() or not np.isclose(values.sum(), 1.0):
        raise ValueError("allocation weights must lie in [0, 1] and sum to one")
    return values


def build_regime_weights(
    regime_signal: pd.Series | pd.DataFrame,
    allocations: Mapping[Hashable, float | Sequence[float] | Mapping[str, float]],
    *,
    confidence_threshold: float | None = None,
    fallback_weights: float | Sequence[float] | Mapping[str, float] = (0.6, 0.4),
) -> pd.DataFrame:
    """Convert regime labels or probabilities into equity/bond decisions.

    A probability frame produces probability-weighted allocations, which avoids
    unnecessary all-or-nothing switches. Low-confidence rows can optionally use
    ``fallback_weights``.
    """

    if not allocations:
        raise ValueError("allocations cannot be empty")
    allocation_values = {
        state: _coerce_allocation(allocation) for state, allocation in allocations.items()
    }
    fallback = _coerce_allocation(fallback_weights)

    if isinstance(regime_signal, pd.Series):
        unknown = set(regime_signal.dropna().unique()) - set(allocation_values)
        if unknown:
            raise ValueError(f"no allocation supplied for regimes: {sorted(unknown, key=str)}")
        rows = [
            allocation_values.get(state, np.array([np.nan, np.nan]))
            for state in regime_signal.to_numpy()
        ]
        weights = pd.DataFrame(rows, index=regime_signal.index, columns=ASSET_COLUMNS)
    elif isinstance(regime_signal, pd.DataFrame):
        missing = set(regime_signal.columns) - set(allocation_values)
        if missing:
            raise ValueError(f"no allocation supplied for regimes: {sorted(missing, key=str)}")
        probabilities = regime_signal.astype(float)
        values = probabilities.to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("regime probabilities must be finite and non-negative")
        row_sums = values.sum(axis=1)
        if (row_sums <= 0).any():
            raise ValueError("each probability row must have positive mass")
        probabilities = probabilities.div(row_sums, axis="index")
        allocation_matrix = np.vstack([allocation_values[state] for state in probabilities.columns])
        weights = pd.DataFrame(
            probabilities.to_numpy() @ allocation_matrix,
            index=probabilities.index,
            columns=ASSET_COLUMNS,
        )
        if confidence_threshold is not None:
            if not 0 <= confidence_threshold <= 1:
                raise ValueError("confidence_threshold must be between 0 and 1")
            low_confidence = probabilities.max(axis="columns") < confidence_threshold
            weights.loc[low_confidence, :] = fallback
    else:
        raise TypeError("regime_signal must be a Series of labels or DataFrame of probabilities")

    weights.index.name = regime_signal.index.name or "date"
    return weights


def apply_rebalance_frequency(
    decision_weights: pd.DataFrame,
    frequency: str = "daily",
) -> pd.DataFrame:
    """Keep decisions only on daily, weekly, or monthly review dates.

    Non-review rows are left missing. :func:`backtest_target_weights` carries
    the last target forward without trading, allowing actual holdings to drift.
    """

    if frequency not in {"daily", "weekly", "monthly"}:
        raise ValueError("frequency must be daily, weekly, or monthly")
    if decision_weights.empty:
        raise ValueError("decision_weights cannot be empty")
    if not isinstance(decision_weights.index, pd.DatetimeIndex):
        raise ValueError("decision_weights must have a DatetimeIndex")
    if not decision_weights.index.is_monotonic_increasing:
        raise ValueError("decision_weights must be sorted chronologically")
    if frequency == "daily":
        return decision_weights.copy()

    period_frequency = "W-FRI" if frequency == "weekly" else "M"
    periods = decision_weights.index.to_period(period_frequency)
    is_period_end = np.r_[periods[:-1] != periods[1:], True]
    # The initial decision is actionable even before the first calendar period
    # ends; later decisions are updated only at period ends.
    is_period_end[0] = True
    review_dates = pd.Series(is_period_end, index=decision_weights.index)
    return decision_weights.where(review_dates, axis="index")


def _validate_returns(asset_returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(asset_returns, pd.DataFrame) or asset_returns.empty:
        raise ValueError("asset_returns must be a non-empty DataFrame")
    missing = set(ASSET_COLUMNS) - set(asset_returns.columns)
    if missing:
        raise ValueError(f"asset_returns is missing columns: {sorted(missing)}")
    if asset_returns.index.duplicated().any() or not asset_returns.index.is_monotonic_increasing:
        raise ValueError("asset_returns index must be unique and sorted chronologically")
    returns = asset_returns.loc[:, ASSET_COLUMNS].astype(float)
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("asset_returns must be finite; clean and align returns before backtesting")
    if (returns <= -1).any().any():
        raise ValueError("asset returns must be greater than -100%")
    return returns


def _validate_decision_weights(decision_weights: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(decision_weights, pd.DataFrame) or decision_weights.empty:
        raise ValueError("decision_weights must be a non-empty DataFrame")
    missing = set(ASSET_COLUMNS) - set(decision_weights.columns)
    if missing:
        raise ValueError(f"decision_weights is missing columns: {sorted(missing)}")
    if (
        decision_weights.index.duplicated().any()
        or not decision_weights.index.is_monotonic_increasing
    ):
        raise ValueError("decision_weights index must be unique and sorted chronologically")
    weights = decision_weights.loc[:, ASSET_COLUMNS].astype(float)
    partial_rows = weights.notna().any(axis="columns") & ~weights.notna().all(axis="columns")
    if partial_rows.any():
        raise ValueError("decision-weight rows must be either complete or entirely missing")
    complete = weights.dropna()
    if complete.empty:
        raise ValueError("decision_weights contains no complete rows")
    values = complete.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("decision weights must be finite")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("decision weights must lie between zero and one")
    if not np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("each decision-weight row must sum to one")
    return weights


def _simulate(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    rebalance: pd.Series,
    *,
    transaction_cost_rate: float,
    charge_initial_trade: bool,
) -> pd.DataFrame:
    gross_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    net_returns: list[float] = []
    equity_weights: list[float] = []
    bond_weights: list[float] = []
    pre_trade_weights: np.ndarray | None = None

    for position in range(len(returns)):
        asset_return = returns.iloc[position].to_numpy(dtype=float)
        target = target_weights.iloc[position].to_numpy(dtype=float)
        if pre_trade_weights is None:
            portfolio_weights = target
            turnover = float(np.abs(target).sum()) if charge_initial_trade else 0.0
        elif bool(rebalance.iloc[position]):
            portfolio_weights = target
            turnover = float(np.abs(target - pre_trade_weights).sum())
        else:
            portfolio_weights = pre_trade_weights
            turnover = 0.0

        gross_return = float(portfolio_weights @ asset_return)
        cost = turnover * transaction_cost_rate
        net_return = (1.0 - cost) * (1.0 + gross_return) - 1.0
        denominator = 1.0 + gross_return
        pre_trade_weights = portfolio_weights * (1.0 + asset_return) / denominator

        gross_returns.append(gross_return)
        turnovers.append(turnover)
        costs.append(cost)
        net_returns.append(net_return)
        equity_weights.append(float(portfolio_weights[0]))
        bond_weights.append(float(portfolio_weights[1]))

    return pd.DataFrame(
        {
            "gross_return": gross_returns,
            "turnover": turnovers,
            "cost": costs,
            "return": net_returns,
            "equity_weight": equity_weights,
            "bond_weight": bond_weights,
        },
        index=returns.index,
    )


def performance_metrics(
    daily_returns: pd.Series,
    *,
    annualization: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> pd.Series:
    """Compute common return and risk statistics from net daily returns."""

    if annualization <= 0:
        raise ValueError("annualization must be positive")
    if annual_risk_free_rate <= -1:
        raise ValueError("annual_risk_free_rate must be greater than -100%")
    returns = pd.Series(daily_returns, copy=True).dropna().astype(float)
    if returns.empty or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("daily_returns must contain finite observations")
    if (returns <= -1).any():
        raise ValueError("daily_returns must be greater than -100%")

    growth = (1.0 + returns).cumprod()
    total_growth = float(growth.iloc[-1])
    total_return = total_growth - 1.0
    annualized_return = total_growth ** (annualization / len(returns)) - 1.0
    daily_volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    annualized_volatility = daily_volatility * np.sqrt(annualization)
    daily_risk_free_rate = (1.0 + annual_risk_free_rate) ** (1.0 / annualization) - 1.0
    sharpe_ratio = (
        float((returns.mean() - daily_risk_free_rate) / daily_volatility * np.sqrt(annualization))
        if daily_volatility > 0
        else np.nan
    )
    wealth_with_origin = np.r_[1.0, growth.to_numpy()]
    drawdowns = wealth_with_origin / np.maximum.accumulate(wealth_with_origin) - 1.0
    max_drawdown = float(-drawdowns.min())
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else np.nan

    return pd.Series(
        {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "observations": float(len(returns)),
        }
    )


def backtest_target_weights(
    asset_returns: pd.DataFrame,
    decision_weights: pd.DataFrame,
    *,
    static_weights: float | Sequence[float] | Mapping[str, float] = (0.6, 0.4),
    transaction_cost_bps: float = 5.0,
    execution_lag: int = 1,
    charge_initial_trade: bool = False,
    annualization: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Compare adaptive decisions with a static 60/40-style allocation.

    A decision stamped at close ``t`` is shifted by ``execution_lag`` and can
    only earn returns from ``t+1`` (with the default lag of one), never returns
    already realized at ``t``. Transaction costs use two-way traded notional and
    account for weights drifting with the previous day's asset returns. Both
    portfolios trade on the same supplied review schedule for a fair comparison.
    """

    if isinstance(execution_lag, bool) or not isinstance(execution_lag, int):
        raise ValueError("execution_lag must be an integer")
    if execution_lag < 1:
        raise ValueError("execution_lag must be at least one to prevent look-ahead bias")
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and non-negative")

    returns = _validate_returns(asset_returns)
    decisions = _validate_decision_weights(decision_weights)
    decisions = decisions.reindex(returns.index)
    decision_updates = decisions.notna().all(axis="columns")
    effective_weights = decisions.ffill().shift(execution_lag)
    effective_rebalance = decision_updates.shift(execution_lag, fill_value=False)
    usable = effective_weights.notna().all(axis=1)
    returns = returns.loc[usable]
    effective_weights = effective_weights.loc[usable]
    effective_rebalance = effective_rebalance.loc[usable]
    if returns.empty:
        raise ValueError("no returns remain after applying the execution lag")

    static = _coerce_allocation(static_weights)
    static_frame = pd.DataFrame(
        np.tile(static, (len(returns), 1)),
        index=returns.index,
        columns=ASSET_COLUMNS,
    )
    cost_rate = transaction_cost_bps / 10_000.0
    adaptive = _simulate(
        returns,
        effective_weights,
        effective_rebalance,
        transaction_cost_rate=cost_rate,
        charge_initial_trade=charge_initial_trade,
    )
    benchmark = _simulate(
        returns,
        static_frame,
        effective_rebalance,
        transaction_cost_rate=cost_rate,
        charge_initial_trade=charge_initial_trade,
    )

    daily = pd.DataFrame(index=returns.index)
    for column in ("gross_return", "turnover", "cost", "return"):
        daily[f"adaptive_{column}"] = adaptive[column]
        daily[f"static_60_40_{column}"] = benchmark[column]
    daily["adaptive_equity_weight"] = adaptive["equity_weight"]
    daily["adaptive_bond_weight"] = adaptive["bond_weight"]
    daily["static_60_40_equity_weight"] = benchmark["equity_weight"]
    daily["static_60_40_bond_weight"] = benchmark["bond_weight"]
    daily["adaptive_wealth"] = (1.0 + daily["adaptive_return"]).cumprod()
    daily["static_60_40_wealth"] = (1.0 + daily["static_60_40_return"]).cumprod()
    adaptive_peak = daily["adaptive_wealth"].cummax().clip(lower=1.0)
    benchmark_peak = daily["static_60_40_wealth"].cummax().clip(lower=1.0)
    daily["adaptive_drawdown"] = daily["adaptive_wealth"] / adaptive_peak - 1.0
    daily["static_60_40_drawdown"] = daily["static_60_40_wealth"] / benchmark_peak - 1.0
    daily.index.name = returns.index.name or "date"

    metrics = pd.DataFrame(
        {
            "adaptive": performance_metrics(
                daily["adaptive_return"],
                annualization=annualization,
                annual_risk_free_rate=annual_risk_free_rate,
            ),
            "static_60_40": performance_metrics(
                daily["static_60_40_return"],
                annualization=annualization,
                annual_risk_free_rate=annual_risk_free_rate,
            ),
        }
    ).T
    metrics.index.name = "strategy"
    return BacktestResult(daily=daily, metrics=metrics)


def run_regime_backtest(
    allocation_returns: pd.DataFrame,
    regime_signal: pd.Series | pd.DataFrame,
    allocations: Mapping[Hashable, float | Sequence[float] | Mapping[str, float]],
    *,
    static_weights: float | Sequence[float] | Mapping[str, float] = (0.6, 0.4),
    transaction_cost_bps: float = 5.0,
    confidence_threshold: float | None = None,
    fallback_weights: float | Sequence[float] | Mapping[str, float] = (0.6, 0.4),
    rebalance_frequency: str = "daily",
    execution_lag: int = 1,
    annualization: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Convenience wrapper for feature-module returns and regime outputs."""

    required = {"equity_return", "bond_return"}
    missing = required - set(allocation_returns.columns)
    if missing:
        raise ValueError(f"allocation_returns is missing columns: {sorted(missing)}")
    asset_returns = allocation_returns.loc[:, ["equity_return", "bond_return"]].rename(
        columns={"equity_return": "equity", "bond_return": "bond"}
    )
    decisions = build_regime_weights(
        regime_signal,
        allocations,
        confidence_threshold=confidence_threshold,
        fallback_weights=fallback_weights,
    )
    decisions = apply_rebalance_frequency(decisions, rebalance_frequency)
    return backtest_target_weights(
        asset_returns,
        decisions,
        static_weights=static_weights,
        transaction_cost_bps=transaction_cost_bps,
        execution_lag=execution_lag,
        annualization=annualization,
        annual_risk_free_rate=annual_risk_free_rate,
    )
