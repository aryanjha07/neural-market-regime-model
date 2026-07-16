"""Small plotting helpers for saved experiment artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REGIME_COLORS = {
    "calm": "#2A9D8F",
    "neutral": "#E9C46A",
    "crisis": "#E76F51",
}


def plot_regimes(
    equity_returns: pd.Series,
    regimes: pd.Series,
    labels: Mapping[int, str],
    output_path: str | Path,
    *,
    events: Mapping[str, str | pd.Timestamp] | None = None,
) -> Path:
    """Plot a synthetic equity curve with background colors by regime."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    equity_curve = (1.0 + equity_returns.fillna(0.0)).cumprod()

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(equity_curve.index, equity_curve, color="#17324D", linewidth=1.4)
    regime_values = regimes.reindex(equity_curve.index).ffill()
    for state, label in labels.items():
        mask = regime_values.eq(state)
        axis.fill_between(
            equity_curve.index,
            equity_curve.min(),
            equity_curve.max(),
            where=mask,
            color=REGIME_COLORS.get(label, "#A8A8A8"),
            alpha=0.16,
            label=label.title(),
        )
    visible_events = 0
    for event_label, event_date in (events or {}).items():
        timestamp = pd.Timestamp(event_date)
        if equity_curve.index.min() <= timestamp <= equity_curve.index.max():
            axis.axvline(
                timestamp,
                color="#5C677D",
                linestyle="--",
                linewidth=1.2,
                label=event_label,
            )
            visible_events += 1
    axis.set_title("Equity Curve and Causally Detected Market Regimes")
    axis.set_ylabel("Growth of $1")
    axis.legend(loc="upper left", ncols=min(3, len(labels) + visible_events))
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_backtest(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    output_path: str | Path,
) -> Path:
    """Save adaptive and static benchmark equity curves."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    strategy = (1.0 + strategy_returns.fillna(0.0)).cumprod()
    benchmark = (1.0 + benchmark_returns.fillna(0.0)).cumprod()

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(strategy.index, strategy, label="Regime adaptive", color="#2A9D8F")
    axis.plot(benchmark.index, benchmark, label="Static 60/40", color="#264653")
    axis.set_title("Out-of-Sample Portfolio Comparison")
    axis.set_ylabel("Growth of $1")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
