"""Synthetic regimes used for smoke tests and demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SyntheticMarket:
    features: pd.DataFrame
    returns: pd.DataFrame
    states: pd.Series


def generate_synthetic_market(
    n_days: int = 1_500,
    seed: int = 42,
    start: str = "2015-01-02",
) -> SyntheticMarket:
    """Generate persistent calm, trending, and crisis market states.

    The crisis state's equity return is a two-component mixture so the neural
    emission model has a genuinely non-Gaussian pattern to learn.
    """

    if n_days < 100:
        raise ValueError("n_days must be at least 100")

    rng = np.random.default_rng(seed)
    transition = np.array(
        [
            [0.965, 0.030, 0.005],
            [0.035, 0.940, 0.025],
            [0.045, 0.055, 0.900],
        ]
    )
    states = np.empty(n_days, dtype=np.int64)
    states[0] = 0
    for day in range(1, n_days):
        states[day] = rng.choice(3, p=transition[states[day - 1]])

    equity_return = np.empty(n_days)
    volatility = np.empty(n_days)
    volume_z = np.empty(n_days)
    vix_change = np.empty(n_days)
    momentum = np.empty(n_days)

    for day, state in enumerate(states):
        if state == 0:  # calm
            equity_return[day] = rng.normal(0.00045, 0.006)
            volatility[day] = abs(rng.normal(0.008, 0.0015))
            volume_z[day] = rng.normal(-0.15, 0.65)
            vix_change[day] = rng.normal(-0.001, 0.025)
            momentum[day] = rng.normal(0.008, 0.015)
        elif state == 1:  # directional / neutral-risk
            equity_return[day] = rng.normal(0.0001, 0.012)
            volatility[day] = abs(rng.normal(0.015, 0.003))
            volume_z[day] = rng.normal(0.25, 0.85)
            vix_change[day] = rng.normal(0.002, 0.045)
            momentum[day] = rng.normal(0.0, 0.035)
        else:  # crisis with a pronounced left tail and occasional rebound
            if rng.random() < 0.78:
                equity_return[day] = rng.normal(-0.004, 0.021)
            else:
                equity_return[day] = rng.normal(0.010, 0.032)
            volatility[day] = abs(rng.normal(0.035, 0.008))
            volume_z[day] = rng.normal(1.5, 1.0)
            vix_change[day] = rng.normal(0.015, 0.09)
            momentum[day] = rng.normal(-0.045, 0.055)

    bond_return = rng.normal(0.00012, 0.0035, n_days) - 0.10 * equity_return
    cash_return = np.full(n_days, 0.02 / 252)
    index = pd.bdate_range(start=start, periods=n_days, name="date")

    feature_frame = pd.DataFrame(
        {
            "equity_return": equity_return,
            "realized_volatility": volatility,
            "volume_zscore": volume_z,
            "vix_change": vix_change,
            "momentum": momentum,
        },
        index=index,
    )
    return_frame = pd.DataFrame(
        {
            "equity_return": equity_return,
            "bond_return": bond_return,
            "cash_return": cash_return,
        },
        index=index,
    )
    state_series = pd.Series(states, index=index, name="true_state")
    return SyntheticMarket(feature_frame, return_frame, state_series)
