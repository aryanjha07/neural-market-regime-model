"""Leakage-safe daily market features and chronological preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from market_regime.data import ticker_frame

FEATURE_COLUMNS = (
    "equity_return",
    "realized_volatility",
    "volume_zscore",
    "vix_change",
    "momentum",
)
RETURN_COLUMNS = ("equity_return", "bond_return")


@dataclass(frozen=True, slots=True)
class MarketDataset:
    """Aligned model features and unscaled asset returns."""

    features: pd.DataFrame
    returns: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _validate_window(window: int, name: str) -> None:
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError(f"{name} must be an integer of at least 2")


def _validate_market_index(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("market_data must be a non-empty DataFrame")
    clean = frame.copy()
    try:
        clean.index = pd.to_datetime(clean.index)
    except (TypeError, ValueError) as exc:
        raise ValueError("market_data must have a date-like index") from exc
    if clean.index.duplicated().any():
        raise ValueError("market_data contains duplicate dates")
    clean = clean.sort_index()
    clean.index.name = frame.index.name or "date"
    return clean


def _return_price(frame: pd.DataFrame, ticker: str) -> pd.Series:
    """Use dividend-adjusted prices when the data source supplies them."""

    column = "adj_close" if "adj_close" in frame.columns else "close"
    if column not in frame.columns:
        raise ValueError(f"{ticker} data must contain close or adj_close")
    prices = pd.to_numeric(frame[column], errors="coerce")
    if (prices.dropna() <= 0).any():
        raise ValueError(f"{ticker} return prices must be positive")
    return prices


def build_market_features(
    market_data: pd.DataFrame,
    *,
    equity_ticker: str = "SPY",
    vix_ticker: str = "^VIX",
    volatility_window: int = 20,
    volume_window: int = 20,
    momentum_window: int = 20,
    annualize_volatility: bool = False,
    trading_days: int = 252,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build end-of-day features using present and past observations only.

    ``volume_zscore`` compares today's volume with the *preceding* window. The
    rolling volatility includes today's already-observed return and never uses
    a centered window. ``equity_return`` is a log return for model stability.
    """

    _validate_window(volatility_window, "volatility_window")
    _validate_window(volume_window, "volume_window")
    _validate_window(momentum_window, "momentum_window")
    if trading_days <= 0:
        raise ValueError("trading_days must be positive")

    market_data = _validate_market_index(market_data)
    equity = ticker_frame(market_data, equity_ticker)
    vix = ticker_frame(market_data, vix_ticker)
    if "volume" not in equity:
        raise ValueError(f"{equity_ticker} data must contain volume")
    if "close" not in vix:
        raise ValueError(f"{vix_ticker} data must contain close")

    equity_close = _return_price(equity, equity_ticker)
    equity_volume = pd.to_numeric(equity["volume"], errors="coerce")
    vix_close = pd.to_numeric(vix["close"], errors="coerce")
    if (vix_close.dropna() <= 0).any():
        raise ValueError("VIX close prices must be positive before computing log returns")

    log_price = np.log(equity_close)
    equity_log_return = log_price.diff()
    realized_volatility = equity_log_return.rolling(
        volatility_window,
        min_periods=volatility_window,
    ).std(ddof=1)
    if annualize_volatility:
        realized_volatility = realized_volatility * np.sqrt(trading_days)

    prior_volume = equity_volume.shift(1)
    prior_volume_mean = prior_volume.rolling(
        volume_window,
        min_periods=volume_window,
    ).mean()
    prior_volume_std = prior_volume.rolling(
        volume_window,
        min_periods=volume_window,
    ).std(ddof=1)
    volume_zscore = (equity_volume - prior_volume_mean) / prior_volume_std.replace(0, np.nan)

    features = pd.DataFrame(
        {
            "equity_return": equity_log_return,
            "realized_volatility": realized_volatility,
            "volume_zscore": volume_zscore,
            "vix_change": np.log(vix_close).diff(),
            "momentum": log_price.diff(momentum_window),
        }
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    features.index.name = market_data.index.name or "date"
    if dropna:
        features = features.dropna(subset=list(FEATURE_COLUMNS))
    return features.loc[:, FEATURE_COLUMNS]


def build_features(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Concise alias for :func:`build_market_features`."""

    return build_market_features(*args, **kwargs)


def build_allocation_returns(
    market_data: pd.DataFrame,
    *,
    equity_ticker: str = "SPY",
    bond_ticker: str = "IEF",
    dropna: bool = True,
) -> pd.DataFrame:
    """Build unscaled adjusted close-to-close returns for the backtest."""

    market_data = _validate_market_index(market_data)
    equity = ticker_frame(market_data, equity_ticker)
    bond = ticker_frame(market_data, bond_ticker)
    equity_price = _return_price(equity, equity_ticker)
    bond_price = _return_price(bond, bond_ticker)

    returns = pd.DataFrame(
        {
            "equity_return": equity_price.pct_change(fill_method=None),
            "bond_return": bond_price.pct_change(fill_method=None),
        }
    ).replace([np.inf, -np.inf], np.nan)
    returns.index.name = market_data.index.name or "date"
    if dropna:
        returns = returns.dropna(subset=list(RETURN_COLUMNS))
    return returns.loc[:, RETURN_COLUMNS]


def prepare_market_dataset(
    market_data: pd.DataFrame,
    **feature_kwargs: Any,
) -> MarketDataset:
    """Build and align model features with raw allocation returns.

    ``bond_ticker`` is consumed here rather than passed to the feature builder.
    This separation ensures fitting a scaler never changes the returns used in
    the backtest.
    """

    bond_ticker = str(feature_kwargs.pop("bond_ticker", "IEF"))
    feature_kwargs.pop("dropna", None)
    equity_ticker = str(feature_kwargs.get("equity_ticker", "SPY"))
    features = build_market_features(market_data, dropna=True, **feature_kwargs)
    returns = build_allocation_returns(
        market_data,
        equity_ticker=equity_ticker,
        bond_ticker=bond_ticker,
        dropna=True,
    )
    common_index = features.index.intersection(returns.index)
    if common_index.empty:
        raise ValueError("features and allocation returns have no overlapping dates")
    return MarketDataset(features.loc[common_index].copy(), returns.loc[common_index].copy())


def clean_feature_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Sort chronologically and remove rows unsafe for model fitting."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"feature frame is missing columns: {missing}")
    clean = frame.copy()
    try:
        clean.index = pd.to_datetime(clean.index)
    except (TypeError, ValueError) as exc:
        raise ValueError("feature frame must have a date-like index") from exc
    if clean.index.duplicated().any():
        raise ValueError("feature frame contains duplicate dates")
    clean = clean.sort_index().replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=list(feature_columns))
    if clean.empty:
        raise ValueError("no complete feature rows remain after cleaning")
    return clean


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> ChronologicalSplit:
    """Split an already-clean frame by position without shuffling."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test set")
    if frame.empty:
        raise ValueError("cannot split an empty frame")
    if frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
        raise ValueError("frame index must be unique and sorted chronologically")

    n_rows = len(frame)
    train_stop = int(n_rows * train_fraction)
    validation_stop = train_stop + int(n_rows * validation_fraction)
    if train_stop < 1 or validation_stop >= n_rows:
        raise ValueError("split fractions produce an empty train or test set")
    if validation_fraction > 0 and validation_stop == train_stop:
        raise ValueError("validation_fraction produces an empty validation set")

    return ChronologicalSplit(
        train=frame.iloc[:train_stop].copy(),
        validation=frame.iloc[train_stop:validation_stop].copy(),
        test=frame.iloc[validation_stop:].copy(),
    )


def chronological_train_test_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = chronological_split(
        frame,
        train_fraction=train_fraction,
        validation_fraction=0.0,
    )
    return split.train, split.test


def fit_feature_scaler(
    train: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
) -> StandardScaler:
    """Fit a standard scaler on training rows only."""

    missing = sorted(set(feature_columns) - set(train.columns))
    if missing:
        raise ValueError(f"training frame is missing columns: {missing}")
    values = train.loc[:, feature_columns].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("training features must be non-empty and finite")
    return StandardScaler().fit(values)


def transform_features(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Apply a previously fitted scaler while preserving dates and columns."""

    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"feature frame is missing columns: {missing}")
    values = frame.loc[:, feature_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("features must be finite before scaling")
    transformed = frame.copy()
    transformed.loc[:, feature_columns] = scaler.transform(values)
    return transformed


def scale_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str] = FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit on train, then transform both train and unseen test rows."""

    scaler = fit_feature_scaler(train, feature_columns)
    return (
        transform_features(train, scaler, feature_columns),
        transform_features(test, scaler, feature_columns),
        scaler,
    )
