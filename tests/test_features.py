from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from market_regime.data import _default_download_end, download_ticker_data
from market_regime.features import (
    FEATURE_COLUMNS,
    build_allocation_returns,
    build_market_features,
    chronological_split,
    scale_train_test,
)


def _market_data(periods: int = 10) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=periods, name="date")
    log_returns = np.array([0.0, 0.01, -0.02, 0.03, -0.01, 0.015, -0.005, 0.02, -0.01, 0.005])
    spy_close = 100.0 * np.exp(np.cumsum(log_returns[:periods]))
    frames = {
        "SPY": pd.DataFrame(
            {
                "close": spy_close,
                "volume": np.array([100, 120, 90, 150, 130, 180, 160, 140, 200, 170])[:periods],
            },
            index=index,
        ),
        "IEF": pd.DataFrame(
            {
                "close": np.linspace(95.0, 97.0, periods),
                "volume": np.full(periods, 50.0),
            },
            index=index,
        ),
        "^VIX": pd.DataFrame(
            {
                "close": np.array([15, 16, 14, 18, 17, 20, 19, 21, 18, 17])[:periods],
                "volume": np.zeros(periods),
            },
            index=index,
        ),
    }
    return pd.concat(frames, axis="columns", names=["ticker", "field"])


def test_ticker_download_uses_validated_csv_cache(tmp_path: Path) -> None:
    index = pd.bdate_range("2024-01-02", periods=3, name="Date")
    downloaded = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Adj Close": [100.5, 101.5, 102.5],
            "Volume": [1_000, 1_100, 1_200],
        },
        index=index,
    )
    calls = 0

    def fake_download(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return downloaded

    first = download_ticker_data(
        "SPY",
        "2024-01-01",
        "2024-01-10",
        cache_dir=tmp_path,
        downloader=fake_download,
    )

    def network_must_not_run(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("cache hit attempted a network download")

    second = download_ticker_data(
        "SPY",
        "2024-01-01",
        "2024-01-10",
        cache_dir=tmp_path,
        downloader=network_must_not_run,
    )

    assert calls == 1
    pd.testing.assert_frame_equal(first, second, check_freq=False)

    subset = download_ticker_data(
        "SPY",
        "2024-01-01",
        "2024-01-08",
        cache_dir=tmp_path,
        downloader=network_must_not_run,
    )
    assert subset.index.max() < pd.Timestamp("2024-01-08")


def test_dynamic_download_end_excludes_an_active_us_session() -> None:
    eastern = ZoneInfo("America/New_York")
    before_final_bar = datetime(2026, 7, 15, 16, 30, tzinfo=eastern)
    after_final_bar = datetime(2026, 7, 15, 17, 30, tzinfo=eastern)

    assert _default_download_end(before_final_bar) == pd.Timestamp("2026-07-15")
    assert _default_download_end(after_final_bar) == pd.Timestamp("2026-07-16")


def test_market_features_use_only_present_and_past_values() -> None:
    market = _market_data()
    original = build_market_features(
        market,
        volatility_window=3,
        volume_window=3,
        momentum_window=3,
        dropna=False,
    )

    changed = market.copy()
    changed.loc[changed.index[-1], ("SPY", "close")] *= 4
    changed.loc[changed.index[-1], ("SPY", "volume")] *= 10
    changed.loc[changed.index[-1], ("^VIX", "close")] *= 3
    revised = build_market_features(
        changed,
        volatility_window=3,
        volume_window=3,
        momentum_window=3,
        dropna=False,
    )

    pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])


def test_feature_formulas_use_log_returns_and_prior_volume_window() -> None:
    market = _market_data()
    features = build_market_features(
        market,
        volatility_window=3,
        volume_window=3,
        momentum_window=3,
        dropna=False,
    )
    date = market.index[3]
    closes = market[("SPY", "close")]
    volumes = market[("SPY", "volume")]
    vix = market[("^VIX", "close")]

    expected_return = np.log(closes.iloc[3]) - np.log(closes.iloc[2])
    expected_volatility = np.log(closes).diff().iloc[1:4].std(ddof=1)
    expected_volume_z = (volumes.iloc[3] - volumes.iloc[:3].mean()) / volumes.iloc[:3].std(ddof=1)
    expected_vix_change = np.log(vix.iloc[3]) - np.log(vix.iloc[2])

    assert features.loc[date, "equity_return"] == pytest.approx(expected_return)
    assert features.loc[date, "realized_volatility"] == pytest.approx(expected_volatility)
    assert features.loc[date, "volume_zscore"] == pytest.approx(expected_volume_z)
    assert features.loc[date, "vix_change"] == pytest.approx(expected_vix_change)


def test_allocation_returns_are_unscaled_simple_returns() -> None:
    market = _market_data()
    returns = build_allocation_returns(market)

    expected_equity = market[("SPY", "close")].iloc[1] / market[("SPY", "close")].iloc[0] - 1
    expected_bond = market[("IEF", "close")].iloc[1] / market[("IEF", "close")].iloc[0] - 1
    assert returns.iloc[0, 0] == pytest.approx(expected_equity)
    assert returns.iloc[0, 1] == pytest.approx(expected_bond)


def test_features_and_backtest_prefer_adjusted_close() -> None:
    market = _market_data()
    market[("SPY", "adj_close")] = market[("SPY", "close")] * np.linspace(1.0, 1.2, len(market))
    market[("IEF", "adj_close")] = market[("IEF", "close")] * np.linspace(1.0, 1.1, len(market))
    market = market.sort_index(axis="columns")

    features = build_market_features(
        market,
        volatility_window=3,
        volume_window=3,
        momentum_window=3,
        dropna=False,
    )
    returns = build_allocation_returns(market)

    expected_log_return = np.log(market[("SPY", "adj_close")]).diff().iloc[1]
    expected_equity_return = market[("SPY", "adj_close")].pct_change().iloc[1]
    expected_bond_return = market[("IEF", "adj_close")].pct_change().iloc[1]
    assert features["equity_return"].iloc[1] == pytest.approx(expected_log_return)
    assert returns["equity_return"].iloc[0] == pytest.approx(expected_equity_return)
    assert returns["bond_return"].iloc[0] == pytest.approx(expected_bond_return)


def test_scaler_is_fit_on_training_rows_only() -> None:
    index = pd.bdate_range("2024-01-01", periods=6)
    frame = pd.DataFrame(
        np.arange(30, dtype=float).reshape(6, 5),
        index=index,
        columns=FEATURE_COLUMNS,
    )
    train, test = frame.iloc[:4], frame.iloc[4:].copy()
    test.iloc[:, :] += 10_000

    scaled_train, scaled_test, scaler = scale_train_test(train, test)

    np.testing.assert_allclose(scaler.mean_, train.mean().to_numpy())
    np.testing.assert_allclose(scaled_train.mean().to_numpy(), np.zeros(5), atol=1e-12)
    assert (scaled_test.to_numpy() > 100).all()


def test_chronological_split_never_shuffles_rows() -> None:
    index = pd.bdate_range("2024-01-01", periods=20)
    frame = pd.DataFrame({"value": np.arange(20)}, index=index)

    split = chronological_split(frame, train_fraction=0.6, validation_fraction=0.2)

    assert split.train["value"].tolist() == list(range(12))
    assert split.validation["value"].tolist() == list(range(12, 16))
    assert split.test["value"].tolist() == list(range(16, 20))
    assert split.train.index.max() < split.validation.index.min() < split.test.index.min()
