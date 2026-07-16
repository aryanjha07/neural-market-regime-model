"""Download and validate daily market data with a reproducible CSV cache."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ("SPY", "IEF", "^VIX")
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _default_download_end(now: datetime | None = None) -> pd.Timestamp:
    """Return an exclusive end date that never includes an active US session."""

    eastern = ZoneInfo("America/New_York")
    current = datetime.now(eastern) if now is None else now.astimezone(eastern)
    today = pd.Timestamp(current.date()).as_unit("ns")
    if current.weekday() >= 5:
        return today + pd.offsets.Day(1)
    # Yahoo's finalized daily bar can arrive after the official close. Before
    # 17:00 ET, exclude the current date and give it a different cache key.
    return today + pd.offsets.Day(1) if current.time() >= time(17, 0) else today


def _as_date(value: str | pd.Timestamp, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date") from exc
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _resolved_dates(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_date = _as_date(start, name="start")
    # yfinance treats end as exclusive. The dynamic default avoids requesting
    # and caching the current US session before its daily bar is finalized.
    end_date = _default_download_end() if end is None else _as_date(end, name="end")
    if end_date <= start_date:
        raise ValueError("end must be later than start")
    return start_date, end_date


def _safe_ticker(ticker: str) -> str:
    prefix = "index_" if ticker.startswith("^") else ""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker.lstrip("^")).strip("_")
    if not cleaned:
        raise ValueError("ticker must contain at least one letter or number")
    return f"{prefix}{cleaned.lower()}"


def cache_path_for(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None,
    cache_dir: str | Path = "data/raw",
) -> Path:
    """Return the cache path for an exact ticker and date request."""

    start_date, end_date = _resolved_dates(start, end)
    filename = f"{_safe_ticker(ticker)}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv"
    return Path(cache_dir) / filename


def _flatten_yfinance_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame

    result = frame
    for level in range(result.columns.nlevels):
        values = result.columns.get_level_values(level).astype(str)
        if ticker in set(values):
            result = result.xs(ticker, axis=1, level=level, drop_level=True)
            break

    if isinstance(result.columns, pd.MultiIndex):
        known = {"open", "high", "low", "close", "adj close", "volume"}
        for level in range(result.columns.nlevels):
            normalized = {
                str(value).strip().lower() for value in result.columns.get_level_values(level)
            }
            if normalized & known:
                result = result.copy()
                result.columns = result.columns.get_level_values(level)
                break
    return result


def validate_ohlcv(frame: pd.DataFrame, ticker: str = "unknown") -> pd.DataFrame:
    """Normalize a single-ticker OHLCV frame or raise on unsafe input.

    Returned columns use lower-case snake case and the index is a sorted,
    timezone-naive ``DatetimeIndex`` named ``date``.
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"No market data returned for {ticker}")

    clean = _flatten_yfinance_columns(frame.copy(), ticker)
    column_map = {column: str(column).strip().lower().replace(" ", "_") for column in clean.columns}
    clean = clean.rename(columns=column_map)
    if clean.columns.duplicated().any():
        raise ValueError(f"Duplicate market-data columns returned for {ticker}")

    missing = sorted(set(OHLCV_COLUMNS) - set(clean.columns))
    if missing:
        raise ValueError(f"Market data for {ticker} is missing columns: {missing}")

    try:
        clean.index = pd.to_datetime(clean.index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Market data for {ticker} has an invalid date index") from exc
    if not isinstance(clean.index, pd.DatetimeIndex):
        raise ValueError(f"Market data for {ticker} must have a DatetimeIndex")
    if clean.index.tz is not None:
        clean.index = clean.index.tz_localize(None)
    clean.index = clean.index.normalize()
    clean.index.name = "date"
    if clean.index.duplicated().any():
        raise ValueError(f"Market data for {ticker} contains duplicate dates")
    clean = clean.sort_index()

    ordered = [*OHLCV_COLUMNS]
    if "adj_close" in clean.columns:
        ordered.insert(4, "adj_close")
    clean = clean.loc[:, ordered].apply(pd.to_numeric, errors="coerce")
    clean = clean.dropna(subset=["close"])
    if clean.empty:
        raise ValueError(f"Market data for {ticker} contains no valid close prices")
    values = clean.loc[:, OHLCV_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Market data for {ticker} contains missing or infinite OHLCV values")
    if (clean["close"] <= 0).any():
        raise ValueError(f"Market data for {ticker} contains non-positive close prices")
    if (clean["volume"] < 0).any():
        raise ValueError(f"Market data for {ticker} contains negative volume")
    return clean


def _read_cached_ticker(path: Path, ticker: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read cached market data at {path}") from exc
    return validate_ohlcv(frame, ticker)


def _covering_cache_path(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str | Path,
) -> Path | None:
    """Find the smallest cached request whose date range covers this one."""

    prefix = f"{_safe_ticker(ticker)}_{start:%Y%m%d}_"
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for path in Path(cache_dir).glob(f"{prefix}*.csv"):
        encoded_end = path.stem.removeprefix(prefix)
        try:
            cached_end = pd.to_datetime(encoded_end, format="%Y%m%d")
        except ValueError:
            continue
        if cached_end >= end:
            candidates.append((cached_end, path))
    return min(candidates, default=(None, None), key=lambda item: item[0])[1]


def download_ticker_data(
    ticker: str,
    start: str | pd.Timestamp = "2005-01-01",
    end: str | pd.Timestamp | None = None,
    *,
    cache_dir: str | Path = "data/raw",
    refresh: bool = False,
    downloader: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Load one ticker from an exact-request CSV cache or download it.

    ``end`` is exclusive, matching yfinance. Supplying ``downloader`` keeps the
    function straightforward to test without network access.
    """

    ticker = str(ticker).strip()
    if not ticker:
        raise ValueError("ticker cannot be empty")
    start_date, end_date = _resolved_dates(start, end)
    cache_path = cache_path_for(ticker, start_date, end_date, cache_dir)
    if cache_path.exists() and not refresh:
        return _read_cached_ticker(cache_path, ticker)
    if not refresh:
        covering_path = _covering_cache_path(ticker, start_date, end_date, cache_dir)
        if covering_path is not None:
            cached = _read_cached_ticker(covering_path, ticker)
            requested = cached.loc[(cached.index >= start_date) & (cached.index < end_date)]
            if not requested.empty:
                return requested

    if downloader is None:
        import yfinance as yf

        downloader = yf.download

    raw = downloader(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    clean = validate_ohlcv(raw, ticker)
    requested = clean.loc[(clean.index >= start_date) & (clean.index < end_date)]
    if requested.empty:
        raise ValueError(f"No market data for {ticker} falls inside the requested date range")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    requested.to_csv(cache_path, index_label="date")
    return requested


def download_market_data(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str | pd.Timestamp = "2005-01-01",
    end: str | pd.Timestamp | None = None,
    *,
    cache_dir: str | Path = "data/raw",
    refresh: bool = False,
    downloader: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return aligned OHLCV data with ``(ticker, field)`` columns."""

    ticker_list = [str(ticker).strip() for ticker in tickers]
    if not ticker_list or any(not ticker for ticker in ticker_list):
        raise ValueError("tickers must contain at least one non-empty symbol")
    if len(set(ticker_list)) != len(ticker_list):
        raise ValueError("tickers must not contain duplicates")

    frames = {
        ticker: download_ticker_data(
            ticker,
            start,
            end,
            cache_dir=cache_dir,
            refresh=refresh,
            downloader=downloader,
        )
        for ticker in ticker_list
    }
    combined = pd.concat(frames, axis="columns", names=["ticker", "field"])
    combined.index.name = "date"
    return combined.sort_index()


def ticker_frame(market_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Select one ticker from a frame returned by :func:`download_market_data`."""

    if not isinstance(market_data.columns, pd.MultiIndex):
        raise ValueError("market_data must have (ticker, field) MultiIndex columns")
    if ticker not in market_data.columns.get_level_values(0):
        raise KeyError(f"Ticker {ticker!r} is not present in market_data")
    selected = market_data.xs(ticker, axis="columns", level=0).copy()
    selected.index.name = market_data.index.name or "date"
    return selected
