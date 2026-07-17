"""Small, dependency-light data contract for the public Streamlit dashboard."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

DEFAULT_ASSETS = {"equity": "SPY", "bond": "IEF", "volatility": "^VIX"}
DEFAULT_EQUITY_WEIGHTS = {"Calm": 0.8, "Trending": 0.6, "Crisis": 0.2}
DEFAULT_FORECAST_URL = (
    "https://github.com/aryanjha07/neural-market-regime-model/"
    "releases/download/live-forecast/latest_forecast.json"
)
DEFAULT_HISTORY_URL = (
    "https://github.com/aryanjha07/neural-market-regime-model/"
    "releases/download/live-forecast/prediction_history.csv"
)
MAX_FORECAST_BYTES = 2_000_000
MAX_HISTORY_BYTES = 10_000_000
UPDATE_TIME_ZONE = ZoneInfo("America/New_York")
UPDATE_WEEKDAYS = frozenset(range(5))
UPDATE_LOCAL_TIME = time(hour=18, minute=37)


class DashboardDataError(ValueError):
    """Raised when published dashboard data is missing, invalid, or unsafe."""


@dataclass(frozen=True, slots=True)
class RegimeProbability:
    state: int
    regime: str
    probability: float


@dataclass(frozen=True, slots=True)
class AllocationPolicy:
    equity_weights_by_regime: dict[str, float]
    fallback_equity_weight: float
    confidence_threshold: float
    rebalance_frequency: str
    execution_lag: int


@dataclass(frozen=True, slots=True)
class ForecastSnapshot:
    schema_version: int
    generated_at: datetime
    data_cutoff: datetime
    model_data_cutoff: datetime
    model_bundle_id: str
    model_created_at: datetime
    new_observations_since_training: int
    meaning: str
    assets: dict[str, str]
    allocation_policy: AllocationPolicy
    current: tuple[RegimeProbability, ...]
    next_session: tuple[RegimeProbability, ...]
    allocation_horizon: tuple[RegimeProbability, ...]
    model: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedForecast:
    snapshot: ForecastSnapshot
    source: str


@dataclass(frozen=True, slots=True)
class LoadedHistory:
    frame: pd.DataFrame
    source: str


@dataclass(frozen=True, slots=True)
class AllocationTarget:
    equity_weight: float
    bond_weight: float
    used_fallback: bool


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DashboardDataError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DashboardDataError(f"{field} must be a non-empty string")
    return value.strip()


def _timestamp(value: Any, field: str, *, timezone_required: bool = False) -> datetime:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DashboardDataError(f"{field} must be an ISO timestamp") from exc
    if timezone_required and parsed.tzinfo is None:
        raise DashboardDataError(f"{field} must include a timezone")
    return parsed


def _unit_interval(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise DashboardDataError(f"{field} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DashboardDataError(f"{field} must be a number") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise DashboardDataError(f"{field} must be finite and between zero and one")
    return number


def _probabilities(value: Any, field: str) -> tuple[RegimeProbability, ...]:
    if not isinstance(value, list) or not value:
        raise DashboardDataError(f"{field} must be a non-empty list")

    rows: list[RegimeProbability] = []
    seen_states: set[int] = set()
    for index, raw_row in enumerate(value):
        row = _mapping(raw_row, f"{field}[{index}]")
        state = row.get("state")
        if isinstance(state, bool) or not isinstance(state, int) or state < 0:
            raise DashboardDataError(f"{field}[{index}].state must be a non-negative integer")
        if state in seen_states:
            raise DashboardDataError(f"{field} contains duplicate states")
        seen_states.add(state)
        rows.append(
            RegimeProbability(
                state=state,
                regime=_text(row.get("regime"), f"{field}[{index}].regime"),
                probability=_unit_interval(row.get("probability"), f"{field}[{index}].probability"),
            )
        )

    rows.sort(key=lambda row: row.state)
    if not math.isclose(sum(row.probability for row in rows), 1.0, abs_tol=1e-6):
        raise DashboardDataError(f"{field} probabilities must sum to one")
    return tuple(rows)


def _allocation_policy(value: Any) -> AllocationPolicy:
    if value is None:
        return AllocationPolicy(dict(DEFAULT_EQUITY_WEIGHTS), 0.6, 0.55, "weekly", 2)
    policy = _mapping(value, "allocation_policy")
    raw_weights = _mapping(
        policy.get("equity_weights_by_regime"),
        "allocation_policy.equity_weights_by_regime",
    )
    weights = {
        _text(regime, "allocation regime"): _unit_interval(
            weight, f"allocation_policy.equity_weights_by_regime.{regime}"
        )
        for regime, weight in raw_weights.items()
    }
    if not weights:
        raise DashboardDataError("allocation policy must contain at least one regime")
    execution_lag = policy.get("execution_lag")
    if isinstance(execution_lag, bool) or not isinstance(execution_lag, int) or execution_lag < 1:
        raise DashboardDataError("allocation_policy.execution_lag must be a positive integer")
    frequency = _text(policy.get("rebalance_frequency"), "allocation_policy.rebalance_frequency")
    if frequency not in {"daily", "weekly", "monthly"}:
        raise DashboardDataError("allocation rebalance frequency is unsupported")
    return AllocationPolicy(
        equity_weights_by_regime=weights,
        fallback_equity_weight=_unit_interval(
            policy.get("fallback_equity_weight"),
            "allocation_policy.fallback_equity_weight",
        ),
        confidence_threshold=_unit_interval(
            policy.get("confidence_threshold"),
            "allocation_policy.confidence_threshold",
        ),
        rebalance_frequency=frequency,
        execution_lag=execution_lag,
    )


def parse_forecast(payload: Any) -> ForecastSnapshot:
    """Validate and convert a public forecast JSON object."""

    root = _mapping(payload, "forecast")
    if root.get("schema_version") != 1:
        raise DashboardDataError("unsupported forecast schema version")
    generated_at = _timestamp(root.get("generated_at"), "generated_at", timezone_required=True)
    data_cutoff = _timestamp(root.get("data_cutoff"), "data_cutoff")
    model_data_cutoff = _timestamp(root.get("model_data_cutoff"), "model_data_cutoff")
    model_created_at = _timestamp(
        root.get("model_created_at"), "model_created_at", timezone_required=True
    )
    if generated_at > datetime.now(UTC) + timedelta(minutes=10):
        raise DashboardDataError("generated_at cannot be in the future")
    if data_cutoff.date() > generated_at.date():
        raise DashboardDataError("data cutoff cannot be later than forecast generation")
    if model_data_cutoff.date() > data_cutoff.date():
        raise DashboardDataError("model cutoff cannot be later than the prediction data cutoff")
    if model_created_at > generated_at + timedelta(minutes=10):
        raise DashboardDataError("model creation cannot be later than forecast generation")

    new_observations = root.get("new_observations_since_training")
    if isinstance(new_observations, bool) or not isinstance(new_observations, int):
        raise DashboardDataError("new_observations_since_training must be an integer")
    if new_observations < 0:
        raise DashboardDataError("new_observations_since_training cannot be negative")

    current = _probabilities(
        root.get("current_regime_probabilities"), "current_regime_probabilities"
    )
    next_session = _probabilities(
        root.get("next_session_regime_probabilities"),
        "next_session_regime_probabilities",
    )
    allocation_horizon = _probabilities(
        root.get("allocation_horizon_regime_probabilities"),
        "allocation_horizon_regime_probabilities",
    )
    current_labels = {(row.state, row.regime) for row in current}
    next_labels = {(row.state, row.regime) for row in next_session}
    allocation_labels = {(row.state, row.regime) for row in allocation_horizon}
    if current_labels != next_labels or current_labels != allocation_labels:
        raise DashboardDataError("forecast probability states do not match")

    model = _mapping(root.get("model"), "model")
    n_states = model.get("n_states")
    if isinstance(n_states, bool) or not isinstance(n_states, int) or n_states != len(current):
        raise DashboardDataError("model.n_states does not match the probability rows")

    raw_assets = root.get("assets", DEFAULT_ASSETS)
    assets = {
        key: _text(value, f"assets.{key}") for key, value in _mapping(raw_assets, "assets").items()
    }
    if set(DEFAULT_ASSETS) - set(assets):
        raise DashboardDataError("assets must define equity, bond, and volatility symbols")

    allocation = _allocation_policy(root.get("allocation_policy"))
    missing_allocations = {row.regime for row in allocation_horizon} - set(
        allocation.equity_weights_by_regime
    )
    if missing_allocations:
        raise DashboardDataError(
            f"allocation policy is missing regimes: {sorted(missing_allocations)}"
        )

    return ForecastSnapshot(
        schema_version=1,
        generated_at=generated_at,
        data_cutoff=data_cutoff,
        model_data_cutoff=model_data_cutoff,
        model_bundle_id=_text(root.get("model_bundle_id"), "model_bundle_id"),
        model_created_at=model_created_at,
        new_observations_since_training=new_observations,
        meaning=_text(root.get("meaning"), "meaning"),
        assets=assets,
        allocation_policy=allocation,
        current=current,
        next_session=next_session,
        allocation_horizon=allocation_horizon,
        model=model,
    )


def _strict_json(data: bytes) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                DashboardDataError(f"forecast contains invalid number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DashboardDataError("forecast is not valid UTF-8 JSON") from exc


def _download(url: str, *, timeout: float, max_bytes: int, media_type: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DashboardDataError("dashboard data URL must use HTTP or HTTPS")
    request = Request(
        url,
        headers={"Accept": media_type, "User-Agent": "neural-market-regime-dashboard/1"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            declared_size = response.headers.get("Content-Length")
            if declared_size is not None and int(declared_size) > max_bytes:
                raise DashboardDataError("dashboard data response is too large")
            data = response.read(max_bytes + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise DashboardDataError(f"could not download dashboard data: {exc}") from exc
    if len(data) > max_bytes:
        raise DashboardDataError("dashboard data response is too large")
    return data


def load_forecast(
    local_path: str | Path = "artifacts/live_predictions/latest_forecast.json",
    remote_url: str | None = DEFAULT_FORECAST_URL,
    *,
    timeout: float = 8.0,
) -> LoadedForecast:
    """Load a validated local forecast, with a remote Release fallback."""

    errors: list[str] = []
    path = Path(local_path)
    if path.is_file():
        try:
            return LoadedForecast(parse_forecast(_strict_json(path.read_bytes())), str(path))
        except (OSError, DashboardDataError) as exc:
            errors.append(f"local forecast: {exc}")
    if remote_url:
        try:
            data = _download(
                remote_url,
                timeout=timeout,
                max_bytes=MAX_FORECAST_BYTES,
                media_type="application/json",
            )
            return LoadedForecast(parse_forecast(_strict_json(data)), remote_url)
        except DashboardDataError as exc:
            errors.append(f"published forecast: {exc}")
    detail = "; ".join(errors) if errors else "no local file or remote URL was provided"
    raise DashboardDataError(f"No valid forecast is available ({detail})")


def parse_history(data: bytes) -> pd.DataFrame:
    """Validate prediction history and return one chronological row per state and date."""

    try:
        frame = pd.read_csv(BytesIO(data))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise DashboardDataError("prediction history is not valid CSV") from exc
    required = {
        "prediction_data_cutoff",
        "model_data_cutoff",
        "model_bundle_id",
        "generated_at",
        "state",
        "regime",
        "current_probability",
        "next_session_probability",
    }
    missing = required - set(frame.columns)
    if frame.empty or missing:
        raise DashboardDataError(f"prediction history is empty or missing {sorted(missing)}")

    result = frame.loc[:, sorted(required)].copy()
    try:
        result["prediction_data_cutoff"] = pd.to_datetime(
            result["prediction_data_cutoff"], errors="raise"
        )
        result["model_data_cutoff"] = pd.to_datetime(result["model_data_cutoff"], errors="raise")
        result["generated_at"] = pd.to_datetime(result["generated_at"], utc=True, errors="raise")
        result["state"] = pd.to_numeric(result["state"], errors="raise")
        for column in ("current_probability", "next_session_probability"):
            result[column] = pd.to_numeric(result[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise DashboardDataError("prediction history contains invalid dates or numbers") from exc

    valid_states = result["state"].map(
        lambda value: (
            math.isfinite(float(value)) and float(value).is_integer() and float(value) >= 0
        )
    )
    if not valid_states.all():
        raise DashboardDataError("prediction history states must be non-negative integers")
    result["state"] = result["state"].astype(int)
    for column in ("regime", "model_bundle_id"):
        values = result[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise DashboardDataError(f"prediction history {column} values cannot be empty")
        result[column] = values.astype(str)
    unstable_labels = result.groupby(["model_bundle_id", "state"])["regime"].nunique().gt(1).any()
    if unstable_labels:
        raise DashboardDataError("prediction history changes a state label within one model")
    if (result["model_data_cutoff"] > result["prediction_data_cutoff"]).any():
        raise DashboardDataError("prediction history contains a model cutoff after its data")
    generated_dates = result["generated_at"].dt.tz_convert(UTC).dt.date
    if (generated_dates < result["prediction_data_cutoff"].dt.date).any():
        raise DashboardDataError("prediction history was generated before its market data")

    probabilities = result[["current_probability", "next_session_probability"]]
    if not probabilities.map(math.isfinite).all().all():
        raise DashboardDataError("prediction history contains non-finite probabilities")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any().any():
        raise DashboardDataError("prediction history probabilities must be between zero and one")
    totals = result.groupby("prediction_data_cutoff", sort=False)[
        ["current_probability", "next_session_probability"]
    ].sum()
    if not ((totals - 1.0).abs() <= 1e-6).all().all():
        raise DashboardDataError("prediction history probabilities must sum to one per date")
    if result[["prediction_data_cutoff", "state"]].duplicated().any():
        raise DashboardDataError("prediction history contains duplicate date-state rows")

    return result.sort_values(["prediction_data_cutoff", "state"]).reset_index(drop=True)


def load_history(
    local_path: str | Path = "artifacts/live_predictions/prediction_history.csv",
    remote_url: str | None = DEFAULT_HISTORY_URL,
    *,
    timeout: float = 8.0,
) -> LoadedHistory:
    """Load validated local history, with a remote Release fallback."""

    errors: list[str] = []
    path = Path(local_path)
    if path.is_file():
        try:
            return LoadedHistory(parse_history(path.read_bytes()), str(path))
        except (OSError, DashboardDataError) as exc:
            errors.append(f"local history: {exc}")
    if remote_url:
        try:
            data = _download(
                remote_url,
                timeout=timeout,
                max_bytes=MAX_HISTORY_BYTES,
                media_type="text/csv",
            )
            return LoadedHistory(parse_history(data), remote_url)
        except DashboardDataError as exc:
            errors.append(f"published history: {exc}")
    detail = "; ".join(errors) if errors else "no local file or remote URL was provided"
    raise DashboardDataError(f"No valid prediction history is available ({detail})")


def business_days_since(cutoff: date, *, today: date | None = None) -> int:
    """Return an approximate weekday lag; exchange holidays are intentionally not inferred."""

    current_day = date.today() if today is None else today
    if cutoff > current_day:
        raise DashboardDataError("data cutoff cannot be in the future")
    if cutoff == current_day:
        return 0
    lag = 0
    cursor = cutoff + timedelta(days=1)
    while cursor <= current_day:
        if cursor.weekday() < 5:
            lag += 1
        cursor += timedelta(days=1)
    return lag


def freshness(snapshot: ForecastSnapshot, *, today: date | None = None) -> tuple[str, int]:
    lag = business_days_since(snapshot.data_cutoff.date(), today=today)
    if lag <= 1:
        return "Fresh", lag
    if lag == 2:
        return "Delayed", lag
    return "Stale", lag


def next_scheduled_run(*, now: datetime | None = None) -> datetime:
    """Return the next weekday 6:37 PM New York workflow start in UTC."""

    current = datetime.now(UTC) if now is None else now
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include timezone information")

    local_now = current.astimezone(UPDATE_TIME_ZONE)
    for day_offset in range(8):
        candidate_date = local_now.date() + timedelta(days=day_offset)
        if candidate_date.weekday() not in UPDATE_WEEKDAYS:
            continue
        candidate = datetime.combine(
            candidate_date,
            UPDATE_LOCAL_TIME,
            tzinfo=UPDATE_TIME_ZONE,
        )
        if candidate > local_now:
            return candidate.astimezone(UTC)
    raise RuntimeError("could not calculate the next scheduled workflow run")


def most_likely(rows: tuple[RegimeProbability, ...]) -> RegimeProbability:
    return max(rows, key=lambda row: row.probability)


def allocation_target(snapshot: ForecastSnapshot) -> AllocationTarget:
    """Mirror the probability-weighted allocation rule used by the backtest."""

    confidence = max(row.probability for row in snapshot.allocation_horizon)
    policy = snapshot.allocation_policy
    use_fallback = confidence < policy.confidence_threshold
    if use_fallback:
        equity_weight = policy.fallback_equity_weight
    else:
        equity_weight = sum(
            row.probability * policy.equity_weights_by_regime[row.regime]
            for row in snapshot.allocation_horizon
        )
    return AllocationTarget(
        equity_weight=equity_weight,
        bond_weight=1.0 - equity_weight,
        used_fallback=use_fallback,
    )


__all__ = [
    "AllocationTarget",
    "DashboardDataError",
    "DEFAULT_FORECAST_URL",
    "DEFAULT_HISTORY_URL",
    "ForecastSnapshot",
    "LoadedForecast",
    "LoadedHistory",
    "allocation_target",
    "business_days_since",
    "freshness",
    "load_forecast",
    "load_history",
    "most_likely",
    "next_scheduled_run",
    "parse_forecast",
    "parse_history",
]
