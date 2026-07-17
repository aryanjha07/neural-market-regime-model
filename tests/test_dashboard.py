import copy
import json
from datetime import date

import pandas as pd
import pytest

from market_regime.dashboard import (
    DashboardDataError,
    allocation_target,
    business_days_since,
    freshness,
    load_forecast,
    parse_forecast,
    parse_history,
)


@pytest.fixture
def forecast_payload() -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-07-17T03:29:57+00:00",
        "data_cutoff": "2026-07-16T00:00:00",
        "model_data_cutoff": "2026-07-11T00:00:00",
        "model_bundle_id": "20260711-example",
        "model_created_at": "2026-07-12T01:00:00+00:00",
        "new_observations_since_training": 4,
        "meaning": "next session estimates a hidden regime, not return direction",
        "assets": {"equity": "SPY", "bond": "IEF", "volatility": "^VIX"},
        "allocation_policy": {
            "equity_weights_by_regime": {"Calm": 0.8, "Trending": 0.6, "Crisis": 0.2},
            "fallback_equity_weight": 0.6,
            "confidence_threshold": 0.55,
            "rebalance_frequency": "weekly",
            "execution_lag": 2,
        },
        "current_regime_probabilities": [
            {"state": 0, "regime": "Crisis", "probability": 0.1},
            {"state": 1, "regime": "Calm", "probability": 0.2},
            {"state": 2, "regime": "Trending", "probability": 0.7},
        ],
        "next_session_regime_probabilities": [
            {"state": 0, "regime": "Crisis", "probability": 0.1},
            {"state": 1, "regime": "Calm", "probability": 0.3},
            {"state": 2, "regime": "Trending", "probability": 0.6},
        ],
        "allocation_horizon_regime_probabilities": [
            {"state": 0, "regime": "Crisis", "probability": 0.2},
            {"state": 1, "regime": "Calm", "probability": 0.1},
            {"state": 2, "regime": "Trending", "probability": 0.7},
        ],
        "model": {"n_states": 3, "training_observations": 5_000},
    }


def test_parse_forecast_and_probability_weighted_allocation(forecast_payload) -> None:
    snapshot = parse_forecast(forecast_payload)
    target = allocation_target(snapshot)

    assert snapshot.model_bundle_id == "20260711-example"
    assert snapshot.assets["equity"] == "SPY"
    assert target.used_fallback is False
    assert target.equity_weight == pytest.approx(0.54)
    assert target.bond_weight == pytest.approx(0.46)


def test_low_confidence_forecast_uses_policy_fallback(forecast_payload) -> None:
    payload = copy.deepcopy(forecast_payload)
    probabilities = [0.34, 0.33, 0.33]
    for row, probability in zip(
        payload["allocation_horizon_regime_probabilities"], probabilities, strict=True
    ):
        row["probability"] = probability

    target = allocation_target(parse_forecast(payload))

    assert target.used_fallback is True
    assert target.equity_weight == pytest.approx(0.6)
    assert target.bond_weight == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(schema_version=2), "schema version"),
        (
            lambda payload: payload["allocation_horizon_regime_probabilities"][0].update(
                probability=float("nan")
            ),
            "finite",
        ),
        (
            lambda payload: payload["allocation_horizon_regime_probabilities"][0].update(
                probability=0.5
            ),
            "sum to one",
        ),
        (lambda payload: payload.update(new_observations_since_training=-1), "cannot be negative"),
        (lambda payload: payload.update(data_cutoff="2099-01-01"), "forecast generation"),
    ],
)
def test_parse_forecast_rejects_unsafe_public_data(forecast_payload, mutation, match) -> None:
    payload = copy.deepcopy(forecast_payload)
    mutation(payload)

    with pytest.raises(DashboardDataError, match=match):
        parse_forecast(payload)


def test_load_forecast_prefers_a_valid_local_file(tmp_path, forecast_payload) -> None:
    path = tmp_path / "latest_forecast.json"
    path.write_text(json.dumps(forecast_payload), encoding="utf-8")

    loaded = load_forecast(path, "https://invalid.example/forecast.json")

    assert loaded.source == str(path)
    assert loaded.snapshot.data_cutoff.date() == date(2026, 7, 16)


def test_prediction_history_requires_normalized_unique_rows() -> None:
    rows = []
    for state, (regime, probability) in enumerate(
        [("Crisis", 0.1), ("Calm", 0.2), ("Trending", 0.7)]
    ):
        rows.append(
            {
                "prediction_data_cutoff": "2026-07-16T00:00:00",
                "model_data_cutoff": "2026-07-11T00:00:00",
                "model_bundle_id": "bundle",
                "generated_at": "2026-07-17T03:29:57+00:00",
                "state": state,
                "regime": regime,
                "current_probability": probability,
                "next_session_probability": probability,
            }
        )
    data = pd.DataFrame(rows).to_csv(index=False).encode()

    parsed = parse_history(data)

    assert len(parsed) == 3
    duplicate = pd.concat([pd.DataFrame(rows), pd.DataFrame([rows[0]])]).to_csv(index=False)
    with pytest.raises(DashboardDataError, match="sum to one|duplicate"):
        parse_history(duplicate.encode())

    fractional_state = copy.deepcopy(rows)
    fractional_state[0]["state"] = 0.5
    with pytest.raises(DashboardDataError, match="non-negative integers"):
        parse_history(pd.DataFrame(fractional_state).to_csv(index=False).encode())

    blank_regime = copy.deepcopy(rows)
    blank_regime[0]["regime"] = " "
    with pytest.raises(DashboardDataError, match="regime values cannot be empty"):
        parse_history(pd.DataFrame(blank_regime).to_csv(index=False).encode())


def test_freshness_counts_weekdays_without_marking_a_weekend_stale(forecast_payload) -> None:
    snapshot = parse_forecast(forecast_payload)

    assert business_days_since(date(2026, 7, 17), today=date(2026, 7, 20)) == 1
    with pytest.raises(DashboardDataError, match="future"):
        business_days_since(date(2026, 7, 21), today=date(2026, 7, 20))
    assert freshness(snapshot, today=date(2026, 7, 17)) == ("Fresh", 1)
    assert freshness(snapshot, today=date(2026, 7, 21)) == ("Stale", 3)
