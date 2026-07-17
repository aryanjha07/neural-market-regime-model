import hashlib
import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from market_regime.config import ExperimentConfig, FeatureConfig, ModelConfig
from market_regime.features import MarketDataset, transform_features
from market_regime.forecast import (
    predict_live_regime,
    run_live_forecast,
    train_live_model,
)
from market_regime.neural_hmm import NeuralEmissionHMM
from market_regime.synthetic import generate_synthetic_market

FORECAST_KEYS = {
    "schema_version",
    "generated_at",
    "data_cutoff",
    "model_data_cutoff",
    "model_bundle_id",
    "model_created_at",
    "new_observations_since_training",
    "meaning",
    "assets",
    "allocation_policy",
    "current_regime_probabilities",
    "next_session_regime_probabilities",
    "allocation_horizon_regime_probabilities",
    "model",
}


def _reject_non_finite_json(value: str) -> None:
    raise AssertionError(f"forecast JSON contains non-finite value {value}")


def _load_strict_json(path) -> dict:  # noqa: ANN001
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_non_finite_json,
    )


def _assert_all_json_numbers_are_finite(value) -> None:  # noqa: ANN001
    if isinstance(value, dict):
        for item in value.values():
            _assert_all_json_numbers_are_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_all_json_numbers_are_finite(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        assert np.isfinite(value)


def _assert_probability_records(records: list[dict], n_states: int) -> None:
    assert len(records) == n_states
    expected_keys = frozenset({"state", "regime", "probability"})
    assert {frozenset(record) for record in records} == {expected_keys}
    assert [record["state"] for record in records] == list(range(n_states))
    assert all(isinstance(record["regime"], str) and record["regime"] for record in records)
    probabilities = np.asarray([record["probability"] for record in records], dtype=float)
    assert np.isfinite(probabilities).all()
    assert ((0.0 <= probabilities) & (probabilities <= 1.0)).all()
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-10)


def test_live_forecast_refits_all_rows_and_saves_meaning(tmp_path) -> None:
    market = generate_synthetic_market(n_days=180, seed=23)
    config = ExperimentConfig(
        seed=23,
        model=ModelConfig(
            n_states=3,
            n_mixtures=2,
            n_restarts=1,
            epochs=1,
            emission_steps=1,
        ),
    )

    result = run_live_forecast(
        MarketDataset(market.features, market.returns),
        config,
        tmp_path,
    )

    assert result.data_cutoff == market.features.index[-1]
    assert np.isclose(result.current_probabilities["probability"].sum(), 1.0)
    assert np.isclose(result.next_session_probabilities["probability"].sum(), 1.0)
    assert result.model_data_cutoff == market.features.index[-1]
    assert result.new_observations_since_training == 0
    assert (tmp_path / "live_neural_hmm.pt").exists()
    assert (tmp_path / "model_manifest.json").exists()
    assert (tmp_path / "latest_forecast.json").exists()

    payload = _load_strict_json(tmp_path / "latest_forecast.json")
    legacy_payload = _load_strict_json(tmp_path / "live_forecast.json")
    manifest = _load_strict_json(tmp_path / "model_manifest.json")
    assert payload == legacy_payload
    assert set(payload) == FORECAST_KEYS
    assert payload["schema_version"] == 1
    assert payload["data_cutoff"] == market.features.index[-1].isoformat()
    assert payload["model_data_cutoff"] == payload["data_cutoff"]
    assert payload["new_observations_since_training"] == 0
    assert isinstance(payload["model_bundle_id"], str) and payload["model_bundle_id"]
    assert payload["model_bundle_id"] == manifest["bundle_id"]
    assert payload["model_created_at"] == manifest["created_at"]
    assert pd.Timestamp(payload["generated_at"]).tzinfo is not None
    assert pd.Timestamp(payload["model_created_at"]).tzinfo is not None
    assert payload["model"]["training_observations"] == len(market.features)
    assert "not the direction" in payload["meaning"]
    assert payload["assets"] == {"equity": "SPY", "bond": "IEF", "volatility": "^VIX"}
    assert payload["allocation_policy"] == {
        "equity_weights_by_regime": {"Calm": 0.8, "Trending": 0.6, "Crisis": 0.2},
        "fallback_equity_weight": 0.6,
        "confidence_threshold": 0.55,
        "rebalance_frequency": "weekly",
        "execution_lag": 2,
    }
    _assert_probability_records(
        payload["current_regime_probabilities"],
        payload["model"]["n_states"],
    )
    _assert_probability_records(
        payload["next_session_regime_probabilities"],
        payload["model"]["n_states"],
    )
    _assert_probability_records(
        payload["allocation_horizon_regime_probabilities"],
        payload["model"]["n_states"],
    )
    _assert_all_json_numbers_are_finite(payload)


def test_predict_live_loads_frozen_bundle_and_advances_only_new_rows(tmp_path, monkeypatch) -> None:
    market = generate_synthetic_market(n_days=190, seed=29)
    training_rows = 160
    training_dataset = MarketDataset(
        market.features.iloc[:training_rows],
        market.returns.iloc[:training_rows],
    )
    full_dataset = MarketDataset(market.features, market.returns)
    config = ExperimentConfig(
        seed=29,
        model=ModelConfig(
            n_states=3,
            n_mixtures=2,
            n_restarts=1,
            epochs=1,
            emission_steps=1,
        ),
    )
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "predictions"
    training = train_live_model(training_dataset, config, model_dir)
    checkpoint = model_dir / "live_neural_hmm.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def fail_if_fitted(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("prediction must not train the model")

    monkeypatch.setattr(NeuralEmissionHMM, "fit", fail_if_fitted)
    prediction = predict_live_regime(full_dataset, config, model_dir, output_dir)

    assert training.data_cutoff == market.features.index[training_rows - 1]
    assert prediction.model_data_cutoff == training.data_cutoff
    assert prediction.data_cutoff == market.features.index[-1]
    assert prediction.new_observations_since_training == 30
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    assert np.isclose(prediction.current_probabilities["probability"].sum(), 1.0)
    assert np.isclose(prediction.next_session_probabilities["probability"].sum(), 1.0)

    payload = _load_strict_json(output_dir / "latest_forecast.json")
    assert set(payload) == FORECAST_KEYS
    assert payload["schema_version"] == 1
    assert pd.Timestamp(payload["model_data_cutoff"]) == training.data_cutoff
    assert pd.Timestamp(payload["data_cutoff"]) == prediction.data_cutoff
    assert pd.Timestamp(payload["model_data_cutoff"]) < pd.Timestamp(payload["data_cutoff"])
    assert payload["new_observations_since_training"] == 30
    assert payload["model"]["training_observations"] == training_rows
    assert payload["model_bundle_id"]
    assert payload["model_created_at"]
    _assert_probability_records(
        payload["current_regime_probabilities"],
        payload["model"]["n_states"],
    )
    _assert_probability_records(
        payload["next_session_regime_probabilities"],
        payload["model"]["n_states"],
    )
    _assert_probability_records(
        payload["allocation_horizon_regime_probabilities"],
        payload["model"]["n_states"],
    )
    _assert_all_json_numbers_are_finite(payload)

    manifest = _load_strict_json(model_dir / "model_manifest.json")
    assert payload["model_bundle_id"] == manifest["bundle_id"]
    assert payload["model_created_at"] == manifest["created_at"]
    scaler = joblib.load(model_dir / manifest["files"]["scaler"])
    model = NeuralEmissionHMM.load_checkpoint(model_dir / manifest["files"]["checkpoint"])
    full_scaled = transform_features(market.features, scaler).to_numpy(dtype=np.float64)
    expected_current = model.filter(full_scaled)[-1].cpu().numpy()
    np.testing.assert_allclose(
        prediction.current_probabilities["probability"].to_numpy(),
        expected_current,
        rtol=1e-10,
        atol=1e-12,
    )
    expected_allocation = expected_current @ np.linalg.matrix_power(
        model.transition_matrix_.cpu().numpy(),
        config.backtest.execution_lag,
    )
    np.testing.assert_allclose(
        [row["probability"] for row in payload["allocation_horizon_regime_probabilities"]],
        expected_allocation,
        rtol=1e-10,
        atol=1e-12,
    )

    labels = {int(state): label for state, label in manifest["regime_labels"].items()}
    assert (
        dict(
            zip(
                prediction.current_probabilities["state"],
                prediction.current_probabilities["regime"],
                strict=True,
            )
        )
        == labels
    )
    history = output_dir / "prediction_history.csv"
    history_frame = pd.read_csv(history)
    assert len(history_frame) == 3
    assert history_frame["model_bundle_id"].unique().tolist() == [manifest["bundle_id"]]
    assert history_frame["model_data_cutoff"].unique().tolist() == [
        training.data_cutoff.isoformat()
    ]
    predict_live_regime(full_dataset, config, model_dir, output_dir)
    assert len(pd.read_csv(history)) == 3


def test_four_state_forecast_gives_every_regime_an_allocation(tmp_path) -> None:
    market = generate_synthetic_market(n_days=180, seed=37)
    config = ExperimentConfig(
        seed=37,
        model=ModelConfig(
            n_states=4,
            n_mixtures=2,
            n_restarts=1,
            epochs=1,
            emission_steps=1,
        ),
    )

    run_live_forecast(MarketDataset(market.features, market.returns), config, tmp_path)
    payload = _load_strict_json(tmp_path / "latest_forecast.json")
    weights = payload["allocation_policy"]["equity_weights_by_regime"]
    regimes = {row["regime"] for row in payload["allocation_horizon_regime_probabilities"]}

    assert regimes <= set(weights)
    for regime in regimes:
        expected = 0.8 if regime == "Calm" else 0.2 if regime == "Crisis" else 0.6
        assert weights[regime] == expected


def test_predict_live_rejects_changed_history_and_configuration(tmp_path) -> None:
    market = generate_synthetic_market(n_days=170, seed=41)
    training_rows = 150
    config = ExperimentConfig(
        seed=41,
        model=ModelConfig(
            n_states=3,
            n_mixtures=2,
            n_restarts=1,
            epochs=1,
            emission_steps=1,
        ),
    )
    model_dir = tmp_path / "model"
    training_dataset = MarketDataset(
        market.features.iloc[:training_rows],
        market.returns.iloc[:training_rows],
    )
    train_live_model(training_dataset, config, model_dir)

    changed_features = market.features.copy()
    changed_features.iloc[0, 0] += 0.01
    with pytest.raises(ValueError, match="historical features changed"):
        predict_live_regime(
            MarketDataset(changed_features, market.returns),
            config,
            model_dir,
            tmp_path / "changed",
        )

    mismatched = replace(config, features=FeatureConfig(momentum_window=10))
    with pytest.raises(ValueError, match="settings do not match"):
        predict_live_regime(
            MarketDataset(market.features, market.returns),
            mismatched,
            model_dir,
            tmp_path / "mismatch",
        )


def test_predict_live_requires_a_trained_bundle(tmp_path) -> None:
    market = generate_synthetic_market(n_days=100, seed=43)

    with pytest.raises(FileNotFoundError, match="train-live first"):
        predict_live_regime(
            MarketDataset(market.features, market.returns),
            ExperimentConfig(),
            tmp_path / "missing",
            tmp_path / "predictions",
        )
