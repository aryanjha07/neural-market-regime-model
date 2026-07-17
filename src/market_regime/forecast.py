"""Versioned full-history training and fast next-session regime inference."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import yaml

from market_regime.config import ExperimentConfig
from market_regime.evaluation import summarize_regimes
from market_regime.features import (
    FEATURE_COLUMNS,
    MarketDataset,
    fit_feature_scaler,
    transform_features,
)
from market_regime.neural_hmm import NeuralEmissionHMM
from market_regime.training import fit_regime_model

MODEL_MANIFEST = "model_manifest.json"
MODEL_CHECKPOINT = "live_neural_hmm.pt"
FEATURE_SCALER = "live_feature_scaler.joblib"
REGIME_SUMMARY = "live_regimes.csv"
TRANSITION_MATRIX = "live_transition_matrix.csv"
TRAINING_CONFIG = "live_config.yaml"
LATEST_FORECAST = "latest_forecast.json"
LEGACY_FORECAST = "live_forecast.json"
PREDICTION_HISTORY = "prediction_history.csv"


@dataclass(slots=True)
class LiveModelTrainingResult:
    """Metadata for one saved full-history neural HMM bundle."""

    data_cutoff: pd.Timestamp
    training_observations: int
    selected_seed: int
    training_log_likelihood_per_observation: float
    regime_summary: pd.DataFrame
    model_dir: Path


@dataclass(slots=True)
class LiveForecastResult:
    """Probabilities produced by a saved model without fitting new weights."""

    data_cutoff: pd.Timestamp
    model_data_cutoff: pd.Timestamp
    new_observations_since_training: int
    current_probabilities: pd.DataFrame
    next_session_probabilities: pd.DataFrame
    regime_summary: pd.DataFrame
    output_dir: Path


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _validated_features(dataset: MarketDataset) -> pd.DataFrame:
    features = dataset.features.loc[:, FEATURE_COLUMNS].copy()
    if features.empty:
        raise ValueError("features cannot be empty")
    if features.index.duplicated().any() or not features.index.is_monotonic_increasing:
        raise ValueError("features must have a unique, chronological index")
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("features must contain only finite values")
    return features


def _data_signature(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "data_start": pd.Timestamp(config.data.start).date().isoformat(),
        "equity_ticker": config.data.equity_ticker,
        "bond_ticker": config.data.bond_ticker,
        "vix_ticker": config.data.vix_ticker,
        "volatility_window": config.features.volatility_window,
        "volume_window": config.features.volume_window,
        "momentum_window": config.features.momentum_window,
    }


def _feature_fingerprint(features: pd.DataFrame) -> str:
    """Hash exact dates, feature order, and values used at the model cutoff."""

    digest = hashlib.sha256()
    digest.update("\x1f".join(FEATURE_COLUMNS).encode("utf-8"))
    digest.update(np.asarray(features.index.asi8, dtype="<i8").tobytes())
    values = np.ascontiguousarray(features.loc[:, FEATURE_COLUMNS].to_numpy(), dtype="<f8")
    digest.update(values.tobytes())
    return digest.hexdigest()


def _labels_from_summary(summary: pd.DataFrame, n_states: int) -> dict[int, str]:
    labels = {
        int(row.state): str(row.regime)
        for row in summary.loc[:, ["state", "regime"]].itertuples(index=False)
    }
    return {state: labels.get(state, "Trending") for state in range(n_states)}


def _probability_table(
    probabilities: np.ndarray,
    labels: dict[int, str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state": list(range(len(probabilities))),
            "regime": [labels[state] for state in range(len(probabilities))],
            "probability": probabilities,
        }
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_manifest(model_dir: Path, config: ExperimentConfig) -> dict[str, Any]:
    path = model_dir / MODEL_MANIFEST
    try:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No trained live model found at {path}; run market-regime train-live first"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read live model manifest at {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("unsupported live model manifest")
    if manifest.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("live model feature order does not match this application version")
    if manifest.get("data_signature") != _data_signature(config):
        raise ValueError(
            "live model data or feature settings do not match the current configuration; "
            "retrain the live model"
        )
    return manifest


def _bundle_file(model_dir: Path, manifest: dict[str, Any], key: str) -> Path:
    try:
        relative = Path(manifest["files"][key])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"live model manifest is missing the {key!r} file") from exc
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("live model manifest contains an unsafe file path")
    path = model_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing live model file: {path}")
    return path


def _continue_filter(
    model: NeuralEmissionHMM,
    cutoff_probabilities: np.ndarray,
    new_observations: np.ndarray,
) -> np.ndarray:
    """Advance a saved causal posterior through observations after training."""

    current = np.asarray(cutoff_probabilities, dtype=np.float64)
    if current.shape != (model.n_states,) or not np.isfinite(current).all() or (current < 0).any():
        raise ValueError("live model manifest has invalid cutoff state probabilities")
    if not np.isclose(current.sum(), 1.0):
        raise ValueError("live model cutoff state probabilities must sum to one")
    transition = model.transition_matrix_.cpu().numpy()
    if (
        transition.shape != (model.n_states, model.n_states)
        or not np.isfinite(transition).all()
        or (transition < 0).any()
        or not np.allclose(transition.sum(axis=1), 1.0)
    ):
        raise ValueError("saved neural HMM has an invalid transition matrix")
    if len(new_observations) == 0:
        return current / current.sum()

    log_emissions = model.emission_log_prob(new_observations).detach().cpu().numpy()
    tiny = np.finfo(np.float64).tiny
    for log_emission in log_emissions:
        prior = current @ transition
        log_posterior = np.log(np.clip(prior, tiny, None)) + log_emission
        log_posterior = log_posterior - float(np.max(log_posterior))
        current = np.exp(log_posterior)
        current = current / current.sum()
    return current


def train_live_model(
    dataset: MarketDataset,
    config: ExperimentConfig,
    model_dir: str | Path = "artifacts/live_model",
    *,
    verbose: bool = False,
) -> LiveModelTrainingResult:
    """Fit on all completed rows and save a versioned inference bundle."""

    config.validate()
    features = _validated_features(dataset)
    scaler = fit_feature_scaler(features)
    observations = transform_features(features, scaler).to_numpy(dtype=np.float64)

    model: NeuralEmissionHMM | None = None
    best_score = -np.inf
    selected_seed = config.seed
    for restart in range(config.model.n_restarts):
        seed = config.seed + restart
        candidate = fit_regime_model(
            "neural",
            observations,
            config,
            seed=seed,
            verbose=verbose,
        )
        score = candidate.score(observations)
        if score > best_score:
            model = candidate
            best_score = score
            selected_seed = seed
    if model is None:  # pragma: no cover - configuration validation guarantees a run
        raise RuntimeError("no live model was fitted")

    filtered = model.filter(observations).cpu().numpy()
    summary = summarize_regimes(
        filtered.argmax(axis=1),
        features["equity_return"].to_numpy(),
        features["realized_volatility"].to_numpy(),
    )
    labels = _labels_from_summary(summary, model.n_states)

    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    created_at = _utc_now()
    bundle_id = f"{features.index[-1]:%Y%m%d}-{uuid4().hex[:12]}"
    bundle = output / "bundles" / bundle_id
    bundle.mkdir(parents=True, exist_ok=False)
    joblib.dump(scaler, bundle / FEATURE_SCALER)
    model.save_checkpoint(bundle / MODEL_CHECKPOINT)
    summary.to_csv(bundle / REGIME_SUMMARY, index=False)
    pd.DataFrame(model.transition_matrix_.cpu().numpy()).to_csv(
        bundle / TRANSITION_MATRIX,
        index_label="state",
    )
    with (bundle / TRAINING_CONFIG).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

    # Stable aliases preserve the original artifact names for local users. The
    # predictor reads immutable versioned paths from the manifest instead.
    for filename in (
        FEATURE_SCALER,
        MODEL_CHECKPOINT,
        REGIME_SUMMARY,
        TRANSITION_MATRIX,
        TRAINING_CONFIG,
    ):
        shutil.copy2(bundle / filename, output / filename)

    manifest = {
        "format_version": 1,
        "bundle_id": bundle_id,
        "created_at": created_at,
        "feature_columns": list(FEATURE_COLUMNS),
        "training_feature_sha256": _feature_fingerprint(features),
        "data_signature": _data_signature(config),
        "training_data": {
            "start": pd.Timestamp(features.index[0]).isoformat(),
            "cutoff": pd.Timestamp(features.index[-1]).isoformat(),
            "observations": len(observations),
        },
        "regime_labels": labels,
        "cutoff_filtered_probabilities": filtered[-1].tolist(),
        "model": {
            **model.architecture_config(),
            "training_observations": len(observations),
            "selected_seed": selected_seed,
            "training_log_likelihood_per_observation": best_score / len(observations),
        },
        "files": {
            "checkpoint": str(Path("bundles") / bundle_id / MODEL_CHECKPOINT),
            "scaler": str(Path("bundles") / bundle_id / FEATURE_SCALER),
            "regime_summary": str(Path("bundles") / bundle_id / REGIME_SUMMARY),
            "transition_matrix": str(Path("bundles") / bundle_id / TRANSITION_MATRIX),
            "config": str(Path("bundles") / bundle_id / TRAINING_CONFIG),
        },
    }
    _atomic_write_json(output / MODEL_MANIFEST, manifest)
    return LiveModelTrainingResult(
        data_cutoff=pd.Timestamp(features.index[-1]),
        training_observations=len(observations),
        selected_seed=selected_seed,
        training_log_likelihood_per_observation=best_score / len(observations),
        regime_summary=summary,
        model_dir=output,
    )


def predict_live_regime(
    dataset: MarketDataset,
    config: ExperimentConfig,
    model_dir: str | Path = "artifacts/live_model",
    output_dir: str | Path = "artifacts/live_predictions",
) -> LiveForecastResult:
    """Load a frozen bundle and estimate the next regime without training."""

    config.validate()
    features = _validated_features(dataset)
    bundle = Path(model_dir)
    manifest = _load_manifest(bundle, config)
    training_data = manifest.get("training_data", {})
    try:
        training_start = pd.Timestamp(training_data["start"])
        model_data_cutoff = pd.Timestamp(training_data["cutoff"])
        training_observations = int(training_data["observations"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("live model manifest has invalid training-data metadata") from exc
    if features.index[0] != training_start:
        raise ValueError(
            "prediction history does not begin at the model's training start; "
            "download the complete configured history"
        )
    if model_data_cutoff not in features.index or features.index[-1] < model_data_cutoff:
        raise ValueError("prediction data ends before the model training cutoff")
    if len(features.loc[:model_data_cutoff]) != training_observations:
        raise ValueError(
            "prediction history no longer matches the model training dates; retrain the live model"
        )
    training_features = features.loc[:model_data_cutoff]
    if _feature_fingerprint(training_features) != manifest.get("training_feature_sha256"):
        raise ValueError(
            "historical features changed since this model was trained; retrain the live model"
        )

    try:
        scaler = joblib.load(_bundle_file(bundle, manifest, "scaler"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing live feature scaler in {bundle}") from exc
    if int(getattr(scaler, "n_features_in_", -1)) != len(FEATURE_COLUMNS):
        raise ValueError("saved feature scaler is incompatible with the configured features")
    model = NeuralEmissionHMM.load_checkpoint(_bundle_file(bundle, manifest, "checkpoint"))
    if model.n_features != len(FEATURE_COLUMNS):
        raise ValueError("saved neural HMM is incompatible with the configured features")

    raw_labels = manifest.get("regime_labels", {})
    try:
        labels = {state: str(raw_labels[str(state)]) for state in range(model.n_states)}
    except (KeyError, TypeError) as exc:
        raise ValueError("live model manifest has incomplete regime labels") from exc
    try:
        summary = pd.read_csv(_bundle_file(bundle, manifest, "regime_summary"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing live regime summary in {bundle}") from exc

    new_features = features.loc[features.index > model_data_cutoff]
    if new_features.empty:
        new_scaled_observations = np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float64)
    else:
        new_scaled_observations = transform_features(new_features, scaler).to_numpy(
            dtype=np.float64
        )
    current_probabilities = _continue_filter(
        model,
        np.asarray(manifest.get("cutoff_filtered_probabilities"), dtype=np.float64),
        new_scaled_observations,
    )
    transition = model.transition_matrix_.cpu().numpy()
    next_probabilities = current_probabilities @ transition
    allocation_probabilities = current_probabilities @ np.linalg.matrix_power(
        transition,
        config.backtest.execution_lag,
    )
    current_table = _probability_table(current_probabilities, labels)
    next_table = _probability_table(next_probabilities, labels)
    allocation_table = _probability_table(allocation_probabilities, labels)
    data_cutoff = pd.Timestamp(features.index[-1])
    new_observations = int((features.index > model_data_cutoff).sum())
    generated_at = _utc_now()

    payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "data_cutoff": data_cutoff.isoformat(),
        "model_data_cutoff": model_data_cutoff.isoformat(),
        "model_bundle_id": manifest.get("bundle_id"),
        "model_created_at": manifest.get("created_at"),
        "new_observations_since_training": new_observations,
        "meaning": (
            "next_session probabilities estimate the next hidden market regime, "
            "not the direction or size of the next return"
        ),
        "assets": {
            "equity": config.data.equity_ticker,
            "bond": config.data.bond_ticker,
            "volatility": config.data.vix_ticker,
        },
        "allocation_policy": {
            "equity_weights_by_regime": {
                label: (
                    config.backtest.calm_equity_weight
                    if label.lower() == "calm"
                    else config.backtest.crisis_equity_weight
                    if label.lower() == "crisis"
                    else config.backtest.neutral_equity_weight
                )
                for label in dict.fromkeys(labels.values())
            },
            "fallback_equity_weight": config.backtest.neutral_equity_weight,
            "confidence_threshold": config.backtest.confidence_threshold,
            "rebalance_frequency": config.backtest.rebalance_frequency,
            "execution_lag": config.backtest.execution_lag,
        },
        "current_regime_probabilities": current_table.to_dict(orient="records"),
        "next_session_regime_probabilities": next_table.to_dict(orient="records"),
        "allocation_horizon_regime_probabilities": allocation_table.to_dict(orient="records"),
        "model": manifest["model"],
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output / LATEST_FORECAST, payload)
    _atomic_write_json(output / LEGACY_FORECAST, payload)

    history = current_table.loc[:, ["state", "regime"]].copy()
    history.insert(0, "prediction_data_cutoff", data_cutoff.isoformat())
    history.insert(1, "model_data_cutoff", model_data_cutoff.isoformat())
    history.insert(2, "model_bundle_id", manifest.get("bundle_id"))
    history.insert(3, "generated_at", generated_at)
    history["current_probability"] = current_table["probability"]
    history["next_session_probability"] = next_table["probability"]
    history_path = output / PREDICTION_HISTORY
    if history_path.exists():
        try:
            previous = pd.read_csv(history_path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read prediction history at {history_path}") from exc
        if set(previous.columns) != set(history.columns):
            raise ValueError("existing prediction history has an incompatible schema")
        history = pd.concat([previous.loc[:, history.columns], history], ignore_index=True)
    history = (
        history.drop_duplicates(["prediction_data_cutoff", "state"], keep="last")
        .sort_values(["prediction_data_cutoff", "state"])
        .reset_index(drop=True)
    )
    _atomic_write_csv(history, history_path)

    return LiveForecastResult(
        data_cutoff=data_cutoff,
        model_data_cutoff=model_data_cutoff,
        new_observations_since_training=new_observations,
        current_probabilities=current_table,
        next_session_probabilities=next_table,
        regime_summary=summary,
        output_dir=output,
    )


def run_live_forecast(
    dataset: MarketDataset,
    config: ExperimentConfig,
    output_dir: str | Path = "artifacts/live_forecast",
    *,
    verbose: bool = False,
) -> LiveForecastResult:
    """Backward-compatible shortcut that trains and then predicts."""

    train_live_model(dataset, config, output_dir, verbose=verbose)
    return predict_live_regime(dataset, config, output_dir, output_dir)


__all__ = [
    "LiveForecastResult",
    "LiveModelTrainingResult",
    "predict_live_regime",
    "run_live_forecast",
    "train_live_model",
]
