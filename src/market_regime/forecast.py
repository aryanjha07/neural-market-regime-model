"""Full-history refit for a next-session regime forecast after evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
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


@dataclass(slots=True)
class LiveForecastResult:
    """Probabilities from a model refitted through the latest completed row."""

    data_cutoff: pd.Timestamp
    current_probabilities: pd.DataFrame
    next_session_probabilities: pd.DataFrame
    regime_summary: pd.DataFrame
    output_dir: Path


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


def run_live_forecast(
    dataset: MarketDataset,
    config: ExperimentConfig,
    output_dir: str | Path = "artifacts/live_forecast",
    *,
    verbose: bool = False,
) -> LiveForecastResult:
    """Refit on all completed observations and forecast the next hidden state.

    This model must not be reused to score the earlier holdout period because it
    has now seen those observations. Run the fixed-split experiment first, then
    use this function only for the operational next-session estimate.
    """

    config.validate()
    features = dataset.features.loc[:, FEATURE_COLUMNS].copy()
    if features.empty:
        raise ValueError("features cannot be empty")
    if features.index.duplicated().any() or not features.index.is_monotonic_increasing:
        raise ValueError("features must have a unique, chronological index")

    scaler = fit_feature_scaler(features)
    scaled = transform_features(features, scaler)
    observations = scaled.to_numpy(dtype=np.float64)
    model: NeuralEmissionHMM | None = None
    best_score = -np.inf
    selected_seed = config.seed
    for restart in range(config.model.n_restarts):
        seed = config.seed + restart
        torch.manual_seed(seed)
        candidate = NeuralEmissionHMM(
            n_states=config.model.n_states,
            n_features=observations.shape[1],
            n_components=config.model.n_mixtures,
            hidden_dim=16,
            min_scale=config.model.min_scale,
            random_state=seed,
        ).fit(
            observations,
            n_iter=config.model.epochs,
            learning_rate=config.model.learning_rate,
            emission_steps=config.model.emission_steps,
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
    next_probabilities = filtered[-1] @ model.transition_matrix_.cpu().numpy()
    states = filtered.argmax(axis=1)
    summary = summarize_regimes(
        states,
        features["equity_return"].to_numpy(),
        features["realized_volatility"].to_numpy(),
    )
    labels = _labels_from_summary(summary, config.model.n_states)
    current_table = _probability_table(filtered[-1], labels)
    next_table = _probability_table(next_probabilities, labels)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output / "live_feature_scaler.joblib")
    model.save_checkpoint(output / "live_neural_hmm.pt")
    summary.to_csv(output / "live_regimes.csv", index=False)
    pd.DataFrame(model.transition_matrix_.cpu().numpy()).to_csv(
        output / "live_transition_matrix.csv",
        index_label="state",
    )
    with (output / "live_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

    payload = {
        "data_cutoff": pd.Timestamp(features.index[-1]).isoformat(),
        "meaning": (
            "next_session probabilities estimate the next hidden market regime, "
            "not the direction or size of the next return"
        ),
        "current_regime_probabilities": current_table.to_dict(orient="records"),
        "next_session_regime_probabilities": next_table.to_dict(orient="records"),
        "model": {
            "n_states": model.n_states,
            "n_features": model.n_features,
            "n_components": model.n_components,
            "hidden_dim": model.hidden_dim,
            "training_observations": len(observations),
            "selected_seed": selected_seed,
            "training_log_likelihood_per_observation": best_score / len(observations),
        },
    }
    with (output / "live_forecast.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return LiveForecastResult(
        data_cutoff=pd.Timestamp(features.index[-1]),
        current_probabilities=current_table,
        next_session_probabilities=next_table,
        regime_summary=summary,
        output_dir=output,
    )


__all__ = ["LiveForecastResult", "run_live_forecast"]
