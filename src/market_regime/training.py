"""Shared model construction and validation-based restart selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from market_regime.baseline import GaussianHMMBaseline, GaussianMixtureHMMBaseline
from market_regime.config import ExperimentConfig
from market_regime.evaluation import conditional_log_likelihood
from market_regime.neural_hmm import NeuralEmissionHMM

MODEL_NAMES = ("gaussian", "mixture", "neural")


@dataclass(slots=True)
class ModelSelection:
    models: dict[str, Any]
    validation_scores: dict[str, float]
    selected_seeds: dict[str, int]


def fit_regime_model(
    name: str,
    observations: np.ndarray,
    config: ExperimentConfig,
    *,
    seed: int,
    verbose: bool = False,
) -> Any:
    """Fit one configured model with a deterministic seed."""

    if name == "gaussian":
        return GaussianHMMBaseline(
            n_states=config.model.n_states,
            covariance_type="full",
            min_covar=config.model.min_covar,
            random_state=seed,
        ).fit(observations)
    if name == "mixture":
        return GaussianMixtureHMMBaseline(
            n_states=config.model.n_states,
            n_mixtures=config.model.n_mixtures,
            covariance_type="diag",
            min_covar=config.model.min_covar,
            random_state=seed,
        ).fit(observations)
    if name == "neural":
        torch.manual_seed(seed)
        return NeuralEmissionHMM(
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
    raise ValueError(f"unknown model name: {name}")


def select_models_by_validation(
    train_observations: np.ndarray,
    score_context: np.ndarray,
    *,
    validation_start: int,
    validation_stop: int,
    config: ExperimentConfig,
    verbose: bool = False,
    progress_prefix: str = "",
) -> ModelSelection:
    """Choose each model's restart using conditional validation likelihood."""

    selected: dict[str, tuple[Any, float, int]] = {}
    for restart in range(config.model.n_restarts):
        seed = config.seed + restart
        for name in MODEL_NAMES:
            candidate = fit_regime_model(
                name,
                train_observations,
                config,
                seed=seed,
                verbose=verbose,
            )
            validation_score = conditional_log_likelihood(
                candidate,
                score_context,
                start=validation_start,
                stop=validation_stop,
            )
            if verbose:
                per_observation = validation_score / (validation_stop - validation_start)
                print(
                    f"{progress_prefix}restart={restart + 1} model={name} "
                    f"validation_nats_per_day={per_observation:.6f}"
                )
            if name not in selected or validation_score > selected[name][1]:
                selected[name] = (candidate, validation_score, seed)

    return ModelSelection(
        models={name: selected[name][0] for name in MODEL_NAMES},
        validation_scores={name: selected[name][1] for name in MODEL_NAMES},
        selected_seeds={name: selected[name][2] for name in MODEL_NAMES},
    )


__all__ = [
    "MODEL_NAMES",
    "ModelSelection",
    "fit_regime_model",
    "select_models_by_validation",
]
