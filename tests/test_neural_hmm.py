"""Focused tests for the neural-emission HMM."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from market_regime.neural_hmm import NeuralEmissionHMM


def _synthetic_regime_series(length: int = 180, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    transition = torch.tensor([[0.96, 0.04], [0.08, 0.92]], dtype=torch.float64)
    means = torch.tensor([[-2.0, -0.7], [2.1, 0.9]], dtype=torch.float64)
    scales = torch.tensor([[0.45, 0.30], [0.55, 0.35]], dtype=torch.float64)

    state = 0
    observations = []
    for _ in range(length):
        observations.append(means[state] + scales[state] * torch.randn(2, generator=generator))
        state = int(torch.multinomial(transition[state], 1, generator=generator).item())
    return torch.stack(observations)


def test_log_space_inference_is_finite_and_normalized() -> None:
    model = NeuralEmissionHMM(
        n_states=3,
        n_features=2,
        n_components=2,
        hidden_dim=8,
        random_state=11,
    )
    observations = torch.tensor([[-1.0e4, 1.0e4], [0.0, 0.0], [1.0e4, -1.0e4]], dtype=torch.float64)

    likelihood = model.log_likelihood(observations)
    filtered = model.filter(observations)
    smoothed = model.posterior(observations)

    assert torch.isfinite(likelihood)
    assert torch.isfinite(filtered).all()
    assert torch.isfinite(smoothed).all()
    assert torch.allclose(filtered.sum(dim=1), torch.ones(3, dtype=torch.float64))
    assert torch.allclose(smoothed.sum(dim=1), torch.ones(3, dtype=torch.float64))


def test_accepts_pandas_feature_frames() -> None:
    model = NeuralEmissionHMM(n_states=2, n_features=2, n_components=2)
    frame = pd.DataFrame({"return": [0.1, -0.2], "volatility": [0.3, 0.4]})

    assert model.filter(frame).shape == (2, 2)


def test_generalized_em_improves_fit_and_learns_stochastic_matrices() -> None:
    observations = _synthetic_regime_series()
    model = NeuralEmissionHMM(
        n_states=2,
        n_features=2,
        n_components=2,
        hidden_dim=8,
        random_state=5,
    )

    model.fit(
        observations,
        n_iter=12,
        emission_steps=3,
        learning_rate=0.02,
        tol=0.0,
    )

    history = model.history_["log_likelihood"]
    assert len(history) == model.n_iter_ + 1
    assert history[-1] > history[0]
    assert torch.isfinite(torch.tensor(history)).all()
    assert torch.allclose(model.start_probabilities_.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(model.transition_matrix_.sum(dim=1), torch.ones(2, dtype=torch.float64))
    assert (model.transition_matrix_ >= 0).all()


def test_next_state_prediction_uses_only_last_filter_and_transition() -> None:
    observations = _synthetic_regime_series(length=60)
    model = NeuralEmissionHMM(
        n_states=2,
        n_features=2,
        n_components=2,
        hidden_dim=6,
        random_state=13,
    ).fit(observations, n_iter=4, emission_steps=2, tol=0.0)

    causal_probabilities = model.filter(observations)
    expected = causal_probabilities[-1] @ model.transition_matrix_

    assert torch.allclose(model.predict_next_state(observations), expected)
    assert torch.allclose(causal_probabilities[:-1], model.filter(observations[:-1]), atol=1e-10)


def test_seeded_refit_is_deterministic_without_warm_start() -> None:
    observations = _synthetic_regime_series(length=80, seed=19)
    model = NeuralEmissionHMM(
        n_states=2,
        n_features=2,
        n_components=2,
        hidden_dim=6,
        random_state=23,
    )
    fit_options = {
        "n_iter": 3,
        "emission_steps": 2,
        "learning_rate": 0.01,
        "tol": 0.0,
    }

    model.fit(observations, **fit_options)
    first_history = torch.tensor(model.history_["log_likelihood"])
    first_transition = model.transition_matrix_

    model.fit(observations, **fit_options)

    assert torch.equal(torch.tensor(model.history_["log_likelihood"]), first_history)
    assert torch.equal(model.transition_matrix_, first_transition)


def test_checkpoint_round_trip_preserves_probabilities(tmp_path) -> None:
    observations = _synthetic_regime_series(length=80)
    model = NeuralEmissionHMM(
        n_states=2,
        n_features=2,
        n_components=2,
        hidden_dim=6,
        min_scale=0.007,
        random_state=17,
    ).fit(observations, n_iter=2, emission_steps=1, tol=0.0)

    path = model.save_checkpoint(tmp_path / "model.pt")
    restored = NeuralEmissionHMM.load_checkpoint(path)

    assert restored.emission.min_scale == model.emission.min_scale
    assert restored.architecture_config() == model.architecture_config()
    assert torch.allclose(restored.filter(observations), model.filter(observations))
    assert torch.allclose(
        restored.predict_next_state(observations),
        model.predict_next_state(observations),
    )


@pytest.mark.parametrize("seed", [1, 3, 7])
def test_training_history_never_decreases(seed: int) -> None:
    observations = _synthetic_regime_series(length=100, seed=seed)
    model = NeuralEmissionHMM(
        n_states=2,
        n_features=2,
        n_components=2,
        hidden_dim=6,
        random_state=seed,
    ).fit(
        observations,
        n_iter=5,
        emission_steps=3,
        learning_rate=0.02,
        tol=0.0,
    )

    differences = torch.diff(torch.tensor(model.history_["log_likelihood"]))
    assert (differences >= 0).all()
