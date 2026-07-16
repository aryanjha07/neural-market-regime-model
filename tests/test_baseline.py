from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hmmlearn")

from market_regime.baseline import GaussianHMMBaseline, GaussianMixtureHMMBaseline


@pytest.fixture
def observations() -> np.ndarray:
    rng = np.random.default_rng(7)
    calm = rng.normal(loc=(-2.0, -1.5), scale=0.15, size=(60, 2))
    crisis = rng.normal(loc=(2.5, 3.0), scale=0.20, size=(60, 2))
    return np.vstack([calm, crisis, calm[:30]])


@pytest.fixture
def fitted_model(observations: np.ndarray) -> GaussianHMMBaseline:
    return GaussianHMMBaseline(
        n_states=2,
        covariance_type="diag",
        n_iter=50,
        random_state=11,
    ).fit(observations)


def test_fit_score_and_smoothed_predictions(
    fitted_model: GaussianHMMBaseline, observations: np.ndarray
) -> None:
    probabilities = fitted_model.predict_proba(observations)
    states = fitted_model.predict(observations)

    assert probabilities.shape == (len(observations), 2)
    assert states.shape == (len(observations),)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(fitted_model.score(observations))
    assert fitted_model.score(observations, per_observation=True) == pytest.approx(
        fitted_model.score(observations) / len(observations)
    )
    assert fitted_model.transition_matrix_.shape == (2, 2)
    np.testing.assert_allclose(fitted_model.transition_matrix_.sum(axis=1), 1.0)


def test_causal_filter_is_unchanged_by_future_rows(
    fitted_model: GaussianHMMBaseline, observations: np.ndarray
) -> None:
    prefix_length = 70
    prefix_probabilities = fitted_model.filter_proba(observations[:prefix_length])
    full_probabilities, log_likelihood = fitted_model.filter_proba(
        observations, return_log_likelihood=True
    )

    np.testing.assert_allclose(
        prefix_probabilities,
        full_probabilities[:prefix_length],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(full_probabilities.sum(axis=1), 1.0)
    assert np.isfinite(log_likelihood)
    np.testing.assert_allclose(log_likelihood, fitted_model.score(observations))


def test_filter_resets_at_independent_sequence_boundary(
    fitted_model: GaussianHMMBaseline, observations: np.ndarray
) -> None:
    first, second = observations[:60], observations[60:]
    combined = fitted_model.filter_proba(observations, lengths=[len(first), len(second)])

    np.testing.assert_allclose(combined[: len(first)], fitted_model.filter_proba(first))
    np.testing.assert_allclose(combined[len(first) :], fitted_model.filter_proba(second))


def test_next_state_probabilities_use_transition_matrix(
    fitted_model: GaussianHMMBaseline, observations: np.ndarray
) -> None:
    current = np.array([0.25, 0.75])
    expected = current @ fitted_model.transition_matrix_

    np.testing.assert_allclose(
        fitted_model.next_state_probabilities(state_probabilities=current), expected
    )
    from_history = fitted_model.next_state_probabilities(X=observations)
    assert from_history.shape == (2,)
    assert from_history.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bad_X, message",
    [
        (np.ones(10), "two-dimensional"),
        (np.array([[1.0], [np.nan]]), "NaN"),
        (np.empty((0, 2)), "at least one row"),
    ],
)
def test_input_validation(bad_X: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GaussianHMMBaseline(n_states=2).fit(bad_X)


def test_inference_before_fit_is_rejected(observations: np.ndarray) -> None:
    model = GaussianHMMBaseline(n_states=2)
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.filter_proba(observations)


def test_lengths_and_next_state_validation(
    fitted_model: GaussianHMMBaseline, observations: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="sum"):
        fitted_model.score(observations, lengths=[10, 10])
    with pytest.raises(ValueError, match="exactly one"):
        fitted_model.next_state_probabilities()
    with pytest.raises(ValueError, match="shape"):
        fitted_model.next_state_probabilities(state_probabilities=[1.0])
    with pytest.raises(ValueError, match="non-negative"):
        fitted_model.next_state_probabilities(state_probabilities=[-1.0, 2.0])


def test_gaussian_mixture_ablation_supports_causal_filtering(
    observations: np.ndarray,
) -> None:
    model = GaussianMixtureHMMBaseline(
        n_states=2,
        n_mixtures=2,
        n_iter=30,
        random_state=5,
    ).fit(observations)

    probabilities = model.filter_proba(observations)

    assert np.isfinite(model.score(observations))
    assert probabilities.shape == (len(observations), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
