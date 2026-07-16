"""Gaussian HMM baseline with explicit causal and smoothed inference APIs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class GaussianHMMBaseline:
    """A small, validated wrapper around :class:`hmmlearn.hmm.GaussianHMM`.

    ``predict_proba`` performs forward-backward smoothing and therefore uses the
    entire supplied sequence. It is appropriate for historical interpretation,
    but not for a live signal. Use ``filter_proba`` for causal probabilities
    based only on observations available up to each row.
    """

    _COVARIANCE_TYPES = frozenset({"spherical", "diag", "full", "tied"})

    def __init__(
        self,
        n_states: int = 3,
        *,
        covariance_type: str = "full",
        n_iter: int = 200,
        tol: float = 1e-4,
        min_covar: float = 1e-4,
        random_state: int | None = 42,
        verbose: bool = False,
    ) -> None:
        if isinstance(n_states, bool) or not isinstance(n_states, int) or n_states < 1:
            raise ValueError("n_states must be a positive integer")
        if covariance_type not in self._COVARIANCE_TYPES:
            allowed = ", ".join(sorted(self._COVARIANCE_TYPES))
            raise ValueError(f"covariance_type must be one of: {allowed}")
        if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
            raise ValueError("n_iter must be a positive integer")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a positive finite number")
        if not np.isfinite(min_covar) or min_covar <= 0:
            raise ValueError("min_covar must be a positive finite number")

        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = float(tol)
        self.min_covar = float(min_covar)
        self.random_state = random_state
        self.verbose = verbose

        self.model_: Any | None = None
        self.n_features_in_: int | None = None
        self.feature_names_in_: tuple[str, ...] | None = None

    def fit(
        self,
        X: ArrayLike,
        lengths: Sequence[int] | None = None,
    ) -> GaussianHMMBaseline:
        """Fit the Gaussian HMM to one or more independent sequences."""

        values = self._validate_X(X, fitting=True)
        checked_lengths = self._validate_lengths(lengths, len(values))
        if len(values) < self.n_states:
            raise ValueError("X must contain at least n_states observations to fit the model")

        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:  # pragma: no cover - exercised by installation
            raise ImportError(
                "GaussianHMMBaseline requires hmmlearn. Install the project "
                "dependencies before fitting the baseline."
            ) from exc

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            min_covar=self.min_covar,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        model.fit(values, lengths=checked_lengths)
        self.model_ = model
        return self

    def score(
        self,
        X: ArrayLike,
        lengths: Sequence[int] | None = None,
        *,
        per_observation: bool = False,
    ) -> float:
        """Return total (or average) sequence log-likelihood."""

        model = self._require_fitted()
        values = self._validate_X(X)
        checked_lengths = self._validate_lengths(lengths, len(values))
        total = float(model.score(values, lengths=checked_lengths))
        return total / len(values) if per_observation else total

    def score_samples(
        self,
        X: ArrayLike,
        lengths: Sequence[int] | None = None,
    ) -> tuple[float, FloatArray]:
        """Return total log-likelihood and smoothed state probabilities."""

        model = self._require_fitted()
        values = self._validate_X(X)
        checked_lengths = self._validate_lengths(lengths, len(values))
        log_likelihood, probabilities = model.score_samples(values, lengths=checked_lengths)
        return float(log_likelihood), np.asarray(probabilities, dtype=np.float64)

    def predict(
        self,
        X: ArrayLike,
        lengths: Sequence[int] | None = None,
    ) -> IntArray:
        """Return the most likely joint state path (Viterbi decoding)."""

        model = self._require_fitted()
        values = self._validate_X(X)
        checked_lengths = self._validate_lengths(lengths, len(values))
        return np.asarray(model.predict(values, lengths=checked_lengths), dtype=np.int64)

    def predict_proba(
        self,
        X: ArrayLike,
        lengths: Sequence[int] | None = None,
    ) -> FloatArray:
        """Return hindsight-smoothed probabilities using the full sequence.

        This method can use observations after row ``t`` when estimating the
        state at ``t``. Use :meth:`filter_proba` for backtests and live signals.
        """

        model = self._require_fitted()
        values = self._validate_X(X)
        checked_lengths = self._validate_lengths(lengths, len(values))
        return np.asarray(model.predict_proba(values, lengths=checked_lengths), dtype=np.float64)

    def filter_proba(
        self,
        X: ArrayLike,
        lengths: Sequence[int] | None = None,
        *,
        return_log_likelihood: bool = False,
    ) -> FloatArray | tuple[FloatArray, float]:
        """Return causal state probabilities from the forward filter.

        Each row depends only on that row and earlier rows in the same sequence.
        When ``lengths`` contains multiple sequences, the filter restarts from
        the learned initial probabilities at every sequence boundary.
        """

        model = self._require_fitted()
        values = self._validate_X(X)
        checked_lengths = self._validate_lengths(lengths, len(values))

        # hmmlearn has no public emission-density API. This stable internal
        # method is isolated here so a future hmmlearn change has one repair site.
        try:
            log_emissions = np.asarray(model._compute_log_likelihood(values), dtype=np.float64)
        except AttributeError as exc:  # pragma: no cover - version guard
            raise RuntimeError(
                "The installed hmmlearn version does not expose Gaussian "
                "emission log-likelihoods required for causal filtering"
            ) from exc

        if log_emissions.shape != (len(values), self.n_states):
            raise RuntimeError("hmmlearn returned an unexpected emission shape")

        sequence_lengths = checked_lengths if checked_lengths is not None else [len(values)]
        filtered = np.empty((len(values), self.n_states), dtype=np.float64)
        total_log_likelihood = 0.0
        start = 0
        for sequence_length in sequence_lengths:
            stop = start + sequence_length
            sequence_probs, sequence_log_likelihood = self._filter_sequence(
                log_emissions[start:stop],
                np.asarray(model.startprob_, dtype=np.float64),
                np.asarray(model.transmat_, dtype=np.float64),
            )
            filtered[start:stop] = sequence_probs
            total_log_likelihood += sequence_log_likelihood
            start = stop

        if return_log_likelihood:
            return filtered, float(total_log_likelihood)
        return filtered

    def filter_predict(
        self,
        X: ArrayLike,
        lengths: Sequence[int] | None = None,
    ) -> IntArray:
        """Return the most probable state independently at each causal step."""

        probabilities = self.filter_proba(X, lengths=lengths)
        return np.asarray(np.argmax(probabilities, axis=1), dtype=np.int64)

    def next_state_probabilities(
        self,
        X: ArrayLike | None = None,
        *,
        state_probabilities: ArrayLike | None = None,
    ) -> FloatArray:
        """Forecast the next hidden-state distribution by one transition.

        Supply either a feature history ``X`` or an explicit current-state
        probability vector. When ``X`` is used, its final *filtered* (causal)
        probability vector becomes the current state estimate.
        """

        model = self._require_fitted()
        if (X is None) == (state_probabilities is None):
            raise ValueError("supply exactly one of X or state_probabilities")

        if X is not None:
            current = np.asarray(self.filter_proba(X)[-1], dtype=np.float64)
        else:
            current = np.asarray(state_probabilities, dtype=np.float64)
            if current.ndim != 1 or current.shape[0] != self.n_states:
                raise ValueError(f"state_probabilities must have shape ({self.n_states},)")
            if not np.all(np.isfinite(current)) or np.any(current < 0):
                raise ValueError("state_probabilities must be finite and non-negative")
            total = float(current.sum())
            if total <= 0:
                raise ValueError("state_probabilities must have a positive sum")
            current = current / total

        forecast = current @ np.asarray(model.transmat_, dtype=np.float64)
        forecast_sum = float(forecast.sum())
        if not np.isfinite(forecast_sum) or forecast_sum <= 0:
            raise RuntimeError("the fitted transition matrix produced invalid probabilities")
        return np.asarray(forecast / forecast_sum, dtype=np.float64)

    @property
    def transition_matrix_(self) -> FloatArray:
        """A copy of the learned state transition matrix."""

        model = self._require_fitted()
        return np.asarray(model.transmat_, dtype=np.float64).copy()

    @property
    def start_probabilities_(self) -> FloatArray:
        """A copy of the learned initial state probabilities."""

        model = self._require_fitted()
        return np.asarray(model.startprob_, dtype=np.float64).copy()

    @property
    def converged_(self) -> bool:
        """Whether hmmlearn's EM convergence monitor reports convergence."""

        model = self._require_fitted()
        return bool(model.monitor_.converged)

    def _validate_X(self, X: ArrayLike, *, fitting: bool = False) -> FloatArray:
        feature_names: tuple[str, ...] | None = None
        columns = getattr(X, "columns", None)
        if columns is not None:
            feature_names = tuple(str(column) for column in columns)

        try:
            values = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain numeric values") from exc

        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional array of shape (time, features)")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("X must contain at least one row and one feature")
        if not np.all(np.isfinite(values)):
            raise ValueError("X contains NaN or infinite values")

        if fitting:
            self.n_features_in_ = values.shape[1]
            self.feature_names_in_ = feature_names
        else:
            self._require_fitted()
            if values.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {values.shape[1]} features, but the fitted model "
                    f"expects {self.n_features_in_}"
                )
            if (
                feature_names is not None
                and self.feature_names_in_ is not None
                and feature_names != self.feature_names_in_
            ):
                raise ValueError("X feature names or order do not match the fitted data")
        return np.ascontiguousarray(values)

    @staticmethod
    def _validate_lengths(lengths: Sequence[int] | None, n_observations: int) -> list[int] | None:
        if lengths is None:
            return None
        checked: list[int] = []
        for length in lengths:
            if isinstance(length, bool) or not isinstance(length, (int, np.integer)):
                raise ValueError("lengths must contain positive integers")
            if int(length) <= 0:
                raise ValueError("lengths must contain positive integers")
            checked.append(int(length))
        if not checked:
            raise ValueError("lengths cannot be empty")
        if sum(checked) != n_observations:
            raise ValueError("lengths must sum to the number of rows in X")
        return checked

    def _require_fitted(self) -> Any:
        if self.model_ is None:
            raise RuntimeError("fit must be called before inference")
        return self.model_

    @classmethod
    def _filter_sequence(
        cls,
        log_emissions: FloatArray,
        start_probabilities: FloatArray,
        transition_matrix: FloatArray,
    ) -> tuple[FloatArray, float]:
        probabilities = np.empty_like(log_emissions)
        log_likelihood = 0.0

        log_weights = cls._safe_log(start_probabilities) + log_emissions[0]
        probabilities[0], normalizer = cls._normalize_log_weights(log_weights)
        log_likelihood += normalizer

        for index in range(1, len(log_emissions)):
            predicted = probabilities[index - 1] @ transition_matrix
            log_weights = cls._safe_log(predicted) + log_emissions[index]
            probabilities[index], normalizer = cls._normalize_log_weights(log_weights)
            log_likelihood += normalizer

        return probabilities, float(log_likelihood)

    @staticmethod
    def _safe_log(values: FloatArray) -> FloatArray:
        with np.errstate(divide="ignore"):
            return np.where(values > 0, np.log(values), -np.inf)

    @staticmethod
    def _normalize_log_weights(log_weights: FloatArray) -> tuple[FloatArray, float]:
        maximum = float(np.max(log_weights))
        if not np.isfinite(maximum):
            raise RuntimeError("all hidden states have zero probability for an observation")
        shifted = np.exp(log_weights - maximum)
        normalizer = maximum + float(np.log(shifted.sum()))
        return shifted / shifted.sum(), normalizer


class GaussianMixtureHMMBaseline(GaussianHMMBaseline):
    """Classical GMM-HMM matched to the neural model's mixture count.

    This ablation separates gains from using Gaussian mixtures at all from gains
    attributable to their neural parameterization.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_mixtures: int = 3,
        *,
        covariance_type: str = "diag",
        n_iter: int = 200,
        tol: float = 1e-4,
        min_covar: float = 1e-4,
        random_state: int | None = 42,
        verbose: bool = False,
    ) -> None:
        if isinstance(n_mixtures, bool) or not isinstance(n_mixtures, int) or n_mixtures < 1:
            raise ValueError("n_mixtures must be a positive integer")
        super().__init__(
            n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            min_covar=min_covar,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_mixtures = n_mixtures

    def fit(
        self,
        X: ArrayLike,
        lengths: Sequence[int] | None = None,
    ) -> GaussianMixtureHMMBaseline:
        """Fit a classical per-state diagonal Gaussian-mixture HMM."""

        values = self._validate_X(X, fitting=True)
        checked_lengths = self._validate_lengths(lengths, len(values))
        if len(values) < self.n_states * self.n_mixtures:
            raise ValueError("X is too short for the requested states and mixtures")

        try:
            from hmmlearn.hmm import GMMHMM
        except ImportError as exc:  # pragma: no cover - installation guard
            raise ImportError("GaussianMixtureHMMBaseline requires hmmlearn") from exc

        model = GMMHMM(
            n_components=self.n_states,
            n_mix=self.n_mixtures,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            min_covar=self.min_covar,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        model.fit(values, lengths=checked_lengths)
        self.model_ = model
        return self


__all__ = ["GaussianHMMBaseline", "GaussianMixtureHMMBaseline"]
