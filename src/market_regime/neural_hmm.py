"""Neural mixture-emission hidden Markov model.

The discrete HMM parameters are updated with expected sufficient statistics,
while a small state-conditioned mixture-density network is optimized with
gradient descent.  All HMM recursions operate in log space.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MixtureDensityEmission(nn.Module):
    """Diagonal Gaussian-mixture emissions conditioned on a discrete state.

    A learned state embedding and MLP produce offsets from deterministic,
    data-informed mixture parameters.  The offsets make the emission model a
    compact MDN while retaining a reliable initialization for short financial
    time series.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        n_components: int,
        hidden_dim: int,
        min_scale: float,
        random_state: int,
    ) -> None:
        super().__init__()
        if n_states < 1:
            raise ValueError("n_states must be at least 1")
        if n_features < 1:
            raise ValueError("n_features must be at least 1")
        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")
        if min_scale <= 0:
            raise ValueError("min_scale must be positive")

        self.n_states = n_states
        self.n_features = n_features
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.min_scale = float(min_scale)
        self.random_state = int(random_state)
        output_dim = n_components * (1 + 2 * n_features)

        self.state_embedding = nn.Embedding(n_states, hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.register_buffer("base_component_logits", torch.zeros(n_states, n_components))
        self.register_buffer("base_means", torch.zeros(n_states, n_components, n_features))
        initial_raw_scale = self._inverse_softplus(torch.tensor(1.0 - min_scale))
        self.register_buffer(
            "base_raw_scales",
            torch.full((n_states, n_components, n_features), initial_raw_scale.item()),
        )
        self.reset_parameters()

    @staticmethod
    def _inverse_softplus(value: Tensor) -> Tensor:
        value = value.clamp_min(torch.finfo(value.dtype).eps)
        return value + torch.log(-torch.expm1(-value))

    def reset_parameters(self) -> None:
        """Reset neural offsets deterministically."""

        devices = []
        if self.state_embedding.weight.is_cuda:
            devices = [self.state_embedding.weight.device]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.random_state)
            nn.init.normal_(self.state_embedding.weight, mean=0.0, std=0.2)
            first = self.network[0]
            last = self.network[2]
            assert isinstance(first, nn.Linear)
            assert isinstance(last, nn.Linear)
            nn.init.xavier_uniform_(first.weight)
            nn.init.zeros_(first.bias)
            # Starting at zero preserves the robust data-informed base model.
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @torch.no_grad()
    def initialize_from_data(self, observations: Tensor) -> None:
        """Initialize state/component locations from deterministic quantiles."""

        if observations.ndim != 2 or observations.shape[1] != self.n_features:
            raise ValueError(f"observations must have shape (time, {self.n_features})")

        n_centers = self.n_states * self.n_components
        order = torch.argsort(observations[:, 0], stable=True)
        quantiles = (
            torch.arange(n_centers, device=observations.device, dtype=observations.dtype) + 0.5
        ) / n_centers
        indices = torch.round(quantiles * (len(order) - 1)).long()
        centers = observations[order[indices]].reshape(
            self.n_states, self.n_components, self.n_features
        )

        feature_scale = observations.std(dim=0, unbiased=False).clamp_min(self.min_scale * 10.0)
        component_scale = (feature_scale / max(n_centers**0.5, 1.0)).clamp_min(self.min_scale * 2.0)
        raw_scale = self._inverse_softplus(component_scale - self.min_scale)

        self.base_component_logits.zero_()
        self.base_means.copy_(centers)
        self.base_raw_scales.copy_(raw_scale.view(1, 1, -1).expand_as(self.base_raw_scales))

    def _raw_mixture_parameters(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return component logits, means, and positive diagonal scales."""

        state_ids = torch.arange(self.n_states, device=self.base_means.device)
        raw = self.network(self.state_embedding(state_ids))
        logits_width = self.n_components
        values_width = self.n_components * self.n_features

        logits_offset = raw[:, :logits_width]
        means_offset = raw[:, logits_width : logits_width + values_width].reshape(
            self.n_states, self.n_components, self.n_features
        )
        scales_offset = raw[:, logits_width + values_width :].reshape(
            self.n_states, self.n_components, self.n_features
        )

        component_logits = self.base_component_logits + logits_offset
        means = self.base_means + means_offset
        scales = F.softplus(self.base_raw_scales + scales_offset) + self.min_scale
        return component_logits, means, scales

    def mixture_parameters(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return mixture weights, means, and positive diagonal scales."""

        component_logits, means, scales = self._raw_mixture_parameters()
        return F.softmax(component_logits, dim=-1), means, scales

    def log_prob(self, observations: Tensor) -> Tensor:
        """Return ``log p(x_t | z_t=k)`` with shape ``(time, states)``."""

        component_logits, means, scales = self._raw_mixture_parameters()
        standardized = (observations[:, None, None, :] - means[None, :, :, :]) / scales[
            None, :, :, :
        ]
        log_component_density = -0.5 * (
            standardized.square() + 2.0 * torch.log(scales)[None, :, :, :] + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        log_weights = F.log_softmax(component_logits, dim=-1)
        return torch.logsumexp(log_component_density + log_weights[None, :, :], dim=-1)


class NeuralEmissionHMM(nn.Module):
    """HMM with state-conditioned neural Gaussian-mixture emissions.

    Parameters are intentionally small and the implementation handles one
    contiguous sequence at a time.  ``filter`` is causal and is therefore the
    appropriate API for backtests and live signals.  ``posterior`` uses both
    past and future observations and is intended for training diagnostics.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        n_components: int = 3,
        hidden_dim: int = 16,
        *,
        min_scale: float = 1e-3,
        transition_smoothing: float = 1e-2,
        initial_persistence: float = 0.90,
        random_state: int = 42,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if n_states < 1:
            raise ValueError("n_states must be at least 1")
        if not 0.0 <= initial_persistence <= 1.0:
            raise ValueError("initial_persistence must be between 0 and 1")
        if transition_smoothing <= 0:
            raise ValueError("transition_smoothing must be positive")

        self.n_states = int(n_states)
        self.n_features = int(n_features)
        self.n_components = int(n_components)
        self.hidden_dim = int(hidden_dim)
        self.transition_smoothing = float(transition_smoothing)
        self.initial_persistence = float(initial_persistence)
        self.random_state = int(random_state)

        initial_probs = torch.full((n_states,), 1.0 / n_states, dtype=dtype)
        if n_states == 1:
            transition_matrix = torch.ones((1, 1), dtype=dtype)
        else:
            off_diagonal = (1.0 - initial_persistence) / (n_states - 1)
            transition_matrix = torch.full((n_states, n_states), off_diagonal, dtype=dtype)
            transition_matrix.fill_diagonal_(initial_persistence)

        self.register_buffer("initial_probabilities", initial_probs)
        self.register_buffer("transition_matrix", transition_matrix)
        self.emission = MixtureDensityEmission(
            n_states=n_states,
            n_features=n_features,
            n_components=n_components,
            hidden_dim=hidden_dim,
            min_scale=min_scale,
            random_state=random_state,
        )
        self.to(device=device, dtype=dtype)

        self.history_: dict[str, list[float]] = {
            "log_likelihood": [],
            "emission_loss": [],
        }
        self.n_iter_ = 0
        self.converged_ = False
        self._emission_initialized = False

    @property
    def start_probabilities_(self) -> Tensor:
        """Learned initial-state probabilities."""

        return self.initial_probabilities.detach().clone()

    @property
    def transition_matrix_(self) -> Tensor:
        """Learned row-stochastic state transition matrix."""

        return self.transition_matrix.detach().clone()

    def architecture_config(self) -> dict[str, Any]:
        """Return every constructor value needed for equivalent inference."""

        parameter = next(self.emission.parameters())
        return {
            "n_states": self.n_states,
            "n_features": self.n_features,
            "n_components": self.n_components,
            "hidden_dim": self.hidden_dim,
            "min_scale": self.emission.min_scale,
            "transition_smoothing": self.transition_smoothing,
            "initial_persistence": self.initial_persistence,
            "random_state": self.random_state,
            "dtype": str(parameter.dtype).removeprefix("torch."),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save architecture and parameters in one reconstructable checkpoint."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "architecture": self.architecture_config(),
                "state_dict": self.state_dict(),
            },
            destination,
        )
        return destination

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> NeuralEmissionHMM:
        """Load a checkpoint written by :meth:`save_checkpoint`."""

        payload = torch.load(
            Path(path),
            map_location=device or "cpu",
            weights_only=True,
        )
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("unsupported neural HMM checkpoint")
        architecture = dict(payload.get("architecture", {}))
        dtype_name = architecture.pop("dtype", None)
        dtype = getattr(torch, str(dtype_name), None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError("checkpoint contains an unsupported dtype")
        model = cls(**architecture, dtype=dtype, device=device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model._emission_initialized = True
        return model

    @property
    def transmat_(self) -> Tensor:
        """Compatibility alias for the learned transition matrix."""

        return self.transition_matrix_

    def _coerce_observations(self, observations: Any) -> Tensor:
        parameter = next(self.emission.parameters())
        if callable(getattr(observations, "to_numpy", None)):
            observations = observations.to_numpy()
        try:
            result = torch.as_tensor(observations, dtype=parameter.dtype, device=parameter.device)
        except (TypeError, ValueError) as exc:
            raise TypeError("observations must be array-like numeric data") from exc
        if result.ndim == 1 and self.n_features == 1:
            result = result[:, None]
        if result.ndim != 2 or result.shape[1] != self.n_features:
            raise ValueError(
                f"observations must have shape (time, {self.n_features}); "
                f"received {tuple(result.shape)}"
            )
        if result.shape[0] < 1:
            raise ValueError("observations cannot be empty")
        if not torch.isfinite(result).all():
            raise ValueError("observations must contain only finite values")
        return result

    def emission_log_prob(self, observations: Any) -> Tensor:
        """Return per-time, per-state emission log probabilities."""

        return self.emission.log_prob(self._coerce_observations(observations))

    def _log_hmm_parameters(self) -> tuple[Tensor, Tensor]:
        tiny = torch.finfo(self.initial_probabilities.dtype).tiny
        log_initial = torch.log(self.initial_probabilities.clamp_min(tiny))
        log_transition = torch.log(self.transition_matrix.clamp_min(tiny))
        return log_initial, log_transition

    @torch.no_grad()
    def _reset_discrete_parameters(self) -> None:
        self.initial_probabilities.fill_(1.0 / self.n_states)
        if self.n_states == 1:
            self.transition_matrix.fill_(1.0)
            return
        off_diagonal = (1.0 - self.initial_persistence) / (self.n_states - 1)
        self.transition_matrix.fill_(off_diagonal)
        self.transition_matrix.fill_diagonal_(self.initial_persistence)

    def _forward_pass(self, log_emission: Tensor) -> tuple[Tensor, Tensor]:
        log_initial, log_transition = self._log_hmm_parameters()
        alpha_rows = [log_initial + log_emission[0]]
        for time_index in range(1, log_emission.shape[0]):
            previous = alpha_rows[-1][:, None] + log_transition
            alpha_rows.append(log_emission[time_index] + torch.logsumexp(previous, dim=0))
        alpha = torch.stack(alpha_rows)
        return alpha, torch.logsumexp(alpha[-1], dim=0)

    def _backward_pass(self, log_emission: Tensor) -> Tensor:
        _, log_transition = self._log_hmm_parameters()
        next_beta = torch.zeros(self.n_states, dtype=log_emission.dtype, device=log_emission.device)
        reversed_rows = [next_beta]
        for time_index in range(log_emission.shape[0] - 2, -1, -1):
            next_beta = torch.logsumexp(
                log_transition + log_emission[time_index + 1][None, :] + next_beta[None, :],
                dim=1,
            )
            reversed_rows.append(next_beta)
        return torch.stack(list(reversed(reversed_rows)))

    def _forward_backward(self, log_emission: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        alpha, log_likelihood = self._forward_pass(log_emission)
        beta = self._backward_pass(log_emission)
        gamma = F.softmax(alpha + beta, dim=1)

        transition_counts = torch.zeros_like(self.transition_matrix)
        if log_emission.shape[0] > 1:
            _, log_transition = self._log_hmm_parameters()
            for time_index in range(log_emission.shape[0] - 1):
                log_xi = (
                    alpha[time_index][:, None]
                    + log_transition
                    + log_emission[time_index + 1][None, :]
                    + beta[time_index + 1][None, :]
                )
                log_xi = log_xi - torch.logsumexp(log_xi.reshape(-1), dim=0)
                transition_counts += torch.exp(log_xi)
        return gamma, transition_counts, log_likelihood

    def forward(self, observations: Any) -> Tensor:
        """Return the differentiable total log-likelihood of one sequence."""

        log_emission = self.emission_log_prob(observations)
        _, log_likelihood = self._forward_pass(log_emission)
        return log_likelihood

    def log_likelihood(self, observations: Any) -> Tensor:
        """Return the total sequence log-likelihood as a scalar tensor."""

        return self.forward(observations)

    @torch.no_grad()
    def score(self, observations: Any) -> float:
        """Return the total sequence log-likelihood as a Python float."""

        return float(self.log_likelihood(observations).item())

    @torch.no_grad()
    def filter(self, observations: Any) -> Tensor:
        """Return causal ``p(z_t | x_1, ..., x_t)`` probabilities."""

        log_emission = self.emission_log_prob(observations)
        alpha, _ = self._forward_pass(log_emission)
        return F.softmax(alpha, dim=1)

    filtering_probabilities = filter

    @torch.no_grad()
    def posterior(self, observations: Any) -> Tensor:
        """Return smoothed ``p(z_t | x_1, ..., x_T)`` probabilities.

        This method uses future observations and must not drive historical
        trading decisions.
        """

        log_emission = self.emission_log_prob(observations)
        gamma, _, _ = self._forward_backward(log_emission)
        return gamma

    @torch.no_grad()
    def predict_next_state(self, observations: Any) -> Tensor:
        """Return tomorrow's state distribution given data through today."""

        filtered_today = self.filter(observations)[-1]
        prediction = filtered_today @ self.transition_matrix
        return prediction / prediction.sum()

    predict_next_state_proba = predict_next_state

    @torch.no_grad()
    def predict_states(self, observations: Any, *, smoothed: bool = False) -> Tensor:
        """Return the most probable state index for every observation."""

        probabilities = self.posterior(observations) if smoothed else self.filter(observations)
        return probabilities.argmax(dim=1)

    @torch.no_grad()
    def _update_discrete_parameters(
        self, gamma: Tensor, transition_counts: Tensor, sequence_length: int
    ) -> None:
        initial_counts = gamma[0] + self.transition_smoothing
        self.initial_probabilities.copy_(initial_counts / initial_counts.sum())

        if sequence_length > 1:
            counts = transition_counts + self.transition_smoothing
            row_totals = counts.sum(dim=1, keepdim=True)
            self.transition_matrix.copy_(counts / row_totals.clamp_min(1e-12))

    def fit(
        self,
        observations: Any,
        *,
        n_iter: int = 50,
        learning_rate: float = 1e-2,
        emission_steps: int = 5,
        tol: float = 1e-4,
        min_iter: int = 3,
        gradient_clip: float = 10.0,
        warm_start: bool = False,
        verbose: bool = False,
    ) -> NeuralEmissionHMM:
        """Fit one sequence with generalized EM.

        Each iteration runs forward-backward, analytically updates the initial
        and transition probabilities, then takes full-batch gradient steps on
        the posterior-weighted emission objective.
        """

        if n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        if emission_steps < 1:
            raise ValueError("emission_steps must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if tol < 0:
            raise ValueError("tol cannot be negative")
        if min_iter < 1:
            raise ValueError("min_iter must be at least 1")

        x = self._coerce_observations(observations)
        if not warm_start:
            self._reset_discrete_parameters()
            self.emission.reset_parameters()
            self.emission.initialize_from_data(x)
            self._emission_initialized = True
        elif not self._emission_initialized:
            self.emission.initialize_from_data(x)
            self._emission_initialized = True

        self.history_ = {"log_likelihood": [], "emission_loss": []}
        self.n_iter_ = 0
        self.converged_ = False
        optimizer = torch.optim.Adam(self.emission.parameters(), lr=learning_rate)

        with torch.no_grad():
            initial_score = self.log_likelihood(x).item()
        self.history_["log_likelihood"].append(float(initial_score))

        for iteration in range(n_iter):
            previous_score = self.history_["log_likelihood"][-1]
            previous_initial = self.initial_probabilities.detach().clone()
            previous_transition = self.transition_matrix.detach().clone()
            previous_emission = {
                name: value.detach().clone() for name, value in self.emission.state_dict().items()
            }
            with torch.no_grad():
                log_emission = self.emission.log_prob(x)
                gamma, transition_counts, _ = self._forward_backward(log_emission)
                self._update_discrete_parameters(gamma, transition_counts, len(x))

            emission_loss_value = math.nan
            fixed_gamma = gamma.detach()
            for _ in range(emission_steps):
                optimizer.zero_grad(set_to_none=True)
                emission_log_prob = self.emission.log_prob(x)
                emission_loss = -(fixed_gamma * emission_log_prob).sum() / len(x)
                if not torch.isfinite(emission_loss):
                    raise RuntimeError(
                        "non-finite emission loss; scale the input features or increase min_scale"
                    )
                emission_loss.backward()
                nn.utils.clip_grad_norm_(self.emission.parameters(), gradient_clip)
                optimizer.step()
                emission_loss_value = float(emission_loss.detach().item())

            with torch.no_grad():
                current_score = float(self.log_likelihood(x).item())
            accepted = current_score >= previous_score
            if not accepted:
                with torch.no_grad():
                    self.initial_probabilities.copy_(previous_initial)
                    self.transition_matrix.copy_(previous_transition)
                    self.emission.load_state_dict(previous_emission, strict=True)
                current_score = previous_score
                for group in optimizer.param_groups:
                    group["lr"] = float(group["lr"]) * 0.5
            self.history_["emission_loss"].append(emission_loss_value)
            self.history_["log_likelihood"].append(current_score)
            self.n_iter_ = iteration + 1

            if verbose:
                print(
                    f"iteration={iteration + 1:03d} "
                    f"log_likelihood={current_score:.6f} "
                    f"emission_loss={emission_loss_value:.6f} "
                    f"accepted={accepted}"
                )

            improvement = current_score - previous_score
            scale = 1.0 + abs(previous_score)
            if accepted and iteration + 1 >= min_iter and improvement <= tol * scale:
                self.converged_ = True
                break

        return self


# A concise alias is convenient in notebooks while the longer name remains
# explicit in reports and serialized model metadata.
NeuralHMM = NeuralEmissionHMM


__all__: Sequence[str] = (
    "MixtureDensityEmission",
    "NeuralEmissionHMM",
    "NeuralHMM",
)
