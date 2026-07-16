import json

import numpy as np

from market_regime.config import BacktestConfig, ExperimentConfig, ModelConfig
from market_regime.experiment import run_dataset_experiment
from market_regime.features import MarketDataset
from market_regime.synthetic import generate_synthetic_market


def test_end_to_end_experiment_writes_auditable_artifacts(tmp_path) -> None:
    market = generate_synthetic_market(n_days=240, seed=19)
    config = ExperimentConfig(
        seed=19,
        model=ModelConfig(
            n_states=3,
            n_mixtures=2,
            n_restarts=1,
            epochs=2,
            emission_steps=1,
            learning_rate=0.01,
        ),
        backtest=BacktestConfig(rebalance_frequency="daily"),
    )

    result = run_dataset_experiment(
        MarketDataset(market.features, market.returns),
        config,
        tmp_path,
    )

    assert result.likelihood.n_observations == 36
    assert np.isfinite(result.likelihood.candidate_per_observation)
    assert np.isfinite(result.mixture_likelihood.baseline_per_observation)
    assert np.isclose(result.latest_regime_probabilities.sum(), 1.0)
    assert np.isclose(result.latest_next_regime_probabilities.sum(), 1.0)
    assert (tmp_path / "neural_hmm.pt").exists()
    assert (tmp_path / "regimes.png").exists()
    assert (tmp_path / "backtest.png").exists()

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert "filtered_regime_probabilities" in report["latest"]
    assert "next_session_regime_probabilities" in report["latest"]
    assert report["likelihood"]["n_observations"] == 36
    assert report["mixture_ablation"]["n_observations"] == 36
    assert set(report["validation_log_likelihood_per_observation"]) == {
        "gaussian",
        "mixture",
        "neural",
    }
