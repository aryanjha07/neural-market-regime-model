import json

import numpy as np

from market_regime.config import ExperimentConfig, ModelConfig
from market_regime.features import MarketDataset
from market_regime.forecast import run_live_forecast
from market_regime.synthetic import generate_synthetic_market


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
    assert (tmp_path / "live_neural_hmm.pt").exists()

    payload = json.loads((tmp_path / "live_forecast.json").read_text(encoding="utf-8"))
    assert payload["model"]["training_observations"] == len(market.features)
    assert "not the direction" in payload["meaning"]
