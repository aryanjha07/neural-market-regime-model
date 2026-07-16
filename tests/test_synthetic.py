import numpy as np

from market_regime.synthetic import generate_synthetic_market


def test_synthetic_market_is_reproducible_and_aligned() -> None:
    first = generate_synthetic_market(n_days=300, seed=7)
    second = generate_synthetic_market(n_days=300, seed=7)

    assert first.features.index.equals(first.returns.index)
    assert first.features.index.equals(first.states.index)
    assert len(first.features) == 300
    assert np.allclose(first.features.to_numpy(), second.features.to_numpy())
    assert set(first.states.unique()).issubset({0, 1, 2})


def test_synthetic_crisis_is_more_volatile_than_calm() -> None:
    market = generate_synthetic_market(n_days=3_000, seed=11)
    calm = market.features.loc[market.states.eq(0), "realized_volatility"]
    crisis = market.features.loc[market.states.eq(2), "realized_volatility"]

    assert crisis.mean() > calm.mean() * 2
