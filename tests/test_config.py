from pathlib import Path

import pytest

from market_regime.config import load_config


def test_default_config_loads() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.model.n_states == 3
    assert config.model.n_mixtures == 3
    assert config.data.equity_ticker == "SPY"
    assert config.backtest.execution_lag == 2
    assert config.split.train_fraction + config.split.validation_fraction < 1


def test_invalid_split_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "split:\n  train_fraction: 0.9\n  validation_fraction: 0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="leave a test set"):
        load_config(path)
