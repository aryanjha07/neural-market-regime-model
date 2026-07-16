"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    start: str = "2005-01-01"
    end: str | None = None
    equity_ticker: str = "SPY"
    bond_ticker: str = "IEF"
    vix_ticker: str = "^VIX"
    cache_dir: str = "data/raw"


@dataclass(slots=True)
class FeatureConfig:
    volatility_window: int = 20
    volume_window: int = 20
    momentum_window: int = 20


@dataclass(slots=True)
class SplitConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15


@dataclass(slots=True)
class ModelConfig:
    n_states: int = 3
    n_mixtures: int = 3
    n_restarts: int = 3
    epochs: int = 40
    emission_steps: int = 5
    learning_rate: float = 0.01
    min_covar: float = 1e-4
    min_scale: float = 1e-3


@dataclass(slots=True)
class BacktestConfig:
    calm_equity_weight: float = 0.80
    neutral_equity_weight: float = 0.60
    crisis_equity_weight: float = 0.20
    transaction_cost_bps: float = 5.0
    confidence_threshold: float = 0.55
    rebalance_frequency: str = "weekly"
    execution_lag: int = 2


@dataclass(slots=True)
class ExperimentConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def validate(self) -> None:
        if self.model.n_states < 2:
            raise ValueError("model.n_states must be at least 2")
        if self.model.n_mixtures < 1:
            raise ValueError("model.n_mixtures must be positive")
        if self.model.n_restarts < 1:
            raise ValueError("model.n_restarts must be positive")
        if self.model.min_covar <= 0 or self.model.min_scale <= 0:
            raise ValueError("model.min_covar and model.min_scale must be positive")
        if not 0 < self.split.train_fraction < 1:
            raise ValueError("split.train_fraction must be between 0 and 1")
        if not 0 < self.split.validation_fraction < 1:
            raise ValueError("split.validation_fraction must be between 0 and 1")
        if self.split.train_fraction + self.split.validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test set")
        weights = (
            self.backtest.calm_equity_weight,
            self.backtest.neutral_equity_weight,
            self.backtest.crisis_equity_weight,
        )
        if any(not 0 <= weight <= 1 for weight in weights):
            raise ValueError("equity weights must be between 0 and 1")
        if self.backtest.rebalance_frequency not in {"daily", "weekly", "monthly"}:
            raise ValueError("rebalance_frequency must be daily, weekly, or monthly")
        if (
            isinstance(self.backtest.execution_lag, bool)
            or not isinstance(self.backtest.execution_lag, int)
            or self.backtest.execution_lag < 2
        ):
            raise ValueError(
                "backtest.execution_lag must be at least 2 when signals use finalized "
                "close data and returns are close-to-close"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_config(values: dict[str, Any]) -> ExperimentConfig:
    allowed = {"seed", "data", "features", "split", "model", "backtest"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")

    config = ExperimentConfig(
        seed=int(values.get("seed", 42)),
        data=DataConfig(**values.get("data", {})),
        features=FeatureConfig(**values.get("features", {})),
        split=SplitConfig(**values.get("split", {})),
        model=ModelConfig(**values.get("model", {})),
        backtest=BacktestConfig(**values.get("backtest", {})),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration file."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError("Configuration root must be a mapping")
    return _construct_config(values)
