"""Command-line entry point for downloads and reproducible experiments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from market_regime.config import ExperimentConfig, load_config
from market_regime.data import download_market_data
from market_regime.experiment import ExperimentResult, run_dataset_experiment
from market_regime.features import MarketDataset, prepare_market_dataset
from market_regime.forecast import LiveForecastResult, run_live_forecast
from market_regime.synthetic import generate_synthetic_market


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-regime",
        description="Train and evaluate Gaussian and neural-emission market-regime HMMs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="run the full pipeline on deterministic synthetic data"
    )
    demo.add_argument("--config", default="configs/default.yaml")
    demo.add_argument("--days", type=int, default=1_200)
    demo.add_argument("--iterations", type=int, default=12)
    demo.add_argument("--restarts", type=int, default=1)
    demo.add_argument("--output", default="artifacts/synthetic_demo")
    demo.add_argument("--verbose", action="store_true")

    run = subparsers.add_parser(
        "run", help="download/cache daily data and run the real-market experiment"
    )
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--output", default="artifacts/market_experiment")
    run.add_argument("--refresh", action="store_true")
    run.add_argument("--verbose", action="store_true")

    forecast = subparsers.add_parser(
        "forecast",
        help="refit the neural HMM on all completed data and estimate the next regime",
    )
    forecast.add_argument("--config", default="configs/default.yaml")
    forecast.add_argument("--output", default="artifacts/live_forecast")
    forecast.add_argument("--refresh", action="store_true")
    forecast.add_argument("--verbose", action="store_true")

    download = subparsers.add_parser("download", help="download and cache market data")
    download.add_argument("--config", default="configs/default.yaml")
    download.add_argument("--refresh", action="store_true")
    return parser


def _real_dataset(config: ExperimentConfig, *, refresh: bool) -> MarketDataset:
    tickers = (
        config.data.equity_ticker,
        config.data.bond_ticker,
        config.data.vix_ticker,
    )
    market_data = download_market_data(
        tickers,
        start=config.data.start,
        end=config.data.end,
        cache_dir=config.data.cache_dir,
        refresh=refresh,
    )
    return prepare_market_dataset(
        market_data,
        equity_ticker=config.data.equity_ticker,
        bond_ticker=config.data.bond_ticker,
        vix_ticker=config.data.vix_ticker,
        volatility_window=config.features.volatility_window,
        volume_window=config.features.volume_window,
        momentum_window=config.features.momentum_window,
    )


def _print_result(result: ExperimentResult) -> None:
    comparison = result.likelihood
    print("\nHeld-out log-likelihood")
    print(f"  Gaussian HMM: {comparison.baseline_per_observation: .4f} per day")
    print(f"  Mixture HMM:  {result.mixture_likelihood.baseline_per_observation: .4f} per day")
    print(f"  Neural HMM:   {comparison.candidate_per_observation: .4f} per day")
    scores = {
        "gaussian": comparison.baseline_per_observation,
        "mixture": result.mixture_likelihood.baseline_per_observation,
        "neural": comparison.candidate_per_observation,
    }
    print(f"  Best model:   {max(scores, key=scores.get)}")
    print("\nNeural adaptive backtest")
    print(result.neural_backtest.metrics.round(4).to_string())
    print("\nLatest probabilities from the fixed evaluation model")
    print((100 * result.latest_regime_probabilities).round(1).astype(str) + "%")
    print("\nNext-session regime probabilities")
    print((100 * result.latest_next_regime_probabilities).round(1).astype(str) + "%")
    print(f"\nArtifacts: {result.output_dir.resolve()}")


def _print_forecast(result: LiveForecastResult) -> None:
    print(f"\nData cutoff: {result.data_cutoff.date()}")
    print("\nCurrent regime probabilities")
    current = result.current_probabilities.copy()
    current["probability"] = (100 * current["probability"]).round(1).astype(str) + "%"
    print(current.to_string(index=False))
    print("\nNext-session regime probabilities")
    following = result.next_session_probabilities.copy()
    following["probability"] = (100 * following["probability"]).round(1).astype(str) + "%"
    print(following.to_string(index=False))
    print("\nThis forecasts the next market regime, not whether price will rise or fall.")
    print(f"Artifacts: {result.output_dir.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "download":
            dataset = _real_dataset(config, refresh=args.refresh)
            print(
                f"Prepared {len(dataset.features):,} complete observations "
                f"from {dataset.features.index[0].date()} to "
                f"{dataset.features.index[-1].date()}."
            )
            return 0

        if args.command == "forecast":
            dataset = _real_dataset(config, refresh=args.refresh)
            forecast_result = run_live_forecast(
                dataset,
                config,
                Path(args.output),
                verbose=args.verbose,
            )
            _print_forecast(forecast_result)
            return 0

        if args.command == "demo":
            config = replace(
                config,
                model=replace(
                    config.model,
                    epochs=args.iterations,
                    n_restarts=args.restarts,
                ),
            )
            synthetic = generate_synthetic_market(n_days=args.days, seed=config.seed)
            dataset = MarketDataset(synthetic.features, synthetic.returns)
        else:
            dataset = _real_dataset(config, refresh=args.refresh)

        result = run_dataset_experiment(
            dataset,
            config,
            Path(args.output),
            verbose=args.verbose,
        )
        _print_result(result)
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
