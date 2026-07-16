# Market Regime Detection with a Neural-Emission HMM

This project studies the different "moods" of the financial market.

The model looks for hidden market conditions such as:

- **Calm:** small price movements and low volatility
- **Trending:** prices move more strongly in one direction
- **Crisis:** large price movements, high volatility, and market stress

These states are not provided as labels. The models discover them from market
data, so this is an **unsupervised machine-learning project**.

This project estimates the probability of the next market regime. It does not
promise to predict tomorrow's exact return or guarantee a profitable trade.

## What The Project Does

The complete pipeline can:

1. Download daily SPY, IEF, and VIX data.
2. Clean and validate the downloaded data.
3. Create market features using only past and present information.
4. Train three market-regime models.
5. Compare the models on unseen historical data.
6. Estimate current and next-session regime probabilities.
7. Test a regime-based stock and bond allocation strategy.
8. Compare that strategy with a normal 60/40 portfolio.
9. Save models, charts, probabilities, and performance reports.

## Data And Features

The default assets are:

| Symbol | Meaning |
|---|---|
| `SPY` | US stock-market ETF |
| `IEF` | US Treasury-bond ETF |
| `^VIX` | Market volatility or "fear" index |

The model uses five daily features:

| Feature | Simple meaning |
|---|---|
| Equity return | How much SPY changed |
| Realized volatility | How unstable SPY has recently been |
| Volume z-score | Whether trading volume is unusually high |
| VIX change | Whether market fear increased or decreased |
| Momentum | Whether SPY has recently moved up or down |

Adjusted prices are used when available so dividends and corporate actions do
not appear as false market movements.

## Models

The project compares three models.

### 1. Gaussian HMM

This is the simplest baseline. Each hidden market state is represented by one
bell-shaped Gaussian distribution.

### 2. Gaussian-Mixture HMM

This stronger baseline uses several Gaussian distributions inside each state.
It can represent more complicated market behavior than one Gaussian.

### 3. Neural-Parameterized HMM

A small PyTorch network produces the mixture weights, means, and scales for
each hidden state. The HMM transition matrix remains visible and interpretable.

The current neural model has the same basic distribution family as the
Gaussian-mixture HMM. That is why the mixture model is included as a fair
comparison. We do not automatically claim that the neural model is better.

## Experiment Flow

```text
Market data
    -> Clean and validate it
    -> Create features
    -> Split data by time
    -> Scale using training data only
    -> Train all three models
    -> Choose the best restart on validation data
    -> Evaluate once on unseen test data
    -> Run the allocation backtest
    -> Save reports and charts
```

The default chronological split is:

```text
Oldest 70%  -> Training
Next 15%    -> Validation
Newest 15%  -> Final test
```

Market data is never randomly shuffled.

## Installation

Python 3.11 through 3.13 is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

When opening a new terminal later, activate the environment again:

```bash
source .venv/bin/activate
```

## Commands

Run all automated tests:

```bash
pytest
```

Run a synthetic demonstration without downloading market data:

```bash
market-regime demo --days 1200 --iterations 12
```

Download and check real market data:

```bash
market-regime download --config configs/default.yaml
```

Run the complete historical experiment:

```bash
market-regime run --config configs/default.yaml
```

Show detailed training progress:

```bash
market-regime run --config configs/default.yaml --verbose
```

Refit the neural model on all completed data and estimate the next regime:

```bash
market-regime forecast --config configs/default.yaml
```

The complete real experiment can take several minutes on a laptop.

## What The Forecast Means

The forecast may look like this:

```text
Calm:      60%
Trending:  25%
Crisis:    15%
```

This means the model believes the next hidden market condition is most likely
calm. It does not mean the market has a 60% chance of going up.

Evaluation and live forecasting are separate commands:

- `market-regime run` protects the test period for honest evaluation.
- `market-regime forecast` trains on all completed data for a fresh estimate.

## Allocation Backtest

The example portfolio changes its stock exposure by regime:

| Regime | Stocks | Bonds |
|---|---:|---:|
| Calm | 80% | 20% |
| Trending | 60% | 40% |
| Crisis | 20% | 80% |

The strategy is compared with a static portfolio containing 60% stocks and 40%
bonds.

The backtest reports:

- Total and annualized return
- Volatility
- Sharpe ratio
- Maximum drawdown
- Turnover
- Transaction costs

The backtest uses a conservative delay so it does not pretend that finalized
closing data could have been traded before it was available.

## Generated Results

Experiment files are written under `artifacts/`.

Important files include:

| File | Meaning |
|---|---|
| `report.json` | Main model and backtest results |
| `regime_probabilities.csv` | Current and future state probabilities |
| `neural_regimes.csv` | Description of the discovered neural-model states |
| `*_transition_matrix.csv` | Probability of moving between states |
| `neural_backtest_daily.csv` | Daily portfolio audit trail |
| `backtest.png` | Adaptive strategy versus static 60/40 |
| `regimes.png` | Market history colored by detected regime |
| `*.joblib` and `*.pt` | Saved scalers and models |

Downloaded market data is cached under `data/raw/`.

Both `data/raw/` and `artifacts/` are ignored by Git because downloaded data and
generated results should not make the repository unnecessarily large.

## Safety Against Future Leakage

Several rules prevent the model from accidentally looking into the future:

- Rolling features use only past and present values.
- The feature scaler is fitted on training data only.
- Training, validation, and test dates remain in order.
- Live probabilities use causal filtering.
- Historical smoothing is never used for trading decisions.
- Portfolio decisions are delayed before earning returns.
- The final test period is not used to choose a random restart.

## Project Structure

```text
configs/default.yaml            Main experiment settings
src/market_regime/data.py       Data download and validation
src/market_regime/features.py   Feature engineering and scaling
src/market_regime/baseline.py   Gaussian and mixture HMM baselines
src/market_regime/neural_hmm.py PyTorch neural-emission HMM
src/market_regime/evaluation.py Model evaluation and regime names
src/market_regime/backtest.py   Allocation backtest
src/market_regime/experiment.py Complete historical experiment
src/market_regime/forecast.py   Full-history next-regime forecast
src/market_regime/cli.py        Terminal commands
tests/                          Automated tests
```

## Verify The Code

```bash
pytest
ruff check src tests
ruff format --check src tests
```

The current test suite checks data preparation, future leakage, HMM filtering,
neural training, model checkpoints, transaction costs, backtest timing, and the
complete experiment workflow.

## Current Limitations

- The current evaluation uses one fixed historical split, not full walk-forward testing.
- Hidden states are statistical clusters, not official economic labels.
- The neural parameterization is not yet more expressive than a classical mixture HMM.
- Results can change with the selected assets, features, dates, and costs.
- `yfinance` is useful for education but is not a professional trading-data feed.
- This project is research software and does not provide investment advice.
