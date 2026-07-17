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

## Live Dashboard

[Open the public Market Regime Monitor](https://aryanjha07-neural-market-regime-model-streamlit-app-qvshs2.streamlit.app/)

The dashboard shows the latest regime probabilities and the example stock and
bond allocation. GitHub Actions refreshes its public forecast automatically.

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

Run expanding-window walk-forward evaluation:

```bash
market-regime walk-forward --config configs/default.yaml
```

Run only the first two folds as a quicker check:

```bash
market-regime walk-forward --config configs/default.yaml --max-folds 2
```

Train and save a reusable live model bundle:

```bash
market-regime train-live --config configs/default.yaml
```

Load that bundle and estimate the next regime without training:

```bash
market-regime predict-live --config configs/default.yaml
```

The original train-then-predict shortcut remains available:

```bash
market-regime forecast --config configs/default.yaml
```

Training can take several minutes on a laptop. Prediction normally processes
only observations added since training and should be much faster.

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
- `market-regime walk-forward` repeats training and testing across many periods.
- `market-regime train-live` creates a reusable full-history model bundle.
- `market-regime predict-live` loads that frozen bundle without calling training.

## Separate Training And Prediction

`train-live` saves the neural checkpoint, scaler, fixed regime names, training
cutoff probabilities, feature fingerprint, configuration, and a versioned
manifest under `artifacts/live_model/`.

`predict-live` verifies that historical features and configuration still match
the bundle. It starts from the saved cutoff probabilities and runs the HMM
update only for newer observations. It writes the latest result and a
deduplicated prediction history under `artifacts/live_predictions/`.

This separation lets a public website generate daily predictions without
waiting for PyTorch training. Retraining can happen less frequently, such as
weekly or monthly, after evaluation and data checks succeed.

## Walk-Forward Evaluation

The default walk-forward experiment starts with about ten years of training
data, uses the next year for validation, and tests on the following year. It
then moves forward and repeats the process with a larger training history.

For every fold, the program:

1. Fits the scaler and restart candidates without test data.
2. Selects each model's restart using validation likelihood.
3. Refits the selected model using all information available before testing.
4. Produces causal probabilities for the untouched test period.
5. Joins all test periods into one continuous cost-aware backtest.

Fold state numbers can change, so every fold learns its own human-readable
regime names before those states are converted into portfolio weights.

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

Walk-forward results are stored under `artifacts/walk_forward/`:

| File | Meaning |
|---|---|
| `walk_forward_report.json` | Main aggregate walk-forward report |
| `folds.csv` | Dates, selected seeds, and scores for every fold |
| `likelihood_summary.csv` | Aggregate unseen-data model comparison |
| `probabilities.csv` | Causal probabilities for every test date and model |
| `decision_weights.csv` | Stock and bond targets before scheduling and execution |
| `backtest_daily.csv` | Continuous daily audit trail across all folds |
| `backtest_metrics.csv` | Model allocation and static 60/40 results |
| `likelihood_by_fold.png` | Model likelihood comparison through time |

Live-operation files include:

| File | Meaning |
|---|---|
| `live_model/model_manifest.json` | Version, schema, cutoff, labels, and bundle file map |
| `live_model/bundles/*/live_neural_hmm.pt` | Immutable trained neural checkpoint |
| `live_model/bundles/*/live_feature_scaler.joblib` | Training-time feature scaler |
| `live_predictions/latest_forecast.json` | Latest website-ready regime estimate |
| `live_predictions/prediction_history.csv` | Deduplicated probability history |

Downloaded market data is cached under `data/raw/`.

Both `data/raw/` and `artifacts/` are ignored by Git because downloaded data and
generated results should not make the repository unnecessarily large.

## Automation

GitHub Actions runs two workflows:

- `CI` runs Ruff and the complete test suite on pushes and pull requests.
- `Update live market forecast` runs at 6:37 PM New York time every weekday.

The forecast workflow retrains the model on Mondays and makes fast predictions
with the saved model on the other weekdays. In `auto` mode, it also trains a
model when no saved one exists yet. US market holidays may produce the same data
cutoff as the previous run, which is normal.

The workflow keeps generated files out of Git. Instead, it creates a GitHub
Release named `live-forecast` containing:

- `latest_forecast.json` for the dashboard
- `prediction_history.csv` for the history chart
- `live_model.tar.gz` for the next prediction run
- `checksums.txt` for file-integrity checks

Each temporary runner downloads the previous model and prediction history before
it starts. Market data is cached for same-day reruns, and Yahoo downloads are
retried three times. If Yahoo revises old prices and the saved fingerprint no
longer matches, `auto` mode retrains once and republishes the repaired model.
`predict-only` never uses this recovery. A forecast is checked before
publication. The workflow uploads uniquely named recovery snapshots before it
updates the friendly filenames, and it keeps the newest snapshots. If a Release
upload is interrupted, the next run can recover the model and full history
instead of starting over. The stable forecast is uploaded last. Failures before
the publication stage leave all public files unchanged. GitHub replaces a
friendly Release filename one file at a time, so a stable URL can be briefly
unavailable while that final upload is happening.

You can also start the workflow from the repository's **Actions** tab:

- `auto` follows the weekly rule and replaces a missing or unreadable model.
- `predict-only` requires an existing published model and never trains.
- `train-and-predict` always trains a fresh model before predicting.

No personal token or market-data secret is required. The workflow uses GitHub's
built-in `GITHUB_TOKEN` and grants it only `contents: write` so it can update the
Release. The schedule starts only after the workflow is committed to the default
branch. GitHub can delay scheduled jobs, and it may disable schedules in an
inactive public repository, so failed or stale runs should be checked in the
Actions tab.

## Public Deployment

Run the dashboard locally from the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The dashboard first uses local files under `artifacts/live_predictions/`. When
those files are not present, it uses optional `FORECAST_URL` and `HISTORY_URL`
settings, then falls back to these public Release assets:

- `https://github.com/aryanjha07/neural-market-regime-model/releases/download/live-forecast/latest_forecast.json`
- `https://github.com/aryanjha07/neural-market-regime-model/releases/download/live-forecast/prediction_history.csv`

To publish with Streamlit Community Cloud:

1. Make the GitHub repository public so the dashboard can read Release files
   without a private access token.
2. Commit and push the dashboard, requirements, and workflow files to `main`.
3. In GitHub **Actions**, run `Update live market forecast` once with
   `train-and-predict`. Wait for the `live-forecast` Release to appear.
4. Open `share.streamlit.io`, choose **Create app**, and select
   `aryanjha07/neural-market-regime-model`.
5. Choose branch `main`, entrypoint `streamlit_app.py`, and Python 3.12.
6. Deploy the app and open the
   [public dashboard](https://aryanjha07-neural-market-regime-model-streamlit-app-qvshs2.streamlit.app/).

No Streamlit secret is needed while this repository and its Release are public.
The URL settings are optional overrides for a fork or a different publication
location. Put overrides in the Community Cloud secrets panel or in a local
`.streamlit/secrets.toml`; that file is ignored by Git.

`requirements.txt` intentionally contains only dashboard packages. Model
training uses `pyproject.toml` inside GitHub Actions, so the public website does
not need to install PyTorch or run training on a small web server.

## Later Data Provider Upgrade

`yfinance` is a sensible free source for this research version, but it should
not be treated as a guaranteed production feed. A later upgrade can stay small:

1. Define a provider interface in `data.py` that returns the same checked OHLCV table.
2. Keep `yfinance` as the free development adapter.
3. Add a licensed provider adapter with authentication, retries, rate limits,
   exchange calendars, and clear adjusted-price rules.
4. Run both adapters over the same dates and test that their columns, time zones,
   missing-session behavior, and corporate-action handling agree.
5. Select the provider in configuration without changing feature engineering,
   model training, prediction, or the dashboard.

This separation is more useful than replacing the model whenever the data source
changes. The current validation and historical fingerprint already catch many
provider revisions instead of silently producing a different forecast.

## Safety Against Future Leakage

Several rules prevent the model from accidentally looking into the future:

- Rolling features use only past and present values.
- The feature scaler is fitted on training data only.
- Training, validation, and test dates remain in order.
- Live probabilities use causal filtering.
- Historical smoothing is never used for trading decisions.
- Portfolio decisions are delayed before earning returns.
- The final test period is not used to choose a random restart.
- Walk-forward fold scalers, state names, and models never use their test rows.
- Walk-forward test decisions are joined before rebalancing and backtesting.
- Live prediction cannot call model training and keeps training-time regime names fixed.
- Live prediction rejects changed history, feature settings, or incomplete bundles.

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
src/market_regime/training.py   Shared model fitting and restart selection
src/market_regime/walk_forward.py Expanding-window evaluation
src/market_regime/forecast.py   Versioned live training and fast prediction
src/market_regime/dashboard.py  Public forecast validation and loading
src/market_regime/cli.py        Terminal commands
streamlit_app.py                Public dashboard entrypoint
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
complete experiment workflow. Live tests also disable fitting during prediction
and compare incremental probabilities with a complete HMM replay.

## Current Limitations

- Walk-forward results still depend on the chosen window lengths and assets.
- Hidden states are statistical clusters, not official economic labels.
- The neural parameterization is not yet more expressive than a classical mixture HMM.
- Results can change with the selected assets, features, dates, and costs.
- Historical data revisions require retraining because the bundle uses an exact fingerprint.
- `yfinance` is useful for education but is not a professional trading-data feed.
- This project is research software and does not provide investment advice.
