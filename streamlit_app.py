"""Public dashboard for the latest market-regime research forecast."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_regime.dashboard import (  # noqa: E402
    DEFAULT_FORECAST_URL,
    DEFAULT_HISTORY_URL,
    DashboardDataError,
    allocation_target,
    freshness,
    load_forecast,
    load_history,
    most_likely,
    next_scheduled_run,
)

LOCAL_FORECAST = ROOT / "artifacts/live_predictions/latest_forecast.json"
LOCAL_HISTORY = ROOT / "artifacts/live_predictions/prediction_history.csv"
REGIME_COLORS = {
    "Calm": "#2AF598",
    "Trending": "#FFB000",
    "Crisis": "#FF4D6D",
}


def _setting(name: str, default: str) -> str:
    environment_value = os.getenv(name)
    if environment_value:
        return environment_value
    try:
        secret_value = st.secrets.get(name)
    except (FileNotFoundError, KeyError):
        secret_value = None
    return str(secret_value) if secret_value else default


@st.cache_data(ttl=300, show_spinner=False)
def _forecast(forecast_url: str):
    return load_forecast(LOCAL_FORECAST, forecast_url)


@st.cache_data(ttl=300, show_spinner=False)
def _history(history_url: str):
    return load_history(LOCAL_HISTORY, history_url)


def _probability_frame(rows) -> pd.DataFrame:  # noqa: ANN001
    return pd.DataFrame(
        {
            "Regime": [row.regime for row in rows],
            "Probability": [row.probability for row in rows],
            "Label": [f"{row.probability:.1%}" for row in rows],
        }
    ).sort_values("Probability", ascending=False)


def _probability_chart(frame: pd.DataFrame) -> None:
    domain = list(REGIME_COLORS)
    colors = [REGIME_COLORS[name] for name in domain]
    st.vega_lite_chart(
        frame,
        {
            "background": None,
            "height": 238,
            "mark": {
                "type": "bar",
                "cornerRadiusEnd": 4,
                "height": 26,
                "opacity": 0.92,
                "stroke": "#E6F1F5",
                "strokeOpacity": 0.2,
            },
            "encoding": {
                "y": {
                    "field": "Regime",
                    "type": "nominal",
                    "sort": "-x",
                    "axis": {"title": None, "labelFontSize": 13, "labelPadding": 8},
                },
                "x": {
                    "field": "Probability",
                    "type": "quantitative",
                    "scale": {"domain": [0, 1]},
                    "axis": {"title": None, "format": ".0%", "grid": True},
                },
                "color": {
                    "field": "Regime",
                    "type": "nominal",
                    "scale": {"domain": domain, "range": colors},
                    "legend": None,
                },
                "tooltip": [
                    {"field": "Regime", "type": "nominal"},
                    {"field": "Probability", "type": "quantitative", "format": ".2%"},
                ],
            },
            "config": {
                "view": {"stroke": None},
                "axis": {
                    "domainColor": "#354550",
                    "gridColor": "#1E2B34",
                    "labelColor": "#A8BAC5",
                    "tickColor": "#354550",
                    "titleColor": "#E6F1F5",
                },
            },
        },
        width="stretch",
    )


def _source_name(source: str) -> str:
    return "GitHub Release" if source.startswith(("http://", "https://")) else "local artifact"


def _update_countdown() -> None:
    target = next_scheduled_run()
    target_milliseconds = int(target.timestamp() * 1000)
    st.iframe(
        f"""
        <div class="update-timer">
          <span class="timer-label">Next automatic data update</span>
          <strong id="countdown">Calculating...</strong>
          <span class="timer-schedule">Weekdays at 6:37 PM New York time</span>
        </div>
        <script>
          const target = {target_milliseconds};
          const output = document.getElementById("countdown");
          function updateCountdown() {{
            const remaining = Math.max(0, target - Date.now());
            const totalSeconds = Math.floor(remaining / 1000);
            const days = Math.floor(totalSeconds / 86400);
            const hours = Math.floor((totalSeconds % 86400) / 3600);
            const minutes = Math.floor((totalSeconds % 3600) / 60);
            const seconds = totalSeconds % 60;
            const parts = [];
            if (days > 0) parts.push(`${{days}}d`);
            parts.push(`${{hours}}h`, `${{minutes}}m`, `${{seconds}}s`);
            output.textContent = parts.join(" ");
          }}
          updateCountdown();
          window.setInterval(updateCountdown, 1000);
        </script>
        <style>
          html, body {{ height: 100%; margin: 0; background: transparent; }}
          .update-timer {{
            box-sizing: border-box;
            height: 100%;
            min-height: 58px;
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 10px 14px;
            border: 1px solid #00E5FF;
            border-radius: 6px;
            background: #0B1218;
            color: #E6F1F5;
            box-shadow: inset 0 0 18px rgba(0, 229, 255, 0.05),
                        0 0 16px rgba(0, 229, 255, 0.10);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          .timer-label {{ color: #8FA5B3; font-size: 14px; }}
          #countdown {{
            color: #F6FDFF;
            font-size: 20px;
            text-shadow: 0 0 12px rgba(0, 229, 255, 0.6);
            white-space: nowrap;
          }}
          .timer-schedule {{ margin-left: auto; color: #65D6E4; font-size: 12px; }}
          @media (max-width: 560px) {{
            .update-timer {{ flex-wrap: wrap; gap: 3px 10px; }}
            .timer-label {{ width: 100%; }}
            .timer-schedule {{ margin-left: 0; width: 100%; }}
          }}
        </style>
        """,
        width="stretch",
        height=88,
        tab_index=-1,
    )


st.set_page_config(
    page_title="Market Regime Monitor",
    page_icon=":material/candlestick_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.html(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #070A0F;
        color: #E6F1F5;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0 0 auto 0;
        z-index: 999999;
        height: 2px;
        background: #00E5FF;
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.85);
        pointer-events: none;
    }
    header[data-testid="stHeader"] { background: rgba(7, 10, 15, 0.96); }
    .block-container {
        width: 100%;
        max-width: none !important;
        padding: 1.65rem clamp(1rem, 2.2vw, 3rem) 3rem;
    }
    h1, h2, h3, p, span, label { letter-spacing: 0 !important; }
    h1 {
        color: #F6FDFF !important;
        font-size: 2rem !important;
        line-height: 1.18 !important;
        text-shadow: 0 0 24px rgba(0, 229, 255, 0.28);
    }
    h2 { color: #F6FDFF !important; font-size: 1.25rem !important; }
    h3 { color: #DDF7FA !important; font-size: 1rem !important; }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p { color: #8FA5B3 !important; }
    hr { border-color: #1C2A33 !important; }
    div[data-testid="stMetric"] {
        background: #0B1218;
        border: 1px solid #20303A;
        border-radius: 6px;
        min-height: 116px;
        padding: 14px 16px;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
    }
    div[data-testid="stMetricLabel"] { color: #8FA5B3; }
    div[data-testid="stMetricValue"] { color: #F6FDFF; font-size: 1.45rem; }
    div[data-testid="stMetricDelta"] { color: #89A0AD; }
    .st-key-status_metrics [data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"] {
        border-color: rgba(255, 176, 0, 0.62);
        box-shadow: inset 0 0 18px rgba(255, 176, 0, 0.025),
                    0 0 16px rgba(255, 176, 0, 0.10);
    }
    .st-key-status_metrics [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] {
        border-color: rgba(0, 229, 255, 0.62);
        box-shadow: inset 0 0 18px rgba(0, 229, 255, 0.025),
                    0 0 16px rgba(0, 229, 255, 0.10);
    }
    .st-key-status_metrics [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] {
        border-color: rgba(42, 245, 152, 0.62);
        box-shadow: inset 0 0 18px rgba(42, 245, 152, 0.025),
                    0 0 16px rgba(42, 245, 152, 0.10);
    }
    .st-key-status_metrics [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] {
        border-color: rgba(255, 77, 109, 0.62);
        box-shadow: inset 0 0 18px rgba(255, 77, 109, 0.025),
                    0 0 16px rgba(255, 77, 109, 0.10);
    }
    div[data-testid="stTabs"] button {
        min-height: 46px;
        color: #8FA5B3;
    }
    button[role="tab"][aria-selected="true"] {
        color: #00E5FF !important;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.4);
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #00E5FF !important;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.65);
    }
    button[data-testid="stBaseButton-secondary"] {
        background: #0B1218;
        border: 1px solid #00BFD4;
        color: #DDFBFF !important;
        border-radius: 6px;
    }
    button[data-testid="stBaseButton-secondary"] p,
    button[data-testid="stBaseButton-secondary"] span {
        color: inherit !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        border-color: #00E5FF;
        color: #FFFFFF !important;
        box-shadow: 0 0 14px rgba(0, 229, 255, 0.24);
    }
    div[data-testid="stSegmentedControl"] button {
        background: #0B1218;
        border-color: #263741;
        color: #A8BAC5;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background: #0C2630;
        border-color: #00E5FF;
        color: #E8FDFF;
        box-shadow: inset 0 0 12px rgba(0, 229, 255, 0.12);
    }
    div[data-testid="stAlert"] { border-radius: 6px; }
    div[data-testid="stDataFrame"] {
        border: 1px solid #20303A;
        border-radius: 6px;
        background: #0B1218;
    }
    iframe[title="st.iframe"] { border: 0; }
    .footer-note { color: #718693; font-size: 0.82rem; margin-top: 1.5rem; }
    @media (max-width: 640px) {
        .block-container { padding: 3.75rem 1rem 2.5rem; }
        h1 { font-size: 1.65rem !important; }
        div[data-testid="stMetric"] { min-height: 104px; }
    }
    </style>
    """
)

forecast_url = _setting("FORECAST_URL", DEFAULT_FORECAST_URL)
history_url = _setting("HISTORY_URL", DEFAULT_HISTORY_URL)

header, refresh_column = st.columns([5, 1], vertical_alignment="center")
with header:
    st.title("Market Regime Monitor")
with refresh_column:
    if st.button(
        "Refresh",
        icon=":material/refresh:",
        help="Reload the latest published forecast",
    ):
        st.cache_data.clear()
        st.rerun()

try:
    loaded = _forecast(forecast_url)
except DashboardDataError as exc:
    st.error("The forecast feed is temporarily unavailable.")
    st.caption(str(exc))
    st.stop()

snapshot = loaded.snapshot
next_regime = most_likely(snapshot.next_session)
current_regime = most_likely(snapshot.current)
freshness_name, business_day_lag = freshness(snapshot)
target = allocation_target(snapshot)

st.caption(
    f"{snapshot.assets['equity']} stocks · {snapshot.assets['bond']} bonds · "
    f"{snapshot.assets['volatility']} volatility · Data through "
    f"{snapshot.data_cutoff:%B %d, %Y}"
)

_update_countdown()

with st.container(key="status_metrics"):
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Next-session regime",
        next_regime.regime,
        f"{next_regime.probability:.1%} model probability",
        delta_color="off",
    )
    metric_columns[1].metric(
        "Current regime",
        current_regime.regime,
        f"{current_regime.probability:.1%} model probability",
        delta_color="off",
    )
    metric_columns[2].metric(
        "Data status",
        freshness_name,
        f"{business_day_lag} weekday lag",
        delta_color="off",
    )
    metric_columns[3].metric(
        "Model age",
        f"{snapshot.new_observations_since_training} sessions",
        f"trained through {snapshot.model_data_cutoff:%b %d, %Y}",
        delta_color="off",
    )

if freshness_name == "Stale":
    st.error(
        "This forecast is stale. Treat it as historical output until the scheduled pipeline "
        "publishes newer completed market data."
    )
elif freshness_name == "Delayed":
    st.warning("The latest completed market data may be delayed by a holiday or provider update.")

forecast_tab, history_tab, model_tab = st.tabs(["Forecast", "History", "Model details"])

with forecast_tab:
    chart_column, allocation_column = st.columns([1.7, 1], gap="large")
    with chart_column:
        st.subheader("Regime probabilities")
        horizon = st.segmented_control(
            "Probability horizon",
            ["Next session", "Current"],
            default="Next session",
            label_visibility="collapsed",
        )
        selected_rows = snapshot.next_session if horizon == "Next session" else snapshot.current
        _probability_chart(_probability_frame(selected_rows))
        st.caption(
            "The model estimates a hidden market condition. It does not predict the exact "
            "price, return, or direction of the next session."
        )

    with allocation_column:
        st.subheader("Research allocation")
        allocation_metrics = st.columns(2)
        allocation_metrics[0].metric(snapshot.assets["equity"], f"{target.equity_weight:.1%}")
        allocation_metrics[1].metric(snapshot.assets["bond"], f"{target.bond_weight:.1%}")
        st.caption(
            "Target uses regime probabilities projected across the "
            f"{snapshot.allocation_policy.execution_lag}-session execution lag."
        )
        st.progress(target.equity_weight, text=f"{snapshot.assets['equity']} weight")
        st.progress(target.bond_weight, text=f"{snapshot.assets['bond']} weight")
        st.caption(
            f"Static comparison: 60% {snapshot.assets['equity']} / "
            f"40% {snapshot.assets['bond']}. "
            f"Policy review frequency: {snapshot.allocation_policy.rebalance_frequency}."
        )
        if target.used_fallback:
            st.info(
                "No regime cleared the model-probability threshold, so the policy uses its 60/40 "
                "fallback target."
            )

    st.divider()
    st.subheader("Default policy by regime")
    policy_frame = pd.DataFrame(
        {
            "Regime": list(snapshot.allocation_policy.equity_weights_by_regime),
            f"{snapshot.assets['equity']} weight": [
                snapshot.allocation_policy.equity_weights_by_regime[regime]
                for regime in snapshot.allocation_policy.equity_weights_by_regime
            ],
        }
    )
    policy_frame[f"{snapshot.assets['bond']} weight"] = (
        1.0 - policy_frame[f"{snapshot.assets['equity']} weight"]
    )
    st.dataframe(
        policy_frame,
        hide_index=True,
        width="stretch",
        column_config={
            f"{snapshot.assets['equity']} weight": st.column_config.ProgressColumn(
                format="percent", min_value=0.0, max_value=1.0
            ),
            f"{snapshot.assets['bond']} weight": st.column_config.ProgressColumn(
                format="percent", min_value=0.0, max_value=1.0
            ),
        },
    )

with history_tab:
    st.subheader("Published next-session probabilities")
    try:
        loaded_history = _history(history_url)
    except DashboardDataError as exc:
        st.info("Prediction history will appear after successful scheduled runs.")
        st.caption(str(exc))
    else:
        history = loaded_history.frame
        history_chart = history.pivot(
            index="prediction_data_cutoff",
            columns="regime",
            values="next_session_probability",
        ).sort_index()
        if len(history_chart) == 1:
            st.info(
                "One forecast has been published. This view becomes a time-series chart after "
                "the next successful scheduled run."
            )
            latest_history = history.loc[
                history["prediction_data_cutoff"] == history_chart.index[-1]
            ]
            _probability_chart(
                latest_history.loc[:, ["regime", "next_session_probability"]].rename(
                    columns={
                        "regime": "Regime",
                        "next_session_probability": "Probability",
                    }
                )
            )
        else:
            st.line_chart(
                history_chart,
                color=[REGIME_COLORS.get(column, "#64748B") for column in history_chart.columns],
                y_label="Probability",
                x_label="Data cutoff",
            )
        st.caption(
            f"{history_chart.index.min():%b %d, %Y} to "
            f"{history_chart.index.max():%b %d, %Y} · {_source_name(loaded_history.source)}"
        )

with model_tab:
    st.subheader("Forecast provenance")
    provenance = pd.DataFrame(
        [
            ("Model", "Neural-emission hidden Markov model"),
            ("Bundle", snapshot.model_bundle_id),
            ("Model created", snapshot.model_created_at.strftime("%b %d, %Y %H:%M UTC")),
            ("Training observations", f"{snapshot.model.get('training_observations', 0):,}"),
            ("Hidden states", str(snapshot.model.get("n_states", "Unknown"))),
            ("Mixture components", str(snapshot.model.get("n_components", "Unknown"))),
            ("Feature count", str(snapshot.model.get("n_features", "Unknown"))),
            ("Selected seed", str(snapshot.model.get("selected_seed", "Unknown"))),
            ("Forecast generated", snapshot.generated_at.strftime("%b %d, %Y %H:%M UTC")),
            ("Data source", _source_name(loaded.source)),
        ],
        columns=["Field", "Value"],
    )
    st.dataframe(provenance, hide_index=True, width="stretch")
    st.caption(
        "Daily inference reuses a frozen model bundle. The scheduled workflow retrains the "
        "model weekly or when revised history invalidates the saved fingerprint."
    )

st.html(
    '<p class="footer-note">Research software only. This dashboard is not investment advice, '
    "and historical backtests do not guarantee future performance.</p>"
)
