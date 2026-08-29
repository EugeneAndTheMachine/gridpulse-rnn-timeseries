"""Reusable Plotly charts."""
import pandas as pd
import plotly.graph_objects as go

from apps.dashboard.config import COLOR_ACTUAL, COLOR_PREDICTED, COLOR_ANOMALY


def forecast_line_chart(df: pd.DataFrame, title: str = "Forecast vs Actual") -> go.Figure:
    """df needs columns: time, predicted, actual (actual may be NaN)."""
    fig = go.Figure()

    if "actual" in df.columns and df["actual"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["actual"],
            name="Actual", mode="lines",
            line=dict(color=COLOR_ACTUAL, width=2),
        ))

    fig.add_trace(go.Scatter(
        x=df["time"], y=df["predicted"],
        name="Predicted", mode="lines",
        line=dict(color=COLOR_PREDICTED, width=2, dash="dash"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time", yaxis_title="Value",
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def residual_chart(df: pd.DataFrame) -> go.Figure:
    """Plot residuals over time. df needs time + residual."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["residual"],
        name="Residual", mode="lines",
        line=dict(color=COLOR_ANOMALY, width=1),
        fill="tozeroy",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Forecast Residuals (predicted − actual)",
        xaxis_title="Time", yaxis_title="Residual",
        height=300, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def anomaly_timeline_chart(
    series_df: pd.DataFrame,
    events_df: pd.DataFrame,
    value_col: str = "actual",
) -> go.Figure:
    """Line chart with anomaly windows shaded."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series_df["time"], y=series_df[value_col],
        name="Value", mode="lines", line=dict(color=COLOR_ACTUAL, width=1.5),
    ))

    for _, ev in events_df.iterrows():
        fig.add_vrect(
            x0=ev["start_time"], x1=ev["end_time"],
            fillcolor=COLOR_ANOMALY, opacity=0.25, line_width=0,
        )

    fig.update_layout(
        title="Anomaly Timeline",
        xaxis_title="Time", yaxis_title="Value",
        height=450, margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig