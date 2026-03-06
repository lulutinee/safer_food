from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import plotly.graph_objects as go

from interface.bacteria_information import MICROORGANISM


def plot_predictions_over_time(
    times: Sequence[Union[int, float]],
    predictions: Dict[str, Sequence[Union[int, float]]],
    *,
    highlight_bacteria: Optional[str] = None,
    base_line_width: int = 2,
    highlight_line_width: int = 5,
) -> go.Figure:
    """
    Plot predicted microbial loads over storage time for multiple bacteria.

    Features
    --------
    - One curve per bacterium.
    - Optional highlighted bacterium (thicker line).
    - Colored horizontal risk bands (50% opacity) for the highlighted bacterium:
        bottom of y-axis → raw : green
        raw → medium           : yellow
        medium → high          : orange
        high → top of y-axis   : red

    Parameters
    ----------
    times : sequence of float
        Storage times in hours.

    predictions : dict
        Dictionary mapping each bacteria code/name to a sequence of predicted
        microbial loads (log cfu/g) at the corresponding time points.

    highlight_bacteria : str, optional
        Bacteria key to highlight and for which threshold zones are displayed.

    base_line_width : int, default=2
        Line width for non-highlighted curves.

    highlight_line_width : int, default=5
        Line width for the highlighted curve.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure compatible with Streamlit.
    """

    if times is None or len(times) == 0:
        raise ValueError("'times' must be a non-empty sequence")

    if predictions is None or len(predictions) == 0:
        raise ValueError("'predictions' must be a non-empty dictionary")

    x = np.asarray(times, dtype=float)
    x0, x1 = float(np.min(x)), float(np.max(x))

    # Determine y-axis limits:
    # - min = minimum of all prediction series, rounded down to lower integer
    # - max = 10
    all_y = np.concatenate(
        [np.asarray(y_values, dtype=float) for y_values in predictions.values()]
    )
    y_axis_min = int(np.floor(np.min(all_y)))
    y_axis_max = 10

    fig = go.Figure()

    # ---------------------------------------------------
    # Add colored threshold zones (if highlighting)
    # ---------------------------------------------------
    if highlight_bacteria is not None:
        if highlight_bacteria not in predictions:
            raise ValueError(
                f"highlight_bacteria='{highlight_bacteria}' not in predictions"
            )

        if highlight_bacteria not in MICROORGANISM:
            raise ValueError(
                f"Bacteria '{highlight_bacteria}' not found in MICROORGANISM"
            )

        data = MICROORGANISM[highlight_bacteria]

        raw_thr = data.get("raw")
        med_thr = data.get("medium")
        high_thr = data.get("high")

        if raw_thr is None:
            raise ValueError(f"'raw' threshold missing for '{highlight_bacteria}'")
        if med_thr is None:
            raise ValueError(f"'medium' threshold missing for '{highlight_bacteria}'")
        if high_thr is None:
            raise ValueError(f"'high' threshold missing for '{highlight_bacteria}'")

        fig.add_hrect(
            y0=y_axis_min,
            y1=raw_thr,
            x0=x0,
            x1=x1,
            fillcolor="green",
            opacity=0.5,
            line_width=0,
            layer="below",
        )

        fig.add_hrect(
            y0=raw_thr,
            y1=med_thr,
            x0=x0,
            x1=x1,
            fillcolor="yellow",
            opacity=0.5,
            line_width=0,
            layer="below",
        )

        fig.add_hrect(
            y0=med_thr,
            y1=high_thr,
            x0=x0,
            x1=x1,
            fillcolor="orange",
            opacity=0.5,
            line_width=0,
            layer="below",
        )

        fig.add_hrect(
            y0=high_thr,
            y1=y_axis_max,
            x0=x0,
            x1=x1,
            fillcolor="red",
            opacity=0.5,
            line_width=0,
            layer="below",
        )

    # ---------------------------------------------------
    # Add growth curves
    # ---------------------------------------------------
    for bacteria, y_values in predictions.items():
        if bacteria not in MICROORGANISM:
            raise ValueError(f"Bacteria '{bacteria}' not found in MICROORGANISM")

        y = np.asarray(y_values, dtype=float)

        if len(y) != len(x):
            raise ValueError(
                f"Length mismatch for '{bacteria}': "
                f"{len(y)} != {len(x)}"
            )

        line_width = (
            highlight_line_width
            if bacteria == highlight_bacteria
            else base_line_width
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=str(bacteria),
                line=dict(width=line_width),
            )
        )

    # Explicitly fix y-axis range
    fig.update_yaxes(range=[y_axis_min, y_axis_max])

    fig.update_layout(
        xaxis_title="storage time (h)",
        yaxis_title="Microbial load (log cfu total)",
        legend_title_text="Bacteria",
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    return fig
