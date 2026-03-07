from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np
import plotly.graph_objects as go

from interface.bacteria_information import MICROORGANISM


def _validate_inputs(
    times: Sequence[Union[int, float]],
    predictions: Dict[str, Sequence[Union[int, float]]],
) -> tuple[np.ndarray, float, float, int, int]:
    """
    Validate plotting inputs and compute shared axis limits.

    Parameters
    ----------
    times : sequence of int or float
        Storage times associated with prediction values.

    predictions : dict[str, sequence of int or float]
        Dictionary mapping each bacterium key to its sequence of predicted
        microbial loads.

    Returns
    -------
    tuple
        A tuple containing:
        - x : np.ndarray
            Time values converted to a float NumPy array.
        - x0 : float
            Minimum x value.
        - x1 : float
            Maximum x value.
        - y_axis_min : int
            Shared lower bound for the y-axis, computed as the global minimum
            across all prediction series, rounded down to the nearest integer.
        - y_axis_max : int
            Shared upper bound for the y-axis, fixed at 10.

    Raises
    ------
    ValueError
        If `times` is empty, if `predictions` is empty, if a bacterium is not
        found in `MICROORGANISM`, or if a prediction series length does not
        match the length of `times`.
    """
    if times is None or len(times) == 0:
        raise ValueError("'times' must be a non-empty sequence")

    if predictions is None or len(predictions) == 0:
        raise ValueError("'predictions' must be a non-empty dictionary")

    x = np.asarray(times, dtype=float)
    x0, x1 = float(np.min(x)), float(np.max(x))

    all_y_values = []

    for bacteria, y_values in predictions.items():
        if bacteria not in MICROORGANISM:
            raise ValueError(f"Bacteria '{bacteria}' not found in MICROORGANISM")

        y = np.asarray(y_values, dtype=float)

        if len(y) != len(x):
            raise ValueError(
                f"Length mismatch for '{bacteria}': {len(y)} != {len(x)}"
            )

        all_y_values.append(y)

    all_y = np.concatenate(all_y_values)
    y_axis_min = int(np.floor(np.min(all_y)))
    y_axis_max = 10

    return x, x0, x1, y_axis_min, y_axis_max


def _get_thresholds(bacteria: str) -> tuple[float, float, float]:
    """
    Retrieve risk thresholds for a given bacterium.

    Parameters
    ----------
    bacteria : str
        Bacterium key used in `MICROORGANISM`.

    Returns
    -------
    tuple[float, float, float]
        Thresholds `(raw, medium, high)` for the bacterium.

    Raises
    ------
    ValueError
        If the bacterium is missing from `MICROORGANISM` or if one of the
        required thresholds is missing.
    """
    if bacteria not in MICROORGANISM:
        raise ValueError(f"Bacteria '{bacteria}' not found in MICROORGANISM")

    data = MICROORGANISM[bacteria]

    raw_thr = data.get("raw")
    med_thr = data.get("medium")
    high_thr = data.get("high")

    if raw_thr is None:
        raise ValueError(f"'raw' threshold missing for '{bacteria}'")
    if med_thr is None:
        raise ValueError(f"'medium' threshold missing for '{bacteria}'")
    if high_thr is None:
        raise ValueError(f"'high' threshold missing for '{bacteria}'")

    return raw_thr, med_thr, high_thr


def _add_risk_zones(
    fig: go.Figure,
    *,
    x0: float,
    x1: float,
    y_axis_min: int,
    y_axis_max: int,
    raw_thr: float,
    med_thr: float,
    high_thr: float,
) -> None:
    """
    Add colored horizontal risk zones to a figure.

    The zones are:
    - green  : y-axis minimum to raw
    - yellow : raw to medium
    - orange : medium to high
    - red    : high to y-axis maximum

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to enrich with horizontal colored bands.

    x0 : float
        Left bound of the x range.

    x1 : float
        Right bound of the x range.

    y_axis_min : int
        Lower bound of the shared y-axis.

    y_axis_max : int
        Upper bound of the shared y-axis.

    raw_thr : float
        Lower risk threshold.

    med_thr : float
        Medium risk threshold.

    high_thr : float
        High risk threshold.
    """
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


def _add_storage_time_marker(
    fig: go.Figure,
    *,
    storage_time: float,
    y_axis_min: int,
    y_axis_max: int,
) -> None:
    """
    Add a vertical red line indicating the storage time.

    The marker spans the full height of the chart, from the shared y-axis
    minimum to the shared y-axis maximum.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to enrich with the storage-time marker.

    storage_time : float
        Storage time to display on the x-axis.

    y_axis_min : int
        Lower bound of the shared y-axis.

    y_axis_max : int
        Upper bound of the shared y-axis.
    """
    fig.add_shape(
        type="line",
        x0=storage_time,
        x1=storage_time,
        y0=y_axis_min,
        y1=y_axis_max,
        line=dict(color="red", width=3),
        layer="above",
    )


def plot_predictions_over_time(
    times: Sequence[Union[int, float]],
    predictions: Dict[str, Sequence[Union[int, float]]],
    *,
    line_width: int = 5,
    storage_time: float | None = None,
) -> Dict[str, go.Figure]:
    """
    Create one growth plot per bacterium and return them as a dictionary.

    This function generates one Plotly figure for each bacterium present in
    `predictions`, rather than combining all bacteria into a single figure.

    Each figure contains:
    - the growth curve for the corresponding bacterium only;
    - the four colored horizontal risk zones associated with that bacterium;
    - the same shared y-axis span across all figures;
    - optionally, a vertical red line marking a provided storage time.

    The y-axis span is intentionally shared across all generated figures to keep
    visual comparisons consistent:
    - lower bound = global minimum across all prediction series, rounded down to
      the nearest integer;
    - upper bound = 10.

    No legend is displayed.

    Parameters
    ----------
    times : sequence of int or float
        Storage times in hours.

    predictions : dict[str, sequence of int or float]
        Dictionary mapping each bacterium key to a sequence of predicted
        microbial loads (for example in log cfu total) at the corresponding
        time points.

    line_width : int, default=5
        Width of the growth curve line in each individual figure.

    storage_time : float or None, default=None
        Optional storage duration to highlight on every figure. When provided,
        a vertical red line is drawn at `x = storage_time`, spanning the full
        height of the plot.

    Returns
    -------
    dict[str, plotly.graph_objects.Figure]
        Dictionary where:
        - each key is a bacterium name/code from `predictions`;
        - each value is the corresponding Plotly figure.

    Raises
    ------
    ValueError
        If inputs are empty, if a bacterium is unknown, if thresholds are
        missing, or if a prediction series length does not match `times`.

    Notes
    -----
    This function assumes that `MICROORGANISM[bacteria]` contains the threshold
    entries:
    - ``raw``
    - ``medium``
    - ``high``

    Example
    -------
    >>> figures = plot_predictions_over_time(times, predictions, storage_time=48.0)
    >>> fig_lm = figures["lm"]
    >>> fig_ec = figures["ec"]
    """
    x, x0, x1, y_axis_min, y_axis_max = _validate_inputs(times, predictions)

    figures: Dict[str, go.Figure] = {}

    for bacteria, y_values in predictions.items():
        y = np.asarray(y_values, dtype=float)
        raw_thr, med_thr, high_thr = _get_thresholds(bacteria)

        fig = go.Figure()

        _add_risk_zones(
            fig,
            x0=x0,
            x1=x1,
            y_axis_min=y_axis_min,
            y_axis_max=y_axis_max,
            raw_thr=raw_thr,
            med_thr=med_thr,
            high_thr=high_thr,
        )

        if storage_time is not None:
            _add_storage_time_marker(
                fig,
                storage_time=float(storage_time),
                y_axis_min=y_axis_min,
                y_axis_max=y_axis_max,
            )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=str(bacteria),
                line=dict(width=line_width),
                showlegend=False,
            )
        )

        fig.update_yaxes(range=[y_axis_min, y_axis_max])
        fig.update_xaxes(range=[x0, x1], constrain="domain")

        fig.update_layout(
            xaxis_title="storage time (h)",
            yaxis_title="Microbial load (log cfu total)",
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor="#2C2A29",
            plot_bgcolor="#2C2A29",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=10, b=10),
        )

        figures[bacteria] = fig

    return figures
