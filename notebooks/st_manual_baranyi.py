from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_logic.sk_baseline import ClassicalModelRegressor
from interface.bacteria_information import MICROORGANISM


# =============================================================================
# Configuration
# =============================================================================

MATRIX_IDS = ["beef", "pork", "poultry", "seafood"]
ORGANISM_IDS = ["ec", "lm", "ss", "ta"]
TEMPERATURES = [4, 6, 10, 20, 37]

PRIMARY_PARAM_NAMES = ["Initial Value", "Lag", "Maximum Rate", "Final Value"]

DEFAULT_DURATION_H = 500.0
DEFAULT_INITIAL_VALUE = 0.0
DEFAULT_FINAL_VALUE = 9.0
DEFAULT_GROWTH_POINTS = 300

DEFAULT_LAG = {
    "ec": {4: 120.0, 6: 90.0, 10: 40.0, 20: 8.0, 37: 2.0},
    "lm": {4: 100.0, 6: 70.0, 10: 30.0, 20: 6.0, 37: 2.0},
    "ss": {4: 140.0, 6: 100.0, 10: 45.0, 20: 10.0, 37: 3.0},
    "ta": {4: 80.0, 6: 60.0, 10: 20.0, 20: 4.0, 37: 1.0},
}

DEFAULT_MAX_RATE = {
    "ec": {4: 0.015, 6: 0.020, 10: 0.050, 20: 0.180, 37: 0.350},
    "lm": {4: 0.012, 6: 0.018, 10: 0.045, 20: 0.150, 37: 0.280},
    "ss": {4: 0.010, 6: 0.016, 10: 0.040, 20: 0.140, 37: 0.300},
    "ta": {4: 0.020, 6: 0.030, 10: 0.070, 20: 0.220, 37: 0.450},
}

BARANYI_OUTPUT_PATH = Path("manual_baranyi_parameters.csv")
ARRHENIUS_OUTPUT_PATH = Path("manual_arrhenius_V2.csv")

ARRHENIUS_R = 8.314462618


# =============================================================================
# Session-state helpers
# =============================================================================

def _state_key(matrix_id: str, organism_id: str, temperature: float, param_name: str) -> str:
    """
    Build a unique Streamlit session-state key for one matrix, one organism,
    one temperature, and one parameter.
    """
    return f"{matrix_id}__{organism_id}__{temperature}__{param_name}"


def build_default_manual_params_df() -> pd.DataFrame:
    """
    Build the complete default manual-parameter dataframe for all possible
    MatrixID x OrganismID x Temperature combinations.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns:
        - MatrixID
        - OrganismID
        - Temperature
        - Initial Value
        - Lag
        - Maximum Rate
        - Final Value
    """
    rows = []

    for matrix_id in MATRIX_IDS:
        for organism_id in ORGANISM_IDS:
            for temperature in TEMPERATURES:
                rows.append(
                    {
                        "MatrixID": matrix_id,
                        "OrganismID": organism_id,
                        "Temperature": float(temperature),
                        "Initial Value": DEFAULT_INITIAL_VALUE,
                        "Lag": float(DEFAULT_LAG[organism_id][temperature]),
                        "Maximum Rate": float(DEFAULT_MAX_RATE[organism_id][temperature]),
                        "Final Value": DEFAULT_FINAL_VALUE,
                    }
                )

    return pd.DataFrame(rows)


def load_saved_manual_params(path: Path) -> pd.DataFrame | None:
    """
    Load the saved manual Baranyi parameters file if it exists.

    If the file contains multiple rows for the same MatrixID x OrganismID x Temperature,
    the last occurrence is kept.

    Parameters
    ----------
    path : Path
        Path to the saved manual-parameter file.

    Returns
    -------
    pd.DataFrame | None
        Loaded dataframe, or None if the file does not exist.
    """
    if not path.exists():
        return None

    df = pd.read_csv(path, sep="\t")

    required_cols = [
        "MatrixID",
        "OrganismID",
        "Temperature",
        "Initial Value",
        "Lag",
        "Maximum Rate",
        "Final Value",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Le fichier '{path}' ne contient pas les colonnes requises : {missing}"
        )

    df = df.copy()
    df["MatrixID"] = df["MatrixID"].astype(str)
    df["OrganismID"] = df["OrganismID"].astype(str)
    df["Temperature"] = df["Temperature"].astype(float)
    df["Initial Value"] = df["Initial Value"].astype(float)
    df["Lag"] = df["Lag"].astype(float)
    df["Maximum Rate"] = df["Maximum Rate"].astype(float)
    df["Final Value"] = df["Final Value"].astype(float)

    df = df.drop_duplicates(
        subset=["MatrixID", "OrganismID", "Temperature"],
        keep="last",
    )

    return df


def merge_default_and_loaded_params(default_df: pd.DataFrame, loaded_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge the full default dataframe with the loaded dataframe.

    When loaded values exist for a given MatrixID x OrganismID x Temperature,
    they replace the default values.

    Parameters
    ----------
    default_df : pd.DataFrame
        Complete default dataframe.
    loaded_df : pd.DataFrame | None
        Loaded dataframe from disk, or None.

    Returns
    -------
    pd.DataFrame
        Complete merged dataframe.
    """
    if loaded_df is None:
        return default_df.copy()

    key_cols = ["MatrixID", "OrganismID", "Temperature"]
    value_cols = ["Initial Value", "Lag", "Maximum Rate", "Final Value"]

    merged = default_df.merge(
        loaded_df[key_cols + value_cols],
        on=key_cols,
        how="left",
        suffixes=("", "_loaded"),
    )

    for col in value_cols:
        loaded_col = f"{col}_loaded"
        merged[col] = merged[loaded_col].combine_first(merged[col])

    cols_to_drop = [f"{col}_loaded" for col in value_cols]
    merged = merged.drop(columns=cols_to_drop)

    return merged


def set_reference_params_in_session(reference_df: pd.DataFrame) -> None:
    """
    Write the reference dataframe values into Streamlit session state.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Complete reference dataframe containing one row per
        MatrixID x OrganismID x Temperature.
    """
    for row in reference_df.itertuples(index=False):
        lag_key = _state_key(row.MatrixID, row.OrganismID, row.Temperature, "Lag")
        rate_key = _state_key(row.MatrixID, row.OrganismID, row.Temperature, "Maximum Rate")

        st.session_state[lag_key] = float(row.Lag)
        st.session_state[rate_key] = float(row._asdict()["Maximum Rate"])


def initialize_session_state() -> None:
    """
    Initialize Streamlit session state with the complete default parameter grid.

    This initialization is performed only once per session, unless the user
    explicitly clicks the 'Charger' button to replace values from disk.
    """
    if "manual_reference_params" not in st.session_state:
        reference_df = build_default_manual_params_df()
        st.session_state["manual_reference_params"] = reference_df
        set_reference_params_in_session(reference_df)


def reload_reference_from_file() -> tuple[bool, str]:
    """
    Reload the complete reference parameter grid from BARANYI_OUTPUT_PATH.

    Default values are created first for all MatrixID x OrganismID x Temperature
    combinations. When the saved file exists, matching rows replace the defaults.

    Returns
    -------
    tuple[bool, str]
        - success flag
        - status message
    """
    try:
        default_df = build_default_manual_params_df()
        loaded_df = load_saved_manual_params(BARANYI_OUTPUT_PATH)
        reference_df = merge_default_and_loaded_params(default_df, loaded_df)

        st.session_state["manual_reference_params"] = reference_df
        set_reference_params_in_session(reference_df)

        if loaded_df is None:
            return True, (
                f"Aucun fichier trouvé à '{BARANYI_OUTPUT_PATH}'. "
                "Les valeurs par défaut ont été conservées."
            )

        return True, (
            f"Valeurs chargées depuis '{BARANYI_OUTPUT_PATH}' "
            "et utilisées comme valeurs par défaut."
        )
    except Exception as exc:
        return False, f"Erreur lors du chargement : {exc}"


def get_manual_primary_params(matrix_id: str) -> pd.DataFrame:
    """
    Build the manual Baranyi-parameter dataframe from the current slider values
    for the selected MatrixID.

    Parameters
    ----------
    matrix_id : str
        Selected food matrix.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with one row per OrganismID x Temperature.
    """
    rows = []

    for organism_id in ORGANISM_IDS:
        for temperature in TEMPERATURES:
            lag = float(st.session_state[_state_key(matrix_id, organism_id, temperature, "Lag")])
            max_rate = float(
                st.session_state[_state_key(matrix_id, organism_id, temperature, "Maximum Rate")]
            )

            rows.append(
                {
                    "MatrixID": matrix_id,
                    "OrganismID": organism_id,
                    "Temperature": float(temperature),
                    "Initial Value": DEFAULT_INITIAL_VALUE,
                    "Lag": lag,
                    "Maximum Rate": max_rate,
                    "Final Value": DEFAULT_FINAL_VALUE,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Baranyi prediction helpers
# =============================================================================

def predict_baranyi(
    times: Iterable[float],
    *,
    initial_value: float,
    lag: float,
    maximum_rate: float,
    final_value: float,
) -> np.ndarray:
    """
    Predict a Baranyi growth curve for a given time grid and parameter set.
    """
    times = np.asarray(times, dtype=float)

    model = ClassicalModelRegressor(model="baranyi")
    model.fit([1, 2, 3, 4, 5], [1, 1, 3, 5, 5])

    model.params_ = {
        "Initial Value": float(initial_value),
        "Lag": float(lag),
        "Maximum Rate": float(maximum_rate),
        "Final Value": float(final_value),
    }

    y_pred = model.predict(times)
    return np.asarray(y_pred, dtype=float)


# =============================================================================
# Plotting helpers
# =============================================================================

def _get_thresholds(organism_id: str) -> Tuple[float, float, float]:
    """
    Retrieve the 'raw', 'medium', and 'high' thresholds for one bacterium.
    """
    if organism_id not in MICROORGANISM:
        raise ValueError(f"Bacteria '{organism_id}' not found in MICROORGANISM")

    raw_thr = MICROORGANISM[organism_id].get("raw")
    med_thr = MICROORGANISM[organism_id].get("medium")
    high_thr = MICROORGANISM[organism_id].get("high")

    if raw_thr is None or med_thr is None or high_thr is None:
        raise ValueError(f"Missing thresholds for bacterium '{organism_id}'")

    return float(raw_thr), float(med_thr), float(high_thr)


def _compute_y_axis_min(curves: Dict[float, np.ndarray]) -> int:
    """
    Compute the shared lower y-axis bound for one organism figure.
    """
    all_y = np.concatenate([np.asarray(v, dtype=float) for v in curves.values()])
    return int(np.floor(np.min(all_y)))


def add_risk_zones(
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
    Add colored horizontal risk zones to a Plotly figure.
    """
    fig.add_hrect(
        y0=y_axis_min,
        y1=raw_thr,
        x0=x0,
        x1=x1,
        fillcolor="green",
        opacity=0.25,
        line_width=0,
        layer="below",
    )
    fig.add_hrect(
        y0=raw_thr,
        y1=med_thr,
        x0=x0,
        x1=x1,
        fillcolor="yellow",
        opacity=0.25,
        line_width=0,
        layer="below",
    )
    fig.add_hrect(
        y0=med_thr,
        y1=high_thr,
        x0=x0,
        x1=x1,
        fillcolor="orange",
        opacity=0.25,
        line_width=0,
        layer="below",
    )
    fig.add_hrect(
        y0=high_thr,
        y1=y_axis_max,
        x0=x0,
        x1=x1,
        fillcolor="red",
        opacity=0.25,
        line_width=0,
        layer="below",
    )


def make_organism_figure(
    organism_id: str,
    duration_h: float,
    matrix_id: str,
    n_points: int = DEFAULT_GROWTH_POINTS,
) -> go.Figure:
    """
    Build the growth figure for one bacterium.
    """
    times = np.linspace(0.0, float(duration_h), int(n_points))

    curves: Dict[float, np.ndarray] = {}
    for temperature in TEMPERATURES:
        lag = float(st.session_state[_state_key(matrix_id, organism_id, temperature, "Lag")])
        max_rate = float(
            st.session_state[_state_key(matrix_id, organism_id, temperature, "Maximum Rate")]
        )

        curves[temperature] = predict_baranyi(
            times,
            initial_value=DEFAULT_INITIAL_VALUE,
            lag=lag,
            maximum_rate=max_rate,
            final_value=DEFAULT_FINAL_VALUE,
        )

    raw_thr, med_thr, high_thr = _get_thresholds(organism_id)
    y_axis_min = _compute_y_axis_min(curves)
    y_axis_max = 10

    fig = go.Figure()

    add_risk_zones(
        fig,
        x0=0.0,
        x1=float(duration_h),
        y_axis_min=y_axis_min,
        y_axis_max=y_axis_max,
        raw_thr=raw_thr,
        med_thr=med_thr,
        high_thr=high_thr,
    )

    for temperature in TEMPERATURES:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=curves[temperature],
                mode="lines",
                name=f"{temperature}°C",
                line=dict(width=2),
            )
        )

    title = MICROORGANISM.get(organism_id, {}).get("usual_name", organism_id)

    fig.update_yaxes(
        range=[y_axis_min, y_axis_max],
        title="Microbial load (log cfu total)",
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.10)",
        zeroline=True,
        zerolinecolor="rgba(0, 0, 0, 0.20)",
    )

    fig.update_xaxes(
        range=[0.0, float(duration_h)],
        title="Storage time (h)",
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.10)",
        zeroline=True,
        zerolinecolor="rgba(0, 0, 0, 0.20)",
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        height=360,
    )

    return fig


# =============================================================================
# Arrhenius fitting helpers
# =============================================================================

def fit_arrhenius_params(temperatures: Iterable[float], values: Iterable[float]) -> Dict[str, float]:
    """
    Fit an Arrhenius model to one primary-model parameter across temperatures.
    """
    x = np.asarray(list(temperatures), dtype=float)
    y = np.asarray(list(values), dtype=float)

    if len(x) < 2:
        raise ValueError("At least two temperatures are required to fit Arrhenius")

    model = ClassicalModelRegressor(model="arrhenius")
    model.fit(x, y)

    params = dict(model.params_)

    required = {"A", "Ea", "R", "x_in_celsius"}
    missing = required - set(params.keys())
    if missing:
        raise ValueError(f"Missing Arrhenius parameters after fit: {missing}")

    return {
        "A": float(params["A"]),
        "Ea": float(params["Ea"]),
        "R": float(params["R"]),
        "x_in_celsius": float(params["x_in_celsius"]),
    }


def make_constant_arrhenius_dict(value: float) -> Dict[str, float]:
    """
    Represent a temperature-independent constant in the same dictionary format
    as an Arrhenius model.
    """
    return {
        "A": float(value),
        "Ea": 0.0,
        "R": ARRHENIUS_R,
        "x_in_celsius": 1.0,
    }


def build_arrhenius_dataframe(baranyi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Arrhenius-parameter dataframe from the manual Baranyi dataframe.
    """
    rows = []
    group_cols = ["MatrixID", "OrganismID"]

    for (matrix_id, organism_id), group in baranyi_df.groupby(group_cols, sort=False):
        group = group.sort_values("Temperature")

        temperatures = group["Temperature"].to_numpy(dtype=float)
        lag_values = group["Lag"].to_numpy(dtype=float)
        max_rate_values = group["Maximum Rate"].to_numpy(dtype=float)

        lag_arrhenius = fit_arrhenius_params(temperatures, lag_values)
        max_rate_arrhenius = fit_arrhenius_params(temperatures, max_rate_values)

        rows.append(
            {
                "MatrixID": matrix_id,
                "OrganismID": organism_id,
                "Initial Value": make_constant_arrhenius_dict(DEFAULT_INITIAL_VALUE),
                "Lag": lag_arrhenius,
                "Maximum Rate": max_rate_arrhenius,
                "Final Value": make_constant_arrhenius_dict(DEFAULT_FINAL_VALUE),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# File I/O helpers
# =============================================================================

def append_tsv(df: pd.DataFrame, path: Path) -> None:
    """
    Append a dataframe to a tab-separated file.
    """
    header = not path.exists()
    df.to_csv(path, sep="\t", index=False, mode="a", header=header)


# =============================================================================
# UI rendering helpers
# =============================================================================

def render_top_controls() -> Tuple[float, str, bool, bool]:
    """
    Render the top control row of the application.

    Returns
    -------
    tuple[float, str, bool, bool]
        - duration_h
        - matrix_id
        - load_clicked
        - save_clicked
    """
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.2, 1.2, 0.8, 0.8])

    with ctrl_col1:
        duration_h = st.number_input(
            "Growth duration (hours)",
            min_value=0.0,
            value=float(DEFAULT_DURATION_H),
            step=10.0,
        )

    with ctrl_col2:
        matrix_id = st.selectbox(
            "MatrixID",
            options=MATRIX_IDS,
            index=0,
        )

    with ctrl_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        load_clicked = st.button("Charger", use_container_width=True)

    with ctrl_col4:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        save_clicked = st.button("Enregistrer", type="primary", use_container_width=True)

    return float(duration_h), matrix_id, load_clicked, save_clicked


def render_organism_column(organism_id: str, duration_h: float, matrix_id: str) -> None:
    """
    Render one bacterium column in a compact layout.

    For each temperature, controls are shown on one row:
    - temperature label
    - Lag slider
    - Maximum Rate slider
    """
    label = MICROORGANISM.get(organism_id, {}).get("usual_name", organism_id)

    st.markdown(f"### {label}")
    st.caption(f"OrganismID: {organism_id}")

    fig = make_organism_figure(
        organism_id=organism_id,
        duration_h=duration_h,
        matrix_id=matrix_id,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Manual parameters**")

    header_col1, header_col2, header_col3 = st.columns([0.7, 1.6, 1.6])
    with header_col1:
        st.markdown("**T**")
    with header_col2:
        st.markdown("**Lag**")
    with header_col3:
        st.markdown("**Maximum Rate**")

    for temperature in TEMPERATURES:
        row_col1, row_col2, row_col3 = st.columns([0.7, 1.6, 1.6])

        with row_col1:
            st.markdown(f"**{temperature}°C**")

        with row_col2:
            st.slider(
                label=f"Lag ({temperature}°C)",
                min_value=float(1e-5),
                max_value=float(duration_h),
                step=1.0,
                key=_state_key(matrix_id, organism_id, temperature, "Lag"),
                label_visibility="collapsed",
            )

        with row_col3:
            st.slider(
                label=f"Maximum Rate ({temperature}°C)",
                min_value=float(1e-10),
                max_value=float(1.0),
                step=0.001,
                key=_state_key(matrix_id, organism_id, temperature, "Maximum Rate"),
                label_visibility="collapsed",
            )


# =============================================================================
# Main Streamlit app
# =============================================================================

def main() -> None:
    """
    Run the Streamlit application for manual tuning of Baranyi parameters.

    Layout
    ------
    - Top row:
        - growth duration input
        - MatrixID selector
        - load button
        - save button
    - Main area:
        - 4 columns, one per bacterium
        - each column contains:
            - the growth graph
            - the compact control grid for all temperatures

    Load behavior
    -------------
    When 'Charger' is clicked:
    1. a complete default dataframe is created for all possible
       MatrixID x OrganismID x Temperature combinations
    2. if BARANYI_OUTPUT_PATH exists, matching saved values replace defaults
    3. session-state values are updated

    Save behavior
    -------------
    When 'Enregistrer' is clicked:
    1. the current manual Baranyi parameters for the selected MatrixID are
       compiled and appended to `manual_baranyi_parameters.csv`
    2. Arrhenius models are fitted for each MatrixID x OrganismID combination,
       then appended to `manual_arrhenius_V2.csv`
    """
    st.set_page_config(layout="wide", page_title="Manual Baranyi tuning")
    st.title("Manual tuning of Baranyi parameters")

    initialize_session_state()

    duration_h, matrix_id, load_clicked, save_clicked = render_top_controls()

    if load_clicked:
        success, message = reload_reference_from_file()
        if success:
            st.success(message)
        else:
            st.error(message)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_organism_column("ec", duration_h, matrix_id)

    with col2:
        render_organism_column("lm", duration_h, matrix_id)

    with col3:
        render_organism_column("ss", duration_h, matrix_id)

    with col4:
        render_organism_column("ta", duration_h, matrix_id)

    if save_clicked:
        try:
            manual_baranyi_df = get_manual_primary_params(matrix_id)
            manual_arrhenius_df = build_arrhenius_dataframe(manual_baranyi_df)

            append_tsv(manual_baranyi_df, BARANYI_OUTPUT_PATH)
            append_tsv(manual_arrhenius_df, ARRHENIUS_OUTPUT_PATH)

            st.success(
                "Parameters successfully saved to "
                f"'{BARANYI_OUTPUT_PATH}' and '{ARRHENIUS_OUTPUT_PATH}'."
            )

            st.markdown("### Preview - manual_baranyi_parameters")
            st.dataframe(manual_baranyi_df, use_container_width=True)

            st.markdown("### Preview - manual_arrhenius_V2")
            st.dataframe(manual_arrhenius_df, use_container_width=True)

        except Exception as exc:
            st.error(f"Error while saving parameters: {exc}")


if __name__ == "__main__":
    main()
