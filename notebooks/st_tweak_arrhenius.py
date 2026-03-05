# app_arrhenius_primary_predictions.py
from __future__ import annotations

# -----------------------------
# Robust import fix (works regardless of where you launch streamlit)
# -----------------------------
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # notebooks/ -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------
# Standard imports
# -----------------------------
import ast
import os
import datetime as dt
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from ml_logic.sk_baseline import ClassicalModelRegressor


# -----------------------------
# Constants
# -----------------------------
PRIMARY_PARAMS = ["Initial Value", "Lag", "Maximum Rate", "Final Value"]
DEFAULT_TEMP_CHOICES_C = [4.0, 6.0, 10.0, 20.0, 30.0]
DEFAULT_SELECTED_4_C = [4.0, 6.0, 10.0, 20.0]

MANUAL_SAVE_PATH = PROJECT_ROOT / "manual_arrhenius_parameters.csv"  # tab-separated


# -----------------------------
# Loading arrhenius_parameters (env path)
# -----------------------------
def load_arrhenius_df_from_env(project_root: Path) -> pd.DataFrame:
    load_dotenv()

    secondary_model_path = os.getenv("SECONDARY_MODEL_PATH")
    if secondary_model_path is None:
        raise ValueError("SECONDARY_MODEL_PATH is not defined in .env")

    csv_path = project_root / secondary_model_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path, sep="\t")
    return df


@st.cache_data(show_spinner=False)
def load_arrhenius_cached(project_root: Path) -> pd.DataFrame:
    df = load_arrhenius_df_from_env(project_root)

    required = ["MatrixID", "OrganismID", *PRIMARY_PARAMS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"arrhenius_parameters missing columns: {missing}")

    df = df.copy()
    df["MatrixID"] = df["MatrixID"].astype(str)
    df["OrganismID"] = df["OrganismID"].astype(str)

    # Convert dict-like strings to actual dicts (for the 4 primary-param columns)
    for p in PRIMARY_PARAMS:
        df[p] = df[p].apply(_coerce_to_dict)

    return df


def _coerce_to_dict(x: Any) -> dict:
    """
    Accepts:
      - dict -> dict
      - string like "{'A': ..., 'Ea': ...}" -> dict via literal_eval
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return {}
        try:
            v = ast.literal_eval(s)
            if isinstance(v, dict):
                return v
        except Exception:
            return {}
    return {}


# -----------------------------
# Extract + validate secondary params
# -----------------------------
def get_secondary_params_for_selection(
    df: pd.DataFrame,
    *,
    matrix_id: str,
    organism_id: str,
) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      { primary_param_name: {"A":..., "Ea":..., "R":..., "x_in_celsius":...}, ... }
    from the wide dataframe structure.
    """
    matrix_id = str(matrix_id)
    organism_id = str(organism_id)

    row_df = df[(df["MatrixID"] == matrix_id) & (df["OrganismID"] == organism_id)]
    if row_df.empty:
        raise ValueError(f"No row found for MatrixID={matrix_id}, OrganismID={organism_id}")
    if len(row_df) > 1:
        # keep first but warn
        row_df = row_df.iloc[[0]]

    row = row_df.iloc[0]
    out: Dict[str, Dict[str, float]] = {}

    for p in PRIMARY_PARAMS:
        d = row[p]
        if not isinstance(d, dict) or not d:
            raise ValueError(f"Missing dict for column '{p}' at MatrixID={matrix_id}, OrganismID={organism_id}")

        for k in ["A", "Ea", "R"]:
            if k not in d:
                raise ValueError(f"Missing key '{k}' in dict for '{p}' (MatrixID={matrix_id}, OrganismID={organism_id})")

        out[p] = {
            "A": float(d["A"]),
            "Ea": float(d["Ea"]),
            "R": float(d["R"]),
            "x_in_celsius": float(d.get("x_in_celsius", 1.0)),
        }

    return out


# -----------------------------
# Modeling helpers
# -----------------------------
def predict_primary_param_arrhenius(
    *,
    A: float,
    Ea: float,
    R: float,
    temperature_c: float,
    x_in_celsius: float = 1.0,
) -> float:
    """
    Uses ClassicalModelRegressor(model='arrhenius'), forcing params_ then predict.
    We pass temperature in °C because your dict indicates x_in_celsius = 1.0.
    """
    sec = ClassicalModelRegressor(model="arrhenius")
    sec.fit([1, 2, 3], [1, 2, 3])  # dummy fit required by your implementation
    sec.params_ = {"A": float(A), "Ea": float(Ea), "R": float(R), "x_in_celsius": float(x_in_celsius)}

    pred = sec.predict([float(temperature_c)])
    return float(np.asarray(pred).ravel()[0])


def compute_primary_params_at_temp(
    secondary_params: Dict[str, Dict[str, float]],
    temperature_c: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in PRIMARY_PARAMS:
        sp = secondary_params[p]
        out[p] = predict_primary_param_arrhenius(
            A=sp["A"], Ea=sp["Ea"], R=sp["R"], temperature_c=temperature_c, x_in_celsius=sp.get("x_in_celsius", 1.0)
        )
    return out


def predict_growth_baranyi(
    primary_params: Dict[str, float],
    times_h: np.ndarray,
) -> np.ndarray:
    prim = ClassicalModelRegressor(model="baranyi")
    prim.fit([1, 2, 3], [1, 2, 3])  # dummy fit required by your implementation
    prim.params_ = {k: float(v) for k, v in primary_params.items()}
    yhat = prim.predict(times_h)
    return np.asarray(yhat, dtype=float)


def make_growth_figure(times_h: np.ndarray, curves: Dict[float, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    for temp_c in sorted(curves.keys()):
        fig.add_trace(go.Scatter(x=times_h, y=curves[temp_c], mode="lines", name=f"{temp_c:g} °C"))
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Time (hours)",
        yaxis_title="Predicted logC (Baranyi)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# -----------------------------
# Save manual parameters (TSV upsert)
# -----------------------------
def upsert_manual_arrhenius_tsv(
    path: Path,
    *,
    matrix_id: str,
    organism_id: str,
    secondary_params: Dict[str, Dict[str, float]],
) -> None:
    """
    Saves one row per primary parameter:
      MatrixID, OrganismID, PrimaryParam, A, Ea, R, x_in_celsius, updated_at
    Upsert key: (MatrixID, OrganismID, PrimaryParam)
    """
    ts = dt.datetime.now().isoformat(timespec="seconds")
    rows = []
    for p in PRIMARY_PARAMS:
        sp = secondary_params[p]
        rows.append(
            {
                "MatrixID": str(matrix_id),
                "OrganismID": str(organism_id),
                "PrimaryParam": p,
                "A": float(sp["A"]),
                "Ea": float(sp["Ea"]),
                "R": float(sp["R"]),
                "x_in_celsius": float(sp.get("x_in_celsius", 1.0)),
                "updated_at": ts,
            }
        )
    new_df = pd.DataFrame(rows)

    if not path.exists():
        new_df.to_csv(path, sep="\t", index=False)
        return

    old = pd.read_csv(path, sep="\t")
    for col in ["MatrixID", "OrganismID", "PrimaryParam"]:
        if col not in old.columns:
            raise ValueError(f"{path.name} exists but is missing column '{col}'")

    old["MatrixID"] = old["MatrixID"].astype(str)
    old["OrganismID"] = old["OrganismID"].astype(str)
    old["PrimaryParam"] = old["PrimaryParam"].astype(str)

    merged = pd.concat([old, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["MatrixID", "OrganismID", "PrimaryParam"], keep="last")
    merged.to_csv(path, sep="\t", index=False)


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="Arrhenius → Baranyi simulator", layout="wide")
st.title("Arrhenius (secondary) → Baranyi (primary) — Simulation")

# Load data
try:
    arrhenius_df = load_arrhenius_cached(PROJECT_ROOT)
except Exception as e:
    st.error(f"Loading error: {type(e).__name__}: {e}")
    st.stop()

# Selectors
matrix_ids = sorted(arrhenius_df["MatrixID"].unique().tolist())
organism_ids = sorted(arrhenius_df["OrganismID"].unique().tolist())

c0, c1 = st.columns(2)
with c0:
    matrix_id = st.selectbox("MatrixID", options=matrix_ids, index=0)
with c1:
    organism_id = st.selectbox("OrganismID", options=organism_ids, index=0)

# Extract secondary params for selection
try:
    sec_params_default = get_secondary_params_for_selection(arrhenius_df, matrix_id=matrix_id, organism_id=organism_id)
except Exception as e:
    st.error(f"Parameter extraction error: {type(e).__name__}: {e}")
    st.stop()

# Store editable secondary params in session
state_key = f"sec::{matrix_id}::{organism_id}"
if state_key not in st.session_state:
    st.session_state[state_key] = sec_params_default

st.subheader("Temperatures (°C) — pick exactly 4")
temps_selected = st.multiselect(
    "Temperatures",
    options=DEFAULT_TEMP_CHOICES_C,
    default=DEFAULT_SELECTED_4_C,
)
if len(temps_selected) != 4:
    st.warning("Select **exactly 4** temperatures to enable calculation.")
temps_selected = [float(t) for t in temps_selected]

st.subheader("Time horizon (hours)")
tcol1, tcol2, tcol3 = st.columns([1, 1, 1])
with tcol1:
    t_start = st.number_input("Start (h)", min_value=0.0, value=0.0, step=1.0)
with tcol2:
    t_stop = st.number_input("Stop (h)", min_value=float(t_start), value=1000.0, step=10.0)
with tcol3:
    n_points = st.number_input("Points", min_value=50, max_value=5000, value=500, step=50)

times_h = np.linspace(float(t_start), float(t_stop), int(n_points))

# Editable secondary params UI
st.subheader("Secondary model parameters (Arrhenius) — editable")
edited: Dict[str, Dict[str, float]] = {}

for p in PRIMARY_PARAMS:
    st.markdown(f"**{p}**")
    a, ea, r, xin = st.columns([1, 1, 1, 1])
    cur = st.session_state[state_key][p]

    with a:
        A = st.number_input(f"A ({p})", value=float(cur["A"]), format="%.6g", key=f"{state_key}::{p}::A")
    with ea:
        Ea = st.number_input(f"Ea ({p})", value=float(cur["Ea"]), format="%.6g", key=f"{state_key}::{p}::Ea")
    with r:
        R = st.number_input(f"R ({p})", value=float(cur["R"]), format="%.6g", key=f"{state_key}::{p}::R")
    with xin:
        x_in_celsius = st.number_input(
            f"x_in_celsius ({p})", value=float(cur.get("x_in_celsius", 1.0)), format="%.3g", key=f"{state_key}::{p}::x_in_celsius"
        )

    edited[p] = {"A": float(A), "Ea": float(Ea), "R": float(R), "x_in_celsius": float(x_in_celsius)}

st.session_state[state_key] = edited

# Buttons
b1, b2, b3 = st.columns([1, 1, 2])
with b1:
    calc_clicked = st.button("Calculer", use_container_width=True, disabled=(len(temps_selected) != 4))
with b2:
    save_clicked = st.button("Enregistrer", use_container_width=True)
with b3:
    st.write(
        f"Selection: **MatrixID={matrix_id}**, **OrganismID={organism_id}**  \n"
        f"Manual save: `{MANUAL_SAVE_PATH}`"
    )

# Save action
if save_clicked:
    try:
        upsert_manual_arrhenius_tsv(
            MANUAL_SAVE_PATH,
            matrix_id=matrix_id,
            organism_id=organism_id,
            secondary_params=edited,
        )
        st.success("Saved (upsert) to manual_arrhenius_parameters.csv (tab-separated).")
    except Exception as e:
        st.error(f"Save error: {type(e).__name__}: {e}")

# Compute only on button click
if "result" not in st.session_state:
    st.session_state["result"] = None

if calc_clicked:
    try:
        primary_params_by_temp: Dict[float, Dict[str, float]] = {}
        curves: Dict[float, np.ndarray] = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for temp_c in temps_selected:
                pp = compute_primary_params_at_temp(edited, temp_c)
                primary_params_by_temp[temp_c] = pp
                curves[temp_c] = predict_growth_baranyi(pp, times_h)

        st.session_state["result"] = {
            "primary_params_by_temp": primary_params_by_temp,
            "curves": curves,
            "temps": temps_selected,
        }

    except Exception as e:
        st.session_state["result"] = None
        st.error(f"Computation error: {type(e).__name__}: {e}")

# Display results only after calculate
res = st.session_state.get("result")
if res is None:
    st.info("Click **Calculer** to compute Baranyi parameters and predicted growth curves.")
else:
    st.subheader("Predicted growth curves (Baranyi)")
    fig = make_growth_figure(times_h, res["curves"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Primary parameters (Baranyi) — computed at selected temperatures")
    rows = []
    for temp_c in sorted(res["primary_params_by_temp"].keys()):
        pp = res["primary_params_by_temp"][temp_c]
        rows.append({"Temperature (°C)": temp_c, **pp})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Secondary parameters (Arrhenius) — current editable values")
    rows2 = []
    for p in PRIMARY_PARAMS:
        sp = edited[p]
        rows2.append(
            {
                "PrimaryParam": p,
                "A": sp["A"],
                "Ea": sp["Ea"],
                "R": sp["R"],
                "x_in_celsius": sp.get("x_in_celsius", 1.0),
            }
        )
    st.dataframe(pd.DataFrame(rows2), use_container_width=True)
