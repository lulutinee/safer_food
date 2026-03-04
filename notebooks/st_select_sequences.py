# notebooks/st_select_sequences.py
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
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_logic.sk_baseline import ClassicalModelRegressor


# -----------------------------
# Config
# -----------------------------
DATA_PATH = PROJECT_ROOT / "notebooks" / "clean_df.csv"
VALIDATED_PATH = PROJECT_ROOT / "notebooks" / "séquences correctes.csv"

REQUIRED_COLS = [
    "ResponseID",
    "OrganismID",
    "MatrixID",
    "In_on",
    "Temperature",
    "Time",
    "Value_y",
]


# -----------------------------
# Data loading / indexing
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    df = pd.read_csv(path, sep="\t")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    df = df.copy()
    df["ResponseID"] = df["ResponseID"].astype(str)
    df["OrganismID"] = df["OrganismID"].astype(str)
    df["MatrixID"] = df["MatrixID"].astype(str)
    df["In_on"] = df["In_on"].astype(str)

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Value_y"] = pd.to_numeric(df["Value_y"], errors="coerce")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")

    df = df.dropna(subset=["ResponseID", "Time", "Value_y"])

    return df


@st.cache_data(show_spinner=False)
def build_sequence_index(df: pd.DataFrame) -> tuple[list[str], dict[str, pd.DataFrame]]:
    # preserve order of appearance
    response_ids = df["ResponseID"].drop_duplicates().tolist()

    seq_map: dict[str, pd.DataFrame] = {}
    for rid in response_ids:
        seq = df.loc[df["ResponseID"] == rid, REQUIRED_COLS].copy()
        seq = seq.sort_values("Time")
        seq_map[rid] = seq

    return response_ids, seq_map


# -----------------------------
# Validation file helpers
# -----------------------------
def read_validated_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        s = pd.read_csv(path)
        if "ResponseID" not in s.columns:
            # tolerate 1-col CSV without header
            if s.shape[1] == 1:
                s.columns = ["ResponseID"]
            else:
                return set()
        return set(s["ResponseID"].astype(str).dropna().unique().tolist())
    except Exception:
        return set()


def append_validated_id(path: Path, response_id: str) -> None:
    response_id = str(response_id)
    existing = read_validated_ids(path)
    if response_id in existing:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        pd.DataFrame({"ResponseID": [response_id]}).to_csv(path, index=False)
        return

    pd.DataFrame({"ResponseID": [response_id]}).to_csv(path, mode="a", index=False, header=False)


# -----------------------------
# Modeling + plot
# -----------------------------
def fit_and_predict_baranyi(
    seq: pd.DataFrame,
    *,
    n_grid: int = 300,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict, str]:
    """
    Fit Baranyi on (Time -> Value_y - baseline) and return:
      - seq_used with Value_sub
      - t_grid
      - yhat_grid
      - params (reg.params_)
      - warning_messages
    Raises Exception if fit/predict fails.
    """
    seq = seq.copy().sort_values("Time")

    baseline = float(seq["Value_y"].iloc[0])
    seq["Value_sub"] = seq["Value_y"] - baseline

    X = seq[["Time"]]
    y = seq["Value_sub"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        reg = ClassicalModelRegressor(model="baranyi")
        reg.fit(X, y)

        t_min = float(seq["Time"].min())
        t_max = float(seq["Time"].max())

        if t_min == t_max:
            t_grid = np.array([t_min, t_min + 1e-6], dtype=float)
        else:
            t_grid = np.linspace(t_min, t_max, n_grid)

        Xg = pd.DataFrame({"Time": t_grid})
        yhat_grid = reg.predict(Xg)

        params = getattr(reg, "params_", {}) or {}

        warning_messages = "; ".join(
            f"{type(warn.message).__name__}: {warn.message}" for warn in w
        )

    return seq, t_grid, np.asarray(yhat_grid, dtype=float), params, warning_messages


def make_figure(seq_used: pd.DataFrame, t_grid: np.ndarray, yhat_grid: np.ndarray) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=seq_used["Time"],
            y=seq_used["Value_sub"],
            mode="markers",
            name="Experimental (baseline-sub)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=yhat_grid,
            mode="lines",
            name="Baranyi model",
        )
    )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Time",
        yaxis_title="Value_y (baseline-sub)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="Sequence reviewer (Baranyi)", layout="wide")
st.title("Sequence reviewer — Baranyi fit (ClassicalModelRegressor)")

# Load and index
try:
    df = load_data(DATA_PATH)
    response_ids, seq_map = build_sequence_index(df)
except Exception as e:
    st.error(f"Data loading error: {type(e).__name__}: {e}")
    st.stop()

n_total = len(response_ids)
if n_total == 0:
    st.warning("No sequences found (ResponseID).")
    st.stop()

# Session state
if "idx" not in st.session_state:
    st.session_state.idx = 0

if "validated_cache" not in st.session_state:
    st.session_state.validated_cache = read_validated_ids(VALIDATED_PATH)

# Sidebar: navigation + jump
with st.sidebar:
    st.subheader("Navigation")
    st.write(f"Total sequences: **{n_total}**")

    jump = st.number_input(
        "Go to index (0-based)",
        min_value=0,
        max_value=n_total - 1,
        value=int(st.session_state.idx),
        step=1,
    )
    if int(jump) != int(st.session_state.idx):
        st.session_state.idx = int(jump)

    st.divider()
    st.subheader("Files")
    st.write(f"Data: `{DATA_PATH}`")
    st.write(f"Validated: `{VALIDATED_PATH}`")
    st.write(f"Validated count: **{len(st.session_state.validated_cache)}**")

# Current sequence
idx = int(st.session_state.idx)
rid = response_ids[idx]
seq = seq_map[rid]

# Metadata
meta = {
    "ResponseID": str(rid),
    "OrganismID": str(seq["OrganismID"].iloc[0]),
    "MatrixID": str(seq["MatrixID"].iloc[0]),
    "In_on": str(seq["In_on"].iloc[0]),
    "Temperature": seq["Temperature"].iloc[0],
    "n_points": int(len(seq)),
}
validated = meta["ResponseID"] in st.session_state.validated_cache

# Progress indicator (sequence number + total)
# (Human-friendly: 1-based position)
pos = idx + 1

top1, top2, top3, top4, top5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.6])
with top1:
    st.metric("Progress", f"{pos} / {n_total}")
with top2:
    st.metric("ResponseID", meta["ResponseID"])
with top3:
    st.metric("MatrixID", meta["MatrixID"])
with top4:
    st.metric("In_on", meta["In_on"])
with top5:
    st.write(
        f"**OrganismID:** {meta['OrganismID']}  \n"
        f"**Temperature:** {meta['Temperature']}  \n"
        f"**Points:** {meta['n_points']}  \n"
        f"**Already validated:** {'✅' if validated else '—'}"
    )

# Optional: visual progress bar
st.progress(pos / n_total)

# Buttons row
b1, b2, b3, b4 = st.columns([1, 1, 1.6, 1])
with b1:
    prev_clicked = st.button("⬅️ Previous", use_container_width=True, disabled=(idx == 0))
with b2:
    next_clicked = st.button("Next ➡️", use_container_width=True, disabled=(idx == n_total - 1))
with b3:
    validate_clicked = st.button("✅ Validate fit", use_container_width=True)
with b4:
    reload_clicked = st.button("🔄 Reload validated", use_container_width=True)

# Handle navigation
if prev_clicked:
    st.session_state.idx = max(0, idx - 1)
    st.rerun()

if next_clicked:
    st.session_state.idx = min(n_total - 1, idx + 1)
    st.rerun()

if reload_clicked:
    st.session_state.validated_cache = read_validated_ids(VALIDATED_PATH)
    st.rerun()

# Main content: plot + details
left, right = st.columns([1.7, 1.0], gap="large")

with left:
    st.subheader("Experimental points + model predictions")
    try:
        seq_used, t_grid, yhat_grid, params, warning_messages = fit_and_predict_baranyi(seq)
        fig = make_figure(seq_used, t_grid, yhat_grid)
        st.plotly_chart(fig, use_container_width=True)

        if warning_messages:
            st.warning(f"Warnings during fit: {warning_messages}")

    except Exception as e:
        st.error(f"Modeling error: {type(e).__name__}: {e}")

        # Still show points (baseline-sub) for debugging
        try:
            seq_dbg = seq.copy().sort_values("Time")
            baseline = float(seq_dbg["Value_y"].iloc[0])
            seq_dbg["Value_sub"] = seq_dbg["Value_y"] - baseline

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=seq_dbg["Time"],
                    y=seq_dbg["Value_sub"],
                    mode="markers",
                    name="Experimental (baseline-sub)",
                )
            )
            fig.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Time",
                yaxis_title="Value_y (baseline-sub)",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

with right:
    st.subheader("Details / actions")

    st.write("**Sequence preview (sorted by Time)**")
    st.dataframe(seq.sort_values("Time"), use_container_width=True, height=240)

    st.write("**Model parameters (if fit succeeded)**")
    try:
        if isinstance(params, dict) and len(params) > 0:
            safe_params = {}
            for k, v in params.items():
                if isinstance(v, (int, float, np.number)):
                    safe_params[k] = float(v)
                else:
                    safe_params[k] = v
            st.json(safe_params)
        else:
            st.info("No params available (fit failed or params_ empty).")
    except Exception:
        st.info("No params available (fit failed).")

    st.divider()

    if validate_clicked:
        try:
            append_validated_id(VALIDATED_PATH, meta["ResponseID"])
            st.session_state.validated_cache.add(meta["ResponseID"])
            st.success(f"Saved ResponseID {meta['ResponseID']} to {VALIDATED_PATH.name}")
        except Exception as e:
            st.error(f"Validation write error: {type(e).__name__}: {e}")
