from pathlib import Path
import streamlit.components.v1 as components

_frontend_dir = (Path(__file__).resolve().parent / "frontend").resolve()

_thermo = components.declare_component(
    "thermometer_slider",
    path=str(_frontend_dir),
)

def thermometer_slider(
    label: str,
    min_value: int = -5,
    max_value: int = 40,
    value: int = 4,
    step: int = 1,
    color: str = "#E14F3D",
    key: str = "thermo",
):
    out = _thermo(
        label=label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        color=color,
        key=key,
        default=value,
    )
    return int(out) if out is not None else int(value)
