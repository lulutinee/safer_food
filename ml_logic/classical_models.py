"""
ml_logic.classical_models
=========================

Pure mathematical implementations of classical predictive microbiology models.

This module contains stateless fitting and prediction functions for:

Primary growth models
---------------------
- Linear growth model
- Baranyi & Roberts (1994) isothermal growth model

Secondary models
----------------
- Arrhenius temperature model

Design philosophy
-----------------
- All functions are PURE (no estimator state).
- No dependency on scikit-learn.
- Inputs and outputs are NumPy-based.
- Each `fit_*` function returns a dictionary of parameters.
- Each `predict_*` function consumes that dictionary.

This separation allows:
- Clean sklearn estimator wrappers
- Independent unit testing
- Reuse outside sklearn pipelines
- Scientific clarity (model math isolated from orchestration)

Units
-----
Unless otherwise specified:

- Time: hours
- Temperature: °C (converted internally to Kelvin when required)
- Growth rate (mu_max): 1/hour (natural logarithm base)
- Concentrations: log10 units
- Activation energy Ea: J/mol
- Gas constant R: J/mol/K
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import curve_fit


# =============================================================================
# LINEAR MODEL
# =============================================================================

def fit_linear(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Fit a linear growth model.

    Model
    -----
        y = logC0 + mu * x

    where:
        logC0 : initial log10 concentration
        mu    : growth rate (slope)

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        Independent variable (e.g., time).
    y : array-like of shape (n_samples,)
        Observed log10 concentrations.

    Returns
    -------
    dict
        {
            "logC0": float,
            "mu": float
        }

    Notes
    -----
    - Uses numpy.polyfit (least squares).
    - No constraints or bounds applied.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    mu, logC0 = np.polyfit(x, y, deg=1)
    return {"logC0": float(logC0), "mu": float(mu)}


def predict_linear(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Predict using the linear growth model.
    """
    x = np.asarray(x, dtype=float).ravel()
    return params["logC0"] + params["mu"] * x


# =============================================================================
# BARANYI & ROBERTS (1994) PRIMARY MODEL
# =============================================================================

def _baranyi_model_reparam(
    t_: np.ndarray,
    log10N0: float,
    delta: float,
    mu_max: float,
    h0: float,
) -> np.ndarray:
    """
    Internal Baranyi model implementation using delta parameterization.

    Reparameterization
    ------------------
        log10Nmax = log10N0 + delta

    This ensures:
        log10Nmax >= log10N0

    Parameters
    ----------
    t_ : array-like
        Time values.
    log10N0 : float
        Initial concentration (log10).
    delta : float
        Difference between Nmax and N0 (must be > 0).
    mu_max : float
        Maximum specific growth rate (1/time, base e).
    h0 : float
        Physiological state parameter.

    Returns
    -------
    np.ndarray
        Predicted log10 concentrations.
    """
    ln10 = np.log(10.0)

    log10Nmax = log10N0 + delta
    y0 = ln10 * log10N0
    ymax = ln10 * log10Nmax

    # q0 = 1 / (exp(h0) - 1) with numerical stability
    denom = np.expm1(h0)
    denom = np.maximum(denom, 1e-12)
    q0 = 1.0 / denom

    A = t_ + (1.0 / mu_max) * np.log(
        (np.exp(-mu_max * t_) + q0) / (1.0 + q0)
    )

    # print(f'classical_models.py: {y0=}, {mu_max=}, {A=}, {ymax=}')

    y_ = y0 + mu_max * A - np.log(
        1.0 + (np.exp(mu_max * A) - 1.0) / np.exp(ymax - y0)
    )

    return y_ / ln10


def fit_baranyi(
    time: np.ndarray,
    logC: np.ndarray,
    *,
    maxfev: int = 20000,
) -> Dict[str, float]:
    """
    Fit the Baranyi & Roberts (1994) isothermal primary growth model.

    Reference
    ---------
    Baranyi, J., & Roberts, T. A. (1994).
    A dynamic approach to predicting bacterial growth in food.
    International Journal of Food Microbiology.

    Parameters
    ----------
    time : array-like of shape (n_samples,)
        Time points (hours).
    logC : array-like of shape (n_samples,)
        Observed log10 concentrations.
    maxfev : int, default=20000
        Maximum number of function evaluations for non-linear fitting.

    Returns
    -------
    dict
        {
            "Initial Value": log10(N0),
            "Lag": lag_time,
            "Maximum Rate": mu_max (1/hour, base e),
            "Final Value": log10(Nmax)
        }

    Implementation details
    ----------------------
    - Time values are automatically sorted.
    - Duplicate time points are averaged.
    - Multi-start optimization improves robustness.
    - Parameter bounds prevent non-physical solutions.
    - delta parameterization enforces Nmax >= N0.

    Notes
    -----
    - mu_max is expressed in natural log units (1/time).
    - Lag = h0 / mu_max.
    """
    t = np.asarray(time, dtype=float).ravel()
    y = np.asarray(logC, dtype=float).ravel()

    if t.size != y.size:
        raise ValueError("time and logC must have the same length.")
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
        raise ValueError("Inputs must not contain NaN or infinite values.")

    # Sort by time
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # Average duplicate time points
    tu, inv = np.unique(t, return_inverse=True)
    if tu.size != t.size:
        y_sum = np.zeros_like(tu)
        y_cnt = np.zeros_like(tu)
        np.add.at(y_sum, inv, y)
        np.add.at(y_cnt, inv, 1.0)
        y = y_sum / y_cnt
        t = tu

    ln10 = np.log(10.0)

    # Initial guesses
    log10N0_0 = float(np.min(y))
    log10Nmax_0 = float(np.max(y))
    delta_0 = max(1e-3, log10Nmax_0 - log10N0_0)

    dy = np.diff(y)
    dt = np.diff(t)
    slopes = dy / np.maximum(dt, 1e-12)

    slope_q = float(np.quantile(slopes[np.isfinite(slopes)], 0.90)) if slopes.size else 0.0
    mu0 = max(1e-4, slope_q * ln10)
    mu0 = min(mu0, 2.0)
    h0_0 = 1.0

    # Bounds
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    lower = (y_min - 5.0, 1e-3, 1e-4, 1e-3)
    upper = (y_max + 5.0, 20.0, 5.0, 200.0)

    mu_candidates = sorted(set(np.clip([mu0, mu0/2, mu0*2], lower[2], upper[2])))
    h0_candidates = sorted(set(np.clip([h0_0, 0.5, 2.0, 5.0], lower[3], upper[3])))

    best_popt: Tuple[float, float, float, float] | None = None
    best_sse = np.inf

    for mu_init in mu_candidates:
        for h0_init in h0_candidates:
            p0 = (log10N0_0, delta_0, mu_init, h0_init)
            try:
                popt, _ = curve_fit(
                    _baranyi_model_reparam,
                    t,
                    y,
                    p0=p0,
                    bounds=(lower, upper),
                    maxfev=int(maxfev),
                )
                yhat = _baranyi_model_reparam(t, *popt)
                sse = float(np.sum((y - yhat) ** 2))
                if sse < best_sse:
                    best_sse = sse
                    best_popt = tuple(float(v) for v in popt)
            except Exception:
                continue

    if best_popt is None:
        raise RuntimeError("Baranyi fit failed.")

    log10N0, delta, mu_max, h0 = best_popt
    log10Nmax = log10N0 + delta
    lag = h0 / mu_max

    return {
        "Initial Value": float(log10N0),
        "Lag": float(lag),
        "Maximum Rate": float(mu_max),
        "Final Value": float(log10Nmax),
    }


def predict_baranyi(time: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Predict log10 concentration using fitted Baranyi parameters.
    """
    t = np.asarray(time, dtype=float).ravel()

    log10N0 = float(params["Initial Value"])
    log10Nmax = float(params["Final Value"])
    mu_max = float(params["Maximum Rate"])
    lag = float(params["Lag"])

    delta = log10Nmax - log10N0
    h0 = mu_max * lag

    return _baranyi_model_reparam(t, log10N0, delta, mu_max, h0)


# =============================================================================
# ARRHENIUS SECONDARY MODEL
# =============================================================================

def fit_arrhenius(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_in_celsius: bool = True,
    R: float = 8.314462618,
) -> Dict[str, float]:
    """
    Fit Arrhenius temperature model.

    Model
    -----
        y(T) = A * exp(-Ea / (R*T))

    Linearized form
    ---------------
        ln(y) = ln(A) - (Ea/R) * (1/T)

    Parameters
    ----------
    x : array-like
        Temperature values (°C if x_in_celsius=True).
    y : array-like
        Positive response variable (e.g., mu_max).
    x_in_celsius : bool
        Convert °C to Kelvin if True.
    R : float
        Gas constant (J/mol/K).

    Returns
    -------
    dict
        {
            "A": float,
            "Ea": float,
            "R": float,
            "x_in_celsius": bool
        }
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if np.any(y <= 0):
        raise ValueError("Arrhenius fit requires strictly positive y values.")

    T = x + 273.15 if x_in_celsius else x
    if np.any(T <= 0):
        raise ValueError("Absolute temperature must be > 0 K.")

    X = 1.0 / T
    Y = np.log(y)

    slope, intercept = np.polyfit(X, Y, deg=1)

    Ea = -slope * R
    A = float(np.exp(intercept))

    return {
        "A": A,
        "Ea": float(Ea),
        "R": float(R),
        "x_in_celsius": bool(x_in_celsius),
    }


def predict_arrhenius(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Predict response using Arrhenius model.
    """
    x = np.asarray(x, dtype=float).ravel()

    A = float(params["A"])
    Ea = float(params["Ea"])
    R = float(params["R"])
    x_in_celsius = bool(params.get("x_in_celsius", True))

    T = x + 273.15 if x_in_celsius else x
    return A * np.exp(-Ea / (R * T))
