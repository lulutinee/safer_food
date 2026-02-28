"""
ml_logic.sk_baseline
====================

Scikit-learn compatible wrapper around "classical" predictive microbiology models.

This module defines a single estimator, :class:`ClassicalModelRegressor`, that
exposes classical model fitting/prediction through the scikit-learn regressor API:

    - fit(X, y)
    - predict(X)

The actual model mathematics live in :mod:`ml_logic.classical_models` and are
implemented as pure (stateless) NumPy/SciPy functions. This wrapper is responsible
for:

- scikit-learn API compliance and learned attributes
- input validation and shape normalization
- compatibility with numpy arrays and pandas Series/DataFrames
- feature name capture / propagation (feature_names_in_, get_feature_names_out)
- storing learned parameters in `params_`

Supported models
----------------

Primary models (growth curves)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1) ``model="linear"``
   Simple linear model:

       y = logC0 + mu * x

   Typical use: simple growth/decline approximation or as a baseline model.

2) ``model="baranyi"``
   Baranyi & Roberts (1994) isothermal growth model, fit on (time, log10N).

   Returned parameters are stored in `params_` with keys:

       - "Initial Value"  : log10(N0)
       - "Lag"            : lag time (same time unit as x)
       - "Maximum Rate"   : mu_max (1/time, natural log base)
       - "Final Value"    : log10(Nmax)

Secondary models (environmental dependence)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
3) ``model="arrhenius"``
   Arrhenius model:

       y(T) = A * exp(-Ea / (R*T))

   Fit performed via linearization:

       ln(y) = ln(A) - (Ea/R) * (1/T)

   Requirements: y must be strictly positive.

Assumptions & constraints
-------------------------
- This estimator supports exactly ONE input feature.
  Therefore, X must represent a single explanatory variable (time, temperature, etc.).
  Accepted shapes:

    - (n_samples,)       for 1D array-like inputs or pandas Series
    - (n_samples, 1)     for 2D array-like inputs or pandas DataFrame (one column)

- For ``model="arrhenius"``:
  - If `x_in_celsius=True`, X is interpreted as °C and converted to Kelvin.
  - If `x_in_celsius=False`, X must already be in Kelvin (> 0).

- For ``model="baranyi"``:
  - Inputs can be unsorted and can contain duplicate time points; the model
    implementation sorts and averages duplicates internally.

Interoperability notes
----------------------
- scikit-learn's core API generally returns NumPy arrays.
  This estimator returns numpy.ndarray from predict().
- Feature names are captured during fit() if X is a pandas object
  and can be retrieved via get_feature_names_out().

Examples
--------
Linear model:

>>> reg = ClassicalModelRegressor(model="linear")
>>> reg.fit([0, 1, 2], [2.0, 2.5, 3.1])
>>> reg.params_
{'logC0': ..., 'mu': ...}

Baranyi model:

>>> reg = ClassicalModelRegressor(model="baranyi")
>>> reg.fit([0, 1, 2, 3, 4], [1.0, 1.4, 2.5, 5.0, 6.0])
>>> reg.params_["Lag"]
...

Arrhenius model:

>>> reg = ClassicalModelRegressor(model="arrhenius", x_in_celsius=True)
>>> reg.fit([5, 10, 15], [0.02, 0.05, 0.12])  # y must be > 0
>>> reg.params_["Ea"]
...

References
----------
- Baranyi, J. & Roberts, T. A. (1994). A dynamic approach to predicting bacterial growth in food.
  International Journal of Food Microbiology.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ml_logic.classical_models import (
    fit_linear,
    predict_linear,
    fit_baranyi,
    predict_baranyi,
    fit_arrhenius,
    predict_arrhenius,
)

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

ArrayLike = Union[np.ndarray, "pd.DataFrame", "pd.Series", Sequence[float]]


class ClassicalModelRegressor(BaseEstimator, RegressorMixin):
    """
    scikit-learn compatible regressor implementing classical models used in predictive microbiology.

    Parameters
    ----------
    model : {"linear", "baranyi", "arrhenius"}, default="linear"
        Which model family to fit.

        - "linear": y = logC0 + mu * x
        - "baranyi": Baranyi & Roberts (1994) primary growth model
        - "arrhenius": Arrhenius secondary model y(T) = A * exp(-Ea/(R*T))

    x_in_celsius : bool, default=True
        Only used when model="arrhenius".
        If True, X is interpreted as °C and converted internally to Kelvin.

    R : float, default=8.314462618
        Gas constant in J/(mol*K). Only used when model="arrhenius".

    maxfev : int, default=20000
        Maximum number of function evaluations for non-linear fitting.
        Only used when model="baranyi".

    Attributes (learned during fit)
    -------------------------------
    params_ : dict
        Learned parameters after calling fit(). Keys depend on `model`.

        - linear:    {"logC0": float, "mu": float}
        - baranyi:   {"Initial Value": float, "Lag": float, "Maximum Rate": float, "Final Value": float}
        - arrhenius: {"A": float, "Ea": float, "R": float, "x_in_celsius": bool}

    model_ : str
        The fitted model name (equal to `model` used for fit()).

    n_features_in_ : int
        Number of features seen during fit(). This estimator supports exactly 1.

    feature_names_in_ : np.ndarray of dtype object, optional
        Feature name(s) seen during fit() if X was a pandas Series/DataFrame.

    Notes
    -----
    - This class intentionally supports a single feature to keep semantics explicit.
      If you need multi-factor secondary models (e.g., T + pH + a_w), use a different
      estimator (e.g., linear regression on engineered features, GAM, neural nets,
      or a custom multi-input model).
    """

    def __init__(
        self,
        model: str = "linear",
        *,
        x_in_celsius: bool = True,
        R: float = 8.314462618,
        maxfev: int = 20000,
    ) -> None:
        self.model = model
        self.x_in_celsius = x_in_celsius
        self.R = R
        self.maxfev = maxfev

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike, y: ArrayLike) -> "ClassicalModelRegressor":
        """
        Fit the selected model y = f(x).

        Parameters
        ----------
        X : array-like, pandas Series, or pandas DataFrame
            Explanatory variable (time, temperature, etc.).
            Accepted shapes:
              - (n_samples,)   for 1D inputs
              - (n_samples, 1) for 2D inputs / DataFrame with one column

        y : array-like of shape (n_samples,)
            Target variable.

        Returns
        -------
        self : ClassicalModelRegressor
            Fitted estimator.

        Raises
        ------
        ValueError
            If X does not contain exactly one feature, or if shapes mismatch,
            or if model-specific constraints are violated (e.g., Arrhenius y <= 0).
        """
        X_np = self._as_2d_numpy_and_capture_feature_names(X, fitting=True)

        # sklearn validation: ensures consistent length, numeric coercion, etc.
        X_np, y_np = check_X_y(X_np, y, ensure_2d=True, dtype=float)

        if X_np.shape[1] != 1:
            raise ValueError(
                f"This estimator supports exactly 1 feature, got X with shape {X_np.shape}."
            )

        x = X_np[:, 0].astype(float)
        y_np = np.asarray(y_np, dtype=float)

        # Dispatch to model-specific pure functions
        if self.model == "linear":
            params = fit_linear(x, y_np)

        elif self.model == "baranyi":
            params = fit_baranyi(x, y_np, maxfev=self.maxfev)

        elif self.model == "arrhenius":
            params = fit_arrhenius(x, y_np, x_in_celsius=self.x_in_celsius, R=self.R)

        else:
            raise ValueError(
                f"Unknown model='{self.model}'. Expected 'linear', 'baranyi', or 'arrhenius'."
            )

        # sklearn learned attributes
        self.params_ = params
        self.model_ = self.model
        self.n_features_in_ = 1

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like, pandas Series, or pandas DataFrame
            Explanatory variable values.
            Accepted shapes:
              - (n_samples,)   for 1D inputs
              - (n_samples, 1) for 2D inputs / DataFrame with one column

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted values.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before fit().
        ValueError
            If X does not contain exactly one feature.
        """
        check_is_fitted(self, attributes=["params_", "model_"])

        X_np = self._as_2d_numpy_and_capture_feature_names(X, fitting=False)
        X_np = check_array(X_np, ensure_2d=True, dtype=float)

        if X_np.shape[1] != 1:
            raise ValueError(
                f"This estimator supports exactly 1 feature, got X with shape {X_np.shape}."
            )

        x = X_np[:, 0].astype(float)

        if self.model_ == "linear":
            y_pred = predict_linear(x, self.params_)

        elif self.model_ == "baranyi":
            y_pred = predict_baranyi(x, self.params_)

        elif self.model_ == "arrhenius":
            y_pred = predict_arrhenius(x, self.params_)

        else:
            raise RuntimeError(f"Unexpected fitted model_='{self.model_}'.")

        return np.asarray(y_pred, dtype=float)

    def get_feature_names_out(
        self,
        input_features: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Return output feature names for feature-name propagation.

        Even though regressors typically do not transform features, this method is useful
        for sklearn pipelines (e.g., ColumnTransformer) and for introspection. Since this
        estimator does not create derived features, it returns the single input feature name.

        Parameters
        ----------
        input_features : sequence of str, optional
            If provided, must have length 1 and will be returned.

        Returns
        -------
        np.ndarray of dtype object
            Array containing exactly one feature name.

        Notes
        -----
        - If `input_features` is not provided:
          - returns `feature_names_in_` if available (pandas input during fit)
          - otherwise returns ["x0"]
        """
        if input_features is not None:
            feats = list(input_features)
            n_expected = getattr(self, "n_features_in_", 1)
            if len(feats) != n_expected:
                raise ValueError(
                    f"input_features has length {len(feats)} but expected {n_expected}."
                )
            return np.asarray(feats, dtype=object)

        if hasattr(self, "feature_names_in_"):
            return np.asarray(self.feature_names_in_, dtype=object)

        return np.asarray(["x0"], dtype=object)

    # ------------------------------------------------------------------
    # Input normalization + pandas feature name handling
    # ------------------------------------------------------------------
    def _as_2d_numpy_and_capture_feature_names(self, X: Any, *, fitting: bool) -> np.ndarray:
        """
        Normalize X into a 2D numpy array suitable for sklearn validation.

        Behavior
        --------
        - If X is a pandas DataFrame:
            - requires exactly 1 column
            - captures column name into feature_names_in_ during fit
            - returns X.to_numpy() with shape (n_samples, 1)
        - If X is a pandas Series:
            - captures Series.name into feature_names_in_ during fit (defaults to "x0")
            - returns shape (n_samples, 1)
        - Otherwise:
            - converts to np.asarray(dtype=float)
            - reshapes 1D inputs to (n_samples, 1)

        Parameters
        ----------
        X : Any
            Input data passed to fit/predict.
        fitting : bool
            True when called by fit(), False when called by predict().

        Returns
        -------
        np.ndarray
            2D array with shape (n_samples, 1).

        Raises
        ------
        ValueError
            If X has invalid dimensionality or (for DataFrame) wrong number of columns.
        """
        if pd is not None:
            if isinstance(X, pd.DataFrame):
                if X.shape[1] != 1:
                    raise ValueError(
                        f"Expected a DataFrame with exactly 1 column, got {X.shape[1]}."
                    )
                if fitting:
                    self.feature_names_in_ = np.asarray(X.columns, dtype=object)
                return X.to_numpy()

            if isinstance(X, pd.Series):
                if fitting:
                    name = X.name if X.name is not None else "x0"
                    self.feature_names_in_ = np.asarray([name], dtype=object)
                return X.to_numpy().reshape(-1, 1)

        X_arr = np.asarray(X, dtype=float)

        if X_arr.ndim == 1:
            return X_arr.reshape(-1, 1)
        if X_arr.ndim == 2:
            return X_arr
        raise ValueError(f"X must be 1D or 2D. Got array with ndim={X_arr.ndim}.")


# class ClassicalModelRegressor(BaseEstimator, RegressorMixin):
#     """
#     A scikit-learn compatible regressor implementing several classical models.

#     Parameters
#     ----------
#     model : {"linear", "baranyi", "arrhenius"}, default="linear"
#         Model family to fit.

#         - "linear":    y = logC0 + mu * x
#         - "baranyi":   Baranyi & Roberts (1994) primary growth model, fitted on (time, log10N)
#         - "arrhenius": Arrhenius secondary model: y(T) = A * exp(-Ea / (R*T))

#     x_in_celsius : bool, default=True
#         Only used for model="arrhenius".
#         If True, X is interpreted as °C and converted to Kelvin.

#     R : float, default=8.314462618
#         Gas constant in J/(mol*K). Only used for model="arrhenius".

#     maxfev : int, default=20000
#         Maximum number of function evaluations for non-linear curve fitting.
#         Only used for model="baranyi".

#     Attributes
#     ----------
#     params_ : dict
#         Learned parameters after calling fit(). Keys depend on `model`.
#         - linear:    {"logC0": float, "mu": float}
#         - baranyi:   {"Initial Value": float, "Lag": float, "Maximum Rate": float, "Final Value": float}
#         - arrhenius: {"A": float, "Ea": float, "R": float, "x_in_celsius": bool}

#     model_ : str
#         The fitted model name (same as `model` used in fit).

#     n_features_in_ : int
#         Number of features seen during fit. This estimator supports exactly 1.

#     feature_names_in_ : np.ndarray of dtype object, optional
#         Input feature names seen during fit if X was a pandas object.

#     Notes
#     -----
#     - This estimator supports exactly one explanatory variable. Therefore, X must be
#       of shape (n_samples, 1) (DataFrame with one column, or Series, or a 1D array-like).
#     """

#     def __init__(
#         self,
#         model: str = "linear",
#         *,
#         x_in_celsius: bool = True,
#         R: float = 8.314462618,
#         maxfev: int = 20000,
#     ) -> None:
#         self.model = model
#         self.x_in_celsius = x_in_celsius
#         self.R = R
#         self.maxfev = maxfev

#     # ------------------------------------------------------------------
#     # Public sklearn API
#     # ------------------------------------------------------------------
#     def fit(self, X: ArrayLike, y: ArrayLike) -> "ClassicalModelRegressor":
#         """
#         Fit the selected model y = f(x).

#         Parameters
#         ----------
#         X : array-like, pandas Series, or pandas DataFrame
#             Explanatory variable (e.g., time in hours, temperature in °C).
#             Expected shape:
#               - (n_samples,) if 1D array-like or Series
#               - (n_samples, 1) if 2D array-like or DataFrame (one column)
#         y : array-like
#             Target variable of shape (n_samples,).

#         Returns
#         -------
#         self : ClassicalModelRegressor
#             Fitted estimator.
#         """
#         # Capture feature names if pandas input
#         X_np = self._as_2d_numpy_and_capture_feature_names(X, fitting=True)

#         X_np, y_np = check_X_y(X_np, y, ensure_2d=True, dtype=float)
#         if X_np.shape[1] != 1:
#             raise ValueError(
#                 f"This estimator supports exactly 1 feature, got X with shape {X_np.shape}."
#             )

#         x = X_np[:, 0].astype(float)
#         y_np = np.asarray(y_np, dtype=float)

#         if x.shape[0] != y_np.shape[0]:
#             raise ValueError("X and y must have the same number of samples.")

#         # Fit according to selected model
#         if self.model == "linear":
#             params = self._fit_linear(x, y_np)

#         elif self.model == "baranyi":
#             params = self._fit_baranyi(self, x, y_np)

#         elif self.model == "arrhenius":
#             params = self._fit_arrhenius(
#                 x, y_np, x_in_celsius=self.x_in_celsius, R=self.R
#             )

#         else:
#             raise ValueError(
#                 f"Unknown model='{self.model}'. Expected 'linear', 'baranyi', or 'arrhenius'."
#             )

#         # sklearn conventions for learned attributes
#         self.params_ = params
#         self.model_ = self.model
#         self.n_features_in_ = 1
#         return self

#     def predict(self, X: ArrayLike) -> np.ndarray:
#         """
#         Predict using the fitted model.

#         Parameters
#         ----------
#         X : array-like, pandas Series, or pandas DataFrame
#             Explanatory variable (e.g., time in hours, temperature in °C).
#             Expected shape:
#               - (n_samples,) if 1D array-like or Series
#               - (n_samples, 1) if 2D array-like or DataFrame (one column)

#         Returns
#         -------
#         y_pred : np.ndarray of shape (n_samples,)
#             Predicted target values.
#         """
#         check_is_fitted(self, attributes=["params_", "model_"])

#         X_np = self._as_2d_numpy_and_capture_feature_names(X, fitting=False)
#         X_np = check_array(X_np, ensure_2d=True, dtype=float)

#         if X_np.shape[1] != 1:
#             raise ValueError(
#                 f"This estimator supports exactly 1 feature, got X with shape {X_np.shape}."
#             )

#         x = X_np[:, 0].astype(float)

#         if self.model_ == "linear":
#             y_pred = self._predict_linear(x, self.params_)

#         elif self.model_ == "baranyi":
#             y_pred = self._predict_baranyi(x, self.params_)

#         elif self.model_ == "arrhenius":
#             y_pred = self._predict_arrhenius(x, self.params_)

#         else:
#             raise RuntimeError(f"Unexpected fitted model_='{self.model_}'.")

#         return np.asarray(y_pred, dtype=float)

#     def get_feature_names_out(
#         self,
#         input_features: Optional[Sequence[str]] = None,
#     ) -> np.ndarray:
#         """
#         Return output feature names.

#         Notes
#         -----
#         For regressors, this method is not heavily used by sklearn itself, but it is
#         coherent for feature-name propagation in pipelines and for model introspection.
#         This estimator does not create new features; it returns the input feature name.

#         Parameters
#         ----------
#         input_features : sequence of str, optional
#             If provided, must match the number of features (1) and will be returned.

#         Returns
#         -------
#         feature_names_out : np.ndarray of dtype object
#             Feature name(s) of the input.
#         """
#         if input_features is not None:
#             input_features = list(input_features)
#             if hasattr(self, "n_features_in_") and len(input_features) != self.n_features_in_:
#                 raise ValueError(
#                     f"input_features has length {len(input_features)} but expected {self.n_features_in_}."
#                 )
#             return np.asarray(input_features, dtype=object)

#         if hasattr(self, "feature_names_in_"):
#             return np.asarray(self.feature_names_in_, dtype=object)

#         n = getattr(self, "n_features_in_", 1)
#         return np.asarray([f"x{i}" for i in range(n)], dtype=object)

#     # ------------------------------------------------------------------
#     # Helpers: input normalization + feature name capture
#     # ------------------------------------------------------------------
#     def _as_2d_numpy_and_capture_feature_names(self, X: Any, *, fitting: bool) -> np.ndarray:
#         """
#         Normalize X into a 2D numpy array of shape (n_samples, 1) when possible.

#         Also captures feature names during fit when X is a pandas Series/DataFrame.

#         Parameters
#         ----------
#         X : Any
#             Input data.
#         fitting : bool
#             True when called from fit(), False when called from predict().

#         Returns
#         -------
#         X_np : np.ndarray
#             2D array representation of X.
#         """
#         if pd is not None:
#             if isinstance(X, pd.DataFrame):
#                 if X.shape[1] != 1:
#                     # Let downstream validation raise a more standard error too,
#                     # but this message is clearer for DataFrame users.
#                     raise ValueError(
#                         f"Expected a DataFrame with exactly 1 column, got {X.shape[1]}."
#                     )
#                 if fitting:
#                     self.feature_names_in_ = np.asarray(X.columns, dtype=object)
#                 return X.to_numpy()

#             if isinstance(X, pd.Series):
#                 if fitting:
#                     name = X.name if X.name is not None else "x0"
#                     self.feature_names_in_ = np.asarray([name], dtype=object)
#                 return X.to_numpy().reshape(-1, 1)

#         # Non-pandas inputs
#         X_arr = np.asarray(X, dtype=float)

#         if X_arr.ndim == 1:
#             return X_arr.reshape(-1, 1)
#         if X_arr.ndim == 2:
#             return X_arr
#         raise ValueError(f"X must be 1D or 2D. Got array with ndim={X_arr.ndim}.")

#     # ------------------------------------------------------------------
#     # Model implementations
#     # ------------------------------------------------------------------
#     @staticmethod
#     def _fit_linear(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
#         """
#         Linear model (primary or secondary):
#             y = logC0 + mu * x

#         Returns
#         -------
#         params : dict
#             {"logC0": float, "mu": float}
#         """
#         mu, logC0 = np.polyfit(x, y, deg=1)
#         return {"logC0": float(logC0), "mu": float(mu)}

#     @staticmethod
#     def _predict_linear(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
#         """
#         Predict with linear model:
#             y = logC0 + mu * x
#         """
#         return params["logC0"] + params["mu"] * x

#     @staticmethod
#     def _fit_baranyi(self, time: np.ndarray, logC: np.ndarray) -> None:
#         """
#         Fit the isothermal Baranyi (1994) primary growth model.

#         Inputs
#         ------
#         time : array-like
#             Time points (hours). Must be 1D.
#         logC : array-like
#             Observed log10 concentrations at corresponding times. Must be 1D.

#         Output (stored in self.params)
#         ------------------------------
#         {
#             'Initial Value': log10(N0),
#             'Lag'          : lag time (hours),
#             'Maximum Rate' : mu_max (1/hour, natural-log base),
#             'Final Value'  : log10(Nmax)
#         }

#         Notes
#         -----
#         This implementation is designed to be robust:
#         - averages duplicate time points (replicates)
#         - uses robust initial guesses
#         - uses multi-start optimization
#         - enforces log10Nmax >= log10N0 via delta parameterization
#         - uses bounds tailored to hours to prevent non-physical runaway fits
#         """

#         # -------------------- sanitize inputs --------------------
#         t = np.asarray(time, dtype=float)
#         y = np.asarray(logC, dtype=float)

#         if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
#             raise ValueError("time and logC must be 1D arrays of the same length.")
#         if np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
#             raise ValueError("time and logC must not contain NaN/inf.")
#         if np.any(np.diff(t) < 0):
#             raise ValueError("time must be sorted in increasing order.")

#         ln10 = np.log(10.0)

#         # -------------------- handle duplicate time points --------------------
#         # If you have replicates at the same time, average them to avoid dt ~ 0 issues.
#         # This significantly reduces unstable slope estimates and optimizer failures.
#         tu, inv = np.unique(t, return_inverse=True)
#         if tu.size != t.size:
#             y_sum = np.zeros_like(tu)
#             y_cnt = np.zeros_like(tu)
#             np.add.at(y_sum, inv, y)
#             np.add.at(y_cnt, inv, 1.0)
#             y = y_sum / y_cnt
#             t = tu

#         # -------------------- model (reparameterized) --------------------
#         # Use delta = log10Nmax - log10N0, constrained > 0, so Nmax >= N0 always.
#         def baranyi_model_reparam(t_, log10N0, delta, mu_max, h0):
#             log10Nmax = log10N0 + delta

#             y0 = ln10 * log10N0
#             ymax = ln10 * log10Nmax

#             # q0 = 1 / (exp(h0) - 1)
#             # h0 close to 0 => exp(h0)-1 ~ 0 => q0 huge (can destabilize)
#             q0 = 1.0 / (np.exp(h0) - 1.0)

#             A = t_ + (1.0 / mu_max) * np.log(
#                 (np.exp(-mu_max * t_) + q0) / (1.0 + q0)
#             )

#             y_ = y0 + mu_max * A - np.log(
#                 1.0 + (np.exp(mu_max * A) - 1.0) / np.exp(ymax - y0)
#             )

#             return y_ / ln10

#         # -------------------- robust initial guesses --------------------
#         # Initial level guesses (robust-ish using min/max)
#         log10N0_0 = float(np.min(y))
#         log10Nmax_0 = float(np.max(y))
#         delta_0 = max(1e-3, log10Nmax_0 - log10N0_0)

#         # Robust slope-based mu guess (ignore dt==0 already handled by dedup)
#         dy = np.diff(y)
#         dt = np.diff(t)
#         slopes = dy / np.maximum(dt, 1e-12)  # safe guard

#         # Use a high quantile rather than max to avoid spikes from noisy points
#         slope_q = float(np.quantile(slopes[np.isfinite(slopes)], 0.90)) if slopes.size else 0.0
#         mu0 = max(1e-4, slope_q * ln10)  # 1/hour (base e)
#         mu0 = min(mu0, 2.0)             # reasonable cap for hours (tune if needed)

#         h0_0 = 1.0  # typical starting point; h0 = mu * lag

#         # -------------------- bounds (hours) --------------------
#         # These are broad but prevent runaway nonsense like Nmax=1311 or Lag=1e10 hours.
#         #
#         # Typical mu_max (1/h) often ~ 0.01 to 2, sometimes higher; we allow up to 5.
#         # We keep mu_max >= 1e-4 to avoid huge (1/mu) blow-ups and astronomical lag.
#         #
#         # Lag = h0/mu. With mu >= 1e-4 and h0 <= 200 => lag <= 2,000,000 hours (still huge),
#         # but in practice optimizer will stay in sensible region. You can tighten h0 if desired.
#         #
#         # delta limits keep Nmax from exploding.
#         y_min = float(np.min(y))
#         y_max = float(np.max(y))

#         lower = (y_min - 5.0, 1e-3, 1e-4, 1e-3)     # (log10N0, delta, mu, h0)
#         upper = (y_max + 5.0, 20.0, 5.0, 200.0)

#         # -------------------- multi-start (robust fitting) --------------------
#         # Try a small grid of initial (mu, h0) values around robust guesses.
#         mu_candidates = [mu0, mu0/2, mu0*2, mu0/5, mu0*5]
#         mu_candidates = [float(np.clip(m, lower[2], upper[2])) for m in mu_candidates]
#         mu_candidates = sorted(set(mu_candidates))

#         h0_candidates = [h0_0, 0.5, 2.0, 5.0, 10.0]
#         h0_candidates = [float(np.clip(h, lower[3], upper[3])) for h in h0_candidates]
#         h0_candidates = sorted(set(h0_candidates))

#         best_popt = None
#         best_sse = np.inf

#         # Helper: skip initial guesses that produce non-finite model outputs
#         def _finite_at_p0(p0):
#             try:
#                 yp = baranyi_model_reparam(t, *p0)
#                 return np.all(np.isfinite(yp))
#             except Exception:
#                 return False

#         for mu_init in mu_candidates:
#             for h0_init in h0_candidates:
#                 p0 = (log10N0_0, delta_0, mu_init, h0_init)

#                 if not _finite_at_p0(p0):
#                     continue

#                 try:
#                     popt, _ = curve_fit(
#                         baranyi_model_reparam,
#                         t, y,
#                         p0=p0,
#                         bounds=(lower, upper),
#                         maxfev=30000
#                     )

#                     yhat = baranyi_model_reparam(t, *popt)
#                     if not np.all(np.isfinite(yhat)):
#                         continue

#                     resid = y - yhat
#                     sse = float(np.sum(resid * resid))
#                     if sse < best_sse:
#                         best_sse = sse
#                         best_popt = popt

#                 except Exception:
#                     # try the next initialization
#                     continue

#         if best_popt is None:
#             raise RuntimeError("Baranyi fit failed: all multi-start initializations failed.")

#         # -------------------- unpack + compute lag --------------------
#         log10N0, delta, mu_max, h0 = best_popt
#         log10Nmax = log10N0 + delta
#         lag = h0 / mu_max  # hours

#         self.params = {
#             "Initial Value": float(log10N0),
#             "Lag": float(lag),
#             "Maximum Rate": float(mu_max),
#             "Final Value": float(log10Nmax),
#         }

#     @staticmethod
#     def _predict_baranyi(time: np.ndarray, params: Dict[str, float]) -> np.ndarray:
#         """
#         Predict log10 concentration using Baranyi parameters.
#         """
#         t = np.asarray(time, dtype=float)

#         log10N0 = params["Initial Value"]
#         log10Nmax = params["Final Value"]
#         mu_max = params["Maximum Rate"]
#         lag = params["Lag"]

#         ln10 = np.log(10.0)
#         y0 = ln10 * log10N0
#         ymax = ln10 * log10Nmax

#         h0 = mu_max * lag
#         q0 = 1.0 / (np.exp(h0) - 1.0)

#         A = t + (1.0 / mu_max) * np.log(
#             (np.exp(-mu_max * t) + q0) / (1.0 + q0)
#         )

#         y = y0 + mu_max * A - np.log(
#             1.0 + (np.exp(mu_max * A) - 1.0) / np.exp(ymax - y0)
#         )

#         return y / ln10

#     @staticmethod
#     def _fit_arrhenius(
#         x: np.ndarray,
#         y: np.ndarray,
#         *,
#         x_in_celsius: bool,
#         R: float,
#     ) -> Dict[str, float]:
#         """
#         Fit Arrhenius secondary model:
#             y(T) = A * exp(-Ea / (R*T))

#         Uses linearization:
#             ln(y) = ln(A) - (Ea/R) * (1/T)

#         Requirements: y must be strictly positive.

#         Parameters
#         ----------
#         x : np.ndarray
#             Temperature values (°C if x_in_celsius=True, else K).
#         y : np.ndarray
#             Positive values to fit (e.g., mu_max).
#         x_in_celsius : bool
#             Interpret x in °C (convert to K) if True.
#         R : float
#             Gas constant (J/mol/K).

#         Returns
#         -------
#         params : dict
#             {"A": float, "Ea": float, "R": float, "x_in_celsius": bool}
#         """
#         x = np.asarray(x, dtype=float)
#         y = np.asarray(y, dtype=float)

#         if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
#             raise ValueError("x and y must be 1D arrays of the same length.")
#         if np.any(y <= 0):
#             raise ValueError("Arrhenius fit requires y > 0 because ln(y) is used.")

#         T = x + 273.15 if x_in_celsius else x
#         if np.any(T <= 0):
#             raise ValueError("Invalid absolute temperature (K) <= 0 encountered.")

#         X = 1.0 / T
#         Y = np.log(y)

#         slope, intercept = np.polyfit(X, Y, deg=1)
#         Ea = -slope * R
#         A = float(np.exp(intercept))

#         return {
#             "A": A,
#             "Ea": float(Ea),          # J/mol
#             "R": float(R),
#             "x_in_celsius": bool(x_in_celsius),
#         }

#     @staticmethod
#     def _predict_arrhenius(x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
#         """
#         Predict with Arrhenius model:
#             y = A * exp(-Ea/(R*T))
#         """
#         A = params["A"]
#         Ea = params["Ea"]
#         R = params["R"]
#         x_in_celsius = params.get("x_in_celsius", True)

#         x = np.asarray(x, dtype=float)
#         T = x + 273.15 if x_in_celsius else x
#         return A * np.exp(-Ea / (R * T))
