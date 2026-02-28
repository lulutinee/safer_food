"""
Docstring for ml_logic.baseline
Modèles de base (modèles classiques)
"""


# Imports
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

from typing import List, Union

class ClassicalModel:
    """
    Docstring for ClassicalModel
    Classical models
        - ClassicalModel() : creates an instance of a classical model
        - .fit() : "training" : identifies model parameters
        - .predict() : predicts bacterial load LogC = f(time)
        - .params : dict of the model parameters {'parameter': value}
        - .model : model type (str). See ClassicalModel.fit() documentation


    """
    def __init__(self) -> None:
        """
        Docstring for __init__
        ClassicalModel() : creates an instance of a classical model
        """
        self.params: dict | None = None
        self.model: str | None = None

    def fit(
        self,
        x: List[float],
        y: List[float],
        model: str = "linear",
    ) -> None:
        """
        fits a model y = f(x)
        Primary models :
            Fits logC = f(time)
        Secondary models :
            Fits primary model parameter y as a function of variable x

        Parameters
        ----------
        x : list[float] : time in hours, temperature in °C, etc.
        y : list[float] : bacterial concentration in log10 cfu /g or /ml,
            growth rate in log/min, etc.
        model : str, default "linear"
            Primary models :
                'linear'    : linear model
                'baranyi'   : full Baranyi model, Baranyi and Roberts (1994)
            Secondary models :
                'linear'    : linear model
                'arrhenius' : Arrhenius model

        If you want to accurately predict bacterial growth : standardize your bacterial loads
            such as bacterial load at t=0 is 1 cfu (log t0 = 0)

        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.shape != y.shape:
            raise ValueError("time and logC should have the same length")

        self.model = model

        if model == "linear":
            self.model = model
            self._fit_linear(x, y)

        elif model == 'baranyi':
            self.model = model
            self._fit_baranyi(x, y)

        elif model == 'arrhenius':
            self.model = model
            self._fit_arrhenius(x, y)

        else:
            raise NotImplementedError(
                f"Model '{model}' not implemented"
            )


    def predict(self, x: Union[float, List[float]]) -> Union[float, List[float]]:
        """
        Docstring for predict
        Primary models :
            predicts y : bacterial load(s) at  : time(s)
        Secondary models :
            predicts primary model parameter y as a function of variable
        The model instance has to be fitted (.fit method) before prediction

        x : can be a single value (float) or a list of time points

        returns : y for every x ex. bacterial load = f(time), or parameter y = f(variable x)
        Returns one float if x = float, and a list if x = list
        """
        if self.params is None or self.model is None:
            raise RuntimeError("Model has to be trained with fit() before predict().")

        if self.model == "linear":
            return self._predict_linear(x)

        elif self.model == 'baranyi':
            return self._predict_baranyi(x)

        elif self.model == 'arrhenius':
            return self._predict_arrhenius(x)

        else:
            raise NotImplementedError(
                f"Prediction not implemented for model '{self.model}'"
            )

    #Linear model
    def _fit_linear(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Régression linéaire :
        logC = logC0 + mu * time
        """
        mu, logC0 = np.polyfit(x, y, deg=1)

        self.params = {
            "logC0": float(logC0),
            "mu": float(mu),
        }
    def _predict_linear(self, time: Union[float, List[float]]) -> Union[float, List[float]]:
        """
        Régression linéaire :
        logC = logC0 + mu * time
        """
        logC0 = self.params["logC0"]
        mu = self.params["mu"]

        if isinstance(time, list):
            return [logC0 + mu * t for t in time]
        else:
            return logC0 + mu * time

    #Baranyi et Roberts 1994
    def _fit_baranyi(self, time: np.ndarray, logC: np.ndarray) -> None:
        """
        Fit the isothermal Baranyi (1994) primary growth model.

        Inputs
        ------
        time : array-like
            Time points (hours). Must be 1D.
        logC : array-like
            Observed log10 concentrations at corresponding times. Must be 1D.

        Output (stored in self.params)
        ------------------------------
        {
            'Initial Value': log10(N0),
            'Lag'          : lag time (hours),
            'Maximum Rate' : mu_max (1/hour, natural-log base),
            'Final Value'  : log10(Nmax)
        }

        Notes
        -----
        This implementation is designed to be robust:
        - averages duplicate time points (replicates)
        - uses robust initial guesses
        - uses multi-start optimization
        - enforces log10Nmax >= log10N0 via delta parameterization
        - uses bounds tailored to hours to prevent non-physical runaway fits
        """

        # -------------------- sanitize inputs --------------------
        t = np.asarray(time, dtype=float)
        y = np.asarray(logC, dtype=float)

        if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
            raise ValueError("time and logC must be 1D arrays of the same length.")
        if np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
            raise ValueError("time and logC must not contain NaN/inf.")
        if np.any(np.diff(t) < 0):
            raise ValueError("time must be sorted in increasing order.")

        ln10 = np.log(10.0)

        # -------------------- handle duplicate time points --------------------
        # If you have replicates at the same time, average them to avoid dt ~ 0 issues.
        # This significantly reduces unstable slope estimates and optimizer failures.
        tu, inv = np.unique(t, return_inverse=True)
        if tu.size != t.size:
            y_sum = np.zeros_like(tu)
            y_cnt = np.zeros_like(tu)
            np.add.at(y_sum, inv, y)
            np.add.at(y_cnt, inv, 1.0)
            y = y_sum / y_cnt
            t = tu

        # -------------------- model (reparameterized) --------------------
        # Use delta = log10Nmax - log10N0, constrained > 0, so Nmax >= N0 always.
        def baranyi_model_reparam(t_, log10N0, delta, mu_max, h0):
            log10Nmax = log10N0 + delta

            y0 = ln10 * log10N0
            ymax = ln10 * log10Nmax

            # q0 = 1 / (exp(h0) - 1)
            # h0 close to 0 => exp(h0)-1 ~ 0 => q0 huge (can destabilize)
            q0 = 1.0 / (np.exp(h0) - 1.0)

            A = t_ + (1.0 / mu_max) * np.log(
                (np.exp(-mu_max * t_) + q0) / (1.0 + q0)
            )

            y_ = y0 + mu_max * A - np.log(
                1.0 + (np.exp(mu_max * A) - 1.0) / np.exp(ymax - y0)
            )

            return y_ / ln10

        # -------------------- robust initial guesses --------------------
        # Initial level guesses (robust-ish using min/max)
        log10N0_0 = float(np.min(y))
        log10Nmax_0 = float(np.max(y))
        delta_0 = max(1e-3, log10Nmax_0 - log10N0_0)

        # Robust slope-based mu guess (ignore dt==0 already handled by dedup)
        dy = np.diff(y)
        dt = np.diff(t)
        slopes = dy / np.maximum(dt, 1e-12)  # safe guard

        # Use a high quantile rather than max to avoid spikes from noisy points
        slope_q = float(np.quantile(slopes[np.isfinite(slopes)], 0.90)) if slopes.size else 0.0
        mu0 = max(1e-4, slope_q * ln10)  # 1/hour (base e)
        mu0 = min(mu0, 2.0)             # reasonable cap for hours (tune if needed)

        h0_0 = 1.0  # typical starting point; h0 = mu * lag

        # -------------------- bounds (hours) --------------------
        # These are broad but prevent runaway nonsense like Nmax=1311 or Lag=1e10 hours.
        #
        # Typical mu_max (1/h) often ~ 0.01 to 2, sometimes higher; we allow up to 5.
        # We keep mu_max >= 1e-4 to avoid huge (1/mu) blow-ups and astronomical lag.
        #
        # Lag = h0/mu. With mu >= 1e-4 and h0 <= 200 => lag <= 2,000,000 hours (still huge),
        # but in practice optimizer will stay in sensible region. You can tighten h0 if desired.
        #
        # delta limits keep Nmax from exploding.
        y_min = float(np.min(y))
        y_max = float(np.max(y))

        lower = (y_min - 5.0, 1e-3, 1e-4, 1e-3)     # (log10N0, delta, mu, h0)
        upper = (y_max + 5.0, 20.0, 5.0, 200.0)

        # -------------------- multi-start (robust fitting) --------------------
        # Try a small grid of initial (mu, h0) values around robust guesses.
        mu_candidates = [mu0, mu0/2, mu0*2, mu0/5, mu0*5]
        mu_candidates = [float(np.clip(m, lower[2], upper[2])) for m in mu_candidates]
        mu_candidates = sorted(set(mu_candidates))

        h0_candidates = [h0_0, 0.5, 2.0, 5.0, 10.0]
        h0_candidates = [float(np.clip(h, lower[3], upper[3])) for h in h0_candidates]
        h0_candidates = sorted(set(h0_candidates))

        best_popt = None
        best_sse = np.inf

        # Helper: skip initial guesses that produce non-finite model outputs
        def _finite_at_p0(p0):
            try:
                yp = baranyi_model_reparam(t, *p0)
                return np.all(np.isfinite(yp))
            except Exception:
                return False

        for mu_init in mu_candidates:
            for h0_init in h0_candidates:
                p0 = (log10N0_0, delta_0, mu_init, h0_init)

                if not _finite_at_p0(p0):
                    continue

                try:
                    popt, _ = curve_fit(
                        baranyi_model_reparam,
                        t, y,
                        p0=p0,
                        bounds=(lower, upper),
                        maxfev=30000
                    )

                    yhat = baranyi_model_reparam(t, *popt)
                    if not np.all(np.isfinite(yhat)):
                        continue

                    resid = y - yhat
                    sse = float(np.sum(resid * resid))
                    if sse < best_sse:
                        best_sse = sse
                        best_popt = popt

                except Exception:
                    # try the next initialization
                    continue

        if best_popt is None:
            raise RuntimeError("Baranyi fit failed: all multi-start initializations failed.")

        # -------------------- unpack + compute lag --------------------
        log10N0, delta, mu_max, h0 = best_popt
        log10Nmax = log10N0 + delta
        lag = h0 / mu_max  # hours

        self.params = {
            "Initial Value": float(log10N0),
            "Lag": float(lag),
            "Maximum Rate": float(mu_max),
            "Final Value": float(log10Nmax),
        }

    def _predict_baranyi(self, time: Union[float, List[float]]) -> Union[float, List[float]]:
        """
        Calcule logC(time) à partir des paramètres du modèle de Baranyi.
        """
        t = np.asarray(time, dtype=float)

        log10N0 = self.params['Initial Value']
        log10Nmax = self.params['Final Value']
        mu_max = self.params['Maximum Rate']
        lag = self.params['Lag']

        ln10 = np.log(10.0)

        y0 = ln10 * log10N0
        ymax = ln10 * log10Nmax

        h0 = mu_max * lag
        q0 = 1.0 / (np.exp(h0) - 1.0)

        A = t + (1.0 / mu_max) * np.log(
            (np.exp(-mu_max * t) + q0) / (1.0 + q0)
        )

        y = y0 + mu_max * A - np.log(
            1.0 + (np.exp(mu_max * A) - 1.0) / np.exp(ymax - y0)
        )

        return y / ln10

    def _fit_arrhenius(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_in_celsius: bool = True,
        R: float = 8.314462618,  # J/(mol*K)
    ) -> None:
        """
        Secondary model (Arrhenius):
            y(T) = A * exp(-Ea / (R*T))

        Paramètres
        ----------
        x : np.ndarray
            Température (°C par défaut, sinon K si x_in_celsius=False)
        y : np.ndarray
            Paramètre primaire (ex: mu_max). Doit être STRICTEMENT > 0.
        x_in_celsius : bool
            True -> x est en °C ; False -> x est en K
        R : float
            Constante des gaz parfaits (J/mol/K)

        Stocke dans self.params:
            {
              "A": ...,
              "Ea": ...,
              "R": ...,
              "x_in_celsius": ...,
              "model": "arrhenius"
            }
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
            raise ValueError("x et y doivent être 1D et de même longueur")

        if np.any(y <= 0):
            raise ValueError("Arrhenius requiert y > 0 (car on utilise ln(y)).")

        # Convertir en Kelvin
        T = x + 273.15 if x_in_celsius else x
        if np.any(T <= 0):
            raise ValueError("Température absolue (K) invalide (<= 0).")

        # Linéarisation : ln(y) = ln(A) - (Ea/R) * (1/T)
        X = 1.0 / T
        Y = np.log(y)

        slope, intercept = np.polyfit(X, Y, deg=1)
        # slope = -(Ea/R)
        Ea = -slope * R
        A = float(np.exp(intercept))

        self.params = {
            "A": A,
            "Ea": float(Ea),   # J/mol
            "R": float(R),
            "x_in_celsius": bool(x_in_celsius),
        }

    def _predict_arrhenius(
        self,
        x: Union[float, List[float]],
    ) -> Union[float, List[float]]:
        """
        Prédit y à partir de x (température), avec :
            y = A * exp(-Ea/(R*T))
        """
        if self.params is None:
            raise RuntimeError("Le modèle doit être entraîné avant predict.")

        A = self.params["A"]
        Ea = self.params["Ea"]
        R = self.params["R"]
        x_in_celsius = self.params.get("x_in_celsius", True)

        x_arr = np.asarray(x, dtype=float)
        T = x_arr + 273.15 if x_in_celsius else x_arr

        y_pred = A * np.exp(-Ea / (R * T))

        # Sortie scalaire si entrée scalaire, sinon liste
        if np.isscalar(x):
            return float(y_pred)
        return y_pred.tolist()



# class FullClassicalModel:
#     """
#     Docstring for FullClassicalModel
#     Creates a full classical model, combining primary and scondary models
#     FullClassicalModel model
#         - FullClassicalModel() : creates an instance of a classical model
#         - .fit() : "training" : identifies model parameters
#         - .predict() : predicts bacterial load LogC = f(time)
#         - .params : dict of the primary and secondary model type and parameters
#             params = {
#                 'primary': {'type': str
#                             'params': {
#                                 'parameter1': value.
#                                 'parameter2': value,
#                                 ...
#                                 }
#                             },
#                 'secondary': {'parameter': value}
#                 }
#     """

#     def __init__(self) -> None:
#         """
#         Docstring for __init__
#         ClassicalModel() : creates an instance of a classical model
#         """
#         self.params: dict | None = None


#     def fit(
#         self,
#         data: List[float],
#         logC: List[float],
#         model: str = "linear",
#     ) -> None:
#         """
#         Docstring for fit

#         :param self: Description
#         :param data: Description
#         :type data: List[float]
#         :param logC: Description
#         :type logC: List[float]
#         :param model: Description
#         :type model: str
#         """

#         pass

#     def predict(self, x: Union[float, List[float]]) -> Union[float, List[float]]:
#         """
#         Docstring for predict

#         """

#         pass
