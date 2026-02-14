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
        time: List[float],
        logC: List[float],
        model: str = "linear",
    ) -> None:
        """
        fits a model logC = f(time)

        Parameters
        ----------
        time : list[float] : time in hours
        logC : list[float] : bacterial concentration in log10 cfu /g or /ml
        model : str, default "linear"
            'linear'    : linear model
            'baranyi'   : full Baranyi model, Baranyi and Roberts (1994)
        If you want to predict bacterial growth : bacterial load at t=0
        should be log t0 = 0 (1 cfu)
        """
        x = np.asarray(time, dtype=float)
        y = np.asarray(logC, dtype=float)

        if x.shape != y.shape:
            raise ValueError("time and logC should have the same length")

        self.model = model

        if model == "linear":
            self.model = model
            self._fit_linear(x, y)

        elif model == 'baranyi':
            self.model = model
            self._fit_baranyi(x, y)

        else:
            raise NotImplementedError(
                f"Model '{model}' not implemented"
            )


    def predict(self, time: Union[float, List[float]]) -> Union[float, List[float]]:
        """
        Docstring for predict
        predicts bacterial load(s) at time(s) time
        The model instance has to be fitted (.fit method) before prediction

        time: growth time (in hours), can be a single value (float) or a list of time points

        returns : bacterial load, for every time point.
        Returns a float if time = float, and a list if time = list
        """
        if self.params is None or self.model is None:
            raise RuntimeError("Model has to be trained with fit() before predict().")

        if self.model == "linear":
            return self._predict_linear(time)

        elif self.model == 'baranyi':
            return self._predict_baranyi(time)

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
        Ajuste le modèle de Baranyi (isotherme, 1994).

        Entrées
        -------
        time : liste ou array des temps
        logC : liste ou array des concentrations (log10)

        Sortie
        ------
        dict :
        {
            'Initial Value' : log10(N0),
            'Lag'           : lambda (même unité que time),
            'Maximum Rate'  : mu_max (1/unité de temps, base e),
            'Final Value'   : log10(Nmax)
        }
        """
        t = np.asarray(time, dtype=float)
        y = np.asarray(logC, dtype=float)

        if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
            raise ValueError("time et logC doivent être 1D et de même longueur")
        if np.any(np.diff(t) < 0):
            raise ValueError("time doit être croissant")

        ln10 = np.log(10.0)

        # -------- modèle interne --------
        def baranyi_model(t, log10N0, log10Nmax, mu_max, h0):
            y0 = ln10 * log10N0
            ymax = ln10 * log10Nmax

            q0 = 1.0 / (np.exp(h0) - 1.0)

            A = t + (1.0 / mu_max) * np.log(
                (np.exp(-mu_max * t) + q0) / (1.0 + q0)
            )

            y = y0 + mu_max * A - np.log(
                1.0 + (np.exp(mu_max * A) - 1.0) / np.exp(ymax - y0)
            )

            return y / ln10

        # -------- valeurs initiales --------
        log10N0_0 = float(np.min(y))
        log10Nmax_0 = float(np.max(y))

        dy = np.diff(y)
        dt = np.diff(t)
        slope0 = np.max(dy / np.maximum(dt, 1e-12))
        mu0 = max(1e-6, slope0 * ln10)

        h0_0 = 1.0

        p0 = (log10N0_0, log10Nmax_0, mu0, h0_0)

        bounds = (
            (-np.inf, -np.inf, 1e-9, 1e-9),
            ( np.inf,  np.inf, np.inf, np.inf)
        )

        popt, _ = curve_fit(
            baranyi_model, t, y, p0=p0, bounds=bounds, maxfev=20000
        )

        log10N0, log10Nmax, mu_max, h0 = popt

        # Lag (lambda) : h0 = mu_max * lambda
        lag = h0 / mu_max

        self.params = {
            'Initial Value': float(log10N0),
            'Lag': float(lag),
            'Maximum Rate': float(mu_max),
            'Final Value': float(log10Nmax)
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
