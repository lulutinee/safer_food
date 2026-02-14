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
            predicts y : bacterial load(s) at x : time(s)
        Secondary models :
            predicts primary model parameter y as a function of variable x
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



class FullClassicalModel:
    """
    Docstring for FullClassicalModel
    Creates a full classical model, combining pri
    """
