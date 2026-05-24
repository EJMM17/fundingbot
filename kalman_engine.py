"""
kalman_engine.py — Filtro de Kalman para suavización y predicción del funding rate.

El funding rate observado en exchanges es ruidoso (microestructura, manipulación
breve, estimaciones imperfectas del exchange). Un filtro de Kalman separa la
señal (tendencia verdadera) del ruido de medición.

Usos:
  1. Suavizar la serie histórica de FR para análisis de persistencia.
  2. Predecir el próximo funding rate antes del snapshot.
  3. Detectar cambios de régimen cuando la innovación del filtro es anómala.

Modelo de espacio de estados:
  x_t = x_{t-1} + w_t   (estado: FR verdadero, random walk)
  z_t = x_t + v_t       (medición: FR observado)

Referencias:
    - Kalman (1960), "A New Approach to Linear Filtering and Prediction Problems"
    - Meinhold & Singpurwalla (1983), "Understanding the Kalman Filter"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("kalman")


class KalmanFundingFilter:
    """
    Filtro de Kalman univariante para series de funding rate.
    Estado: random walk con drift opcional.
    """

    def __init__(
        self,
        process_variance: float = 1e-6,
        measurement_variance: float = 1e-4,
        initial_estimate: float = 0.0,
        initial_error: float = 1.0,
    ) -> None:
        """
        Args:
            process_variance (Q): varianza del ruido de proceso.
                Valor pequeño = estado cambia lentamente.
            measurement_variance (R): varianza del ruido de medición.
                Valor grande = confía menos en las observaciones.
        """
        self.Q = process_variance
        self.R = measurement_variance
        self.x = initial_estimate  # Estimación del estado
        self.P = initial_error     # Error de estimación
        self._history: list[dict[str, float]] = []

    def update(self, measurement: float) -> dict[str, float]:
        """
        Actualiza el filtro con una nueva medición.
        Retorna dict con x (estado filtrado), P (error), K (ganancia de Kalman).
        """
        # Predicción
        x_pred = self.x
        P_pred = self.P + self.Q

        # Actualización
        K = P_pred / (P_pred + self.R)  # Ganancia de Kalman
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1.0 - K) * P_pred

        result = {
            "filtered": float(self.x),
            "prediction_error": float(measurement - x_pred),
            "gain": float(K),
            "error_covariance": float(self.P),
        }
        self._history.append(result)
        return result

    def predict(self, steps: int = 1) -> float:
        """
        Predice el estado a n pasos futuros.
        Con modelo random walk: predicción = estado actual.
        """
        return float(self.x)

    def batch_filter(self, measurements: np.ndarray) -> dict[str, np.ndarray]:
        """
        Aplica el filtro a una serie completa.
        Retorna arrays de filtered, predictions, gains.
        """
        measurements = np.asarray(measurements, dtype=float)
        n = len(measurements)
        filtered = np.zeros(n)
        pred_errors = np.zeros(n)
        gains = np.zeros(n)

        # Guardar estado original
        orig_x, orig_P = self.x, self.P

        for i, z in enumerate(measurements):
            res = self.update(z)
            filtered[i] = res["filtered"]
            pred_errors[i] = res["prediction_error"]
            gains[i] = res["gain"]

        # Restaurar estado original
        self.x, self.P = orig_x, orig_P
        self._history.clear()

        return {
            "filtered": filtered,
            "prediction_errors": pred_errors,
            "gains": gains,
        }

    @staticmethod
    def estimate_noise_parameters(measurements: np.ndarray) -> dict[str, float]:
        """
        Estima Q y R vía maximum likelihood simplificado (ratio de varianzas).
        R ≈ var(residuos de primer orden)
        Q ≈ var(diferencias de segundo orden) / 2
        """
        m = np.asarray(measurements, dtype=float)
        if len(m) < 5:
            return {"Q": 1e-6, "R": 1e-4}

        # Ruido de medición: varianza de primeras diferencias / 2
        diff1 = np.diff(m)
        R_est = float(np.var(diff1, ddof=1) / 2.0)

        # Ruido de proceso: varianza de segundas diferencias / 6
        diff2 = np.diff(m, n=2)
        Q_est = float(np.var(diff2, ddof=1) / 6.0)

        # Bounds para evitar inestabilidad
        R_est = max(1e-8, min(R_est, 1.0))
        Q_est = max(1e-10, min(Q_est, R_est * 0.1))

        return {"Q": Q_est, "R": R_est}


class AdaptiveKalmanFundingFilter:
    """
    Filtro de Kalman con R adaptativo.
    Cuando la innovación es anómala (outlier), aumenta R temporalmente
    para no confiar ciegamente en la observación.
    """

    def __init__(
        self,
        base_Q: float = 1e-6,
        base_R: float = 1e-4,
        outlier_threshold: float = 3.0,
        R_inflation: float = 10.0,
    ) -> None:
        self.base_Q = base_Q
        self.base_R = base_R
        self.outlier_threshold = outlier_threshold
        self.R_inflation = R_inflation
        self.x = 0.0
        self.P = 1.0
        self._history: list[dict[str, Any]] = []

    def update(self, measurement: float) -> dict[str, Any]:
        x_pred = self.x
        P_pred = self.P + self.base_Q

        # Innovación
        innovation = measurement - x_pred
        innovation_std = math.sqrt(P_pred + self.base_R)

        # Adaptación: si innovación es outlier, inflar R
        R_effective = self.base_R
        is_outlier = False
        if innovation_std > 1e-12 and abs(innovation / innovation_std) > self.outlier_threshold:
            R_effective = self.base_R * self.R_inflation
            is_outlier = True

        K = P_pred / (P_pred + R_effective)
        self.x = x_pred + K * innovation
        self.P = (1.0 - K) * P_pred

        result = {
            "filtered": float(self.x),
            "prediction_error": float(innovation),
            "gain": float(K),
            "error_covariance": float(self.P),
            "is_outlier": is_outlier,
            "R_effective": float(R_effective),
        }
        self._history.append(result)
        return result

    def predict(self, steps: int = 1) -> float:
        return float(self.x)


import math
