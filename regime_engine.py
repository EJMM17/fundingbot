"""
regime_engine.py — Modelo de 2 regímenes para el funding rate.

Detecta si el funding rate está en:
  Regimen 0 ("calm"):  volatilidad baja, estructura estable, predecible.
  Regimen 1 ("chaos"): volatilidad alta, estructura rota, impredecible.

Implementación: EM simplificado sobre una mixture de 2 normales con
volatilidades distintas, aplicado a los cambios del funding rate.

Uso:
  - Si régimen = "chaos": reducir sizing agresivamente, aumentar stops.
  - Si régimen = "calm": operar con tamaño normal.

Referencias:
    - Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series"
    - Kritzman et al. (2012), "Regime Shifts: Implications for Dynamic Strategies"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("regime")


class FundingRegimeModel:
    """
    Modelo de 2 regímenes para cambios del funding rate.
    """

    def __init__(self) -> None:
        self.regime_labels = ["calm", "chaos"]

    @staticmethod
    def fit_em(
        changes: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> dict[str, Any]:
        """
        EM algorithm simplificado para mixture de 2 normales.
        Retorna parámetros y probabilidades de régimen.
        """
        x = np.asarray(changes, dtype=float)
        x = x[~np.isnan(x)]
        n = len(x)
        if n < 20:
            return {
                "mu_calm": 0.0, "sigma_calm": 1e-3,
                "mu_chaos": 0.0, "sigma_chaos": 1e-2,
                "prob_calm": 0.5, "prob_chaos": 0.5,
                "current_regime": "calm",
            }

        # Inicialización por percentiles de volatilidad
        sigma_total = np.std(x, ddof=1)
        mu_calm, mu_chaos = 0.0, 0.0
        sigma_calm = sigma_total * 0.3
        sigma_chaos = sigma_total * 2.0
        pi_calm = 0.7  # Probabilidad a priori de calm

        for _ in range(max_iter):
            # E-step: calcular responsabilidades
            pdf_calm = pi_calm * _norm_pdf(x, mu_calm, sigma_calm)
            pdf_chaos = (1.0 - pi_calm) * _norm_pdf(x, mu_chaos, sigma_chaos)
            gamma_calm = pdf_calm / (pdf_calm + pdf_chaos + 1e-12)

            # M-step: actualizar parámetros
            n_calm = np.sum(gamma_calm)
            n_chaos = n - n_calm

            if n_calm > 0:
                mu_calm_new = np.sum(gamma_calm * x) / n_calm
                sigma_calm_new = math.sqrt(np.sum(gamma_calm * (x - mu_calm_new) ** 2) / n_calm)
            else:
                mu_calm_new, sigma_calm_new = mu_calm, sigma_calm

            if n_chaos > 0:
                gamma_chaos = 1.0 - gamma_calm
                mu_chaos_new = np.sum(gamma_chaos * x) / n_chaos
                sigma_chaos_new = math.sqrt(np.sum(gamma_chaos * (x - mu_chaos_new) ** 2) / n_chaos)
            else:
                mu_chaos_new, sigma_chaos_new = mu_chaos, sigma_chaos

            pi_calm_new = n_calm / n

            # Convergencia
            delta = (
                abs(mu_calm_new - mu_calm)
                + abs(sigma_calm_new - sigma_calm)
                + abs(mu_chaos_new - mu_chaos)
                + abs(sigma_chaos_new - sigma_chaos)
                + abs(pi_calm_new - pi_calm)
            )

            mu_calm, sigma_calm = mu_calm_new, max(1e-6, sigma_calm_new)
            mu_chaos, sigma_chaos = mu_chaos_new, max(1e-6, sigma_chaos_new)
            pi_calm = np.clip(pi_calm_new, 0.1, 0.9)

            if delta < tol:
                break

        # Régimen actual = última observación
        last_pdf_calm = pi_calm * _norm_pdf(x[-1], mu_calm, sigma_calm)
        last_pdf_chaos = (1.0 - pi_calm) * _norm_pdf(x[-1], mu_chaos, sigma_chaos)
        current_regime = "calm" if last_pdf_calm > last_pdf_chaos else "chaos"

        return {
            "mu_calm": float(mu_calm),
            "sigma_calm": float(sigma_calm),
            "mu_chaos": float(mu_chaos),
            "sigma_chaos": float(sigma_chaos),
            "prob_calm": float(pi_calm),
            "prob_chaos": float(1.0 - pi_calm),
            "current_regime": current_regime,
            "log_likelihood": float(np.sum(np.log(last_pdf_calm + last_pdf_chaos + 1e-12))),
        }

    @staticmethod
    def classify(
        fr_history: np.ndarray,
        hurst: float | None = None,
        vol_regime: str = "normal",
    ) -> dict[str, Any]:
        """
        Clasifica el régimen actual del funding usando EM + métricas externas.
        """
        fr = np.asarray(fr_history, dtype=float)
        if len(fr) < 10:
            return {"regime": "unknown", "confidence": 0.0, "details": {}}

        changes = np.diff(fr)
        em_result = FundingRegimeModel.fit_em(changes)

        # Refinar con Hurst y volatilidad externa
        regime = em_result["current_regime"]
        confidence = max(em_result["prob_calm"], em_result["prob_chaos"])

        if hurst is not None:
            if hurst > 0.6 and regime == "calm":
                confidence = min(1.0, confidence + 0.15)
            elif hurst < 0.4 and regime == "chaos":
                confidence = min(1.0, confidence + 0.15)
            elif hurst > 0.6 and regime == "chaos":
                # Contradicción: alta persistencia pero EM dice caos
                # Puede ser caos persistente (muy malo) → mantener caos pero bajar confianza
                confidence = max(0.0, confidence - 0.20)

        if vol_regime == "extreme":
            regime = "chaos"
            confidence = min(1.0, confidence + 0.20)
        elif vol_regime == "calm" and regime == "chaos":
            confidence = max(0.0, confidence - 0.15)

        return {
            "regime": regime,
            "confidence": float(confidence),
            "details": em_result,
        }


def _norm_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """PDF de normal."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


import math
