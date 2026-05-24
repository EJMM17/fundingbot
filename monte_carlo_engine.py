"""
monte_carlo_engine.py — Simulación Monte Carlo con colas pesadas para
optimización de sizing y stops.

En vez de asumir retornos normales (lo cual subestima drásticamente el
riesgo en crypto), usamos una mixture:
  - Cuerpo de la distribución: Normal(μ, σ) ajustada a retornos centrales
  - Colas: GPD(ξ, σ_gpd) ajustada a excesos sobre umbral (EVT)

Cada path simula el P&L de mantener una posición LONG por un horizonte
de hold (ej. 1 ciclo de funding + post-cobro). El resultado alimenta:
  - Probabilidad de ruina (P&L < -stop)
  - Kelly fraction modificado para colas pesadas (Half-Kelly con ajuste EVT)
  - Tamaño óptimo como % del capital dado el edge y la cola

Referencias:
    - Maier-Paape & Zhu (2018) — Kelly criterion for fat-tailed returns
    - Kallsen & Muhle-Karbe (2010) — On using shadow prices for frictional markets
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

import math_engine

logger = logging.getLogger("mc")


class MonteCarloRiskEngine:
    """
    Simulación Monte Carlo de P&L con colas pesadas (Normal+GPD mixture).
    """

    def __init__(self, n_paths: int = 10_000, random_seed: int | None = None) -> None:
        self.n_paths = n_paths
        if random_seed is not None:
            np.random.seed(random_seed)

    @staticmethod
    def _fit_mixture(log_returns: np.ndarray, threshold_pct: float = 0.90):
        """
        Ajusta mixture Normal+GPD a log-returns.
        Retorna dict con parámetros para simulación.
        """
        lr = np.asarray(log_returns, dtype=float)
        lr = lr[~np.isnan(lr)]
        n = len(lr)
        if n < 50:
            # Fallback a normal pura
            return {
                "type": "normal",
                "mu": float(np.mean(lr)),
                "sigma": float(np.std(lr, ddof=1)) or 1e-6,
                "xi": 0.0,
                "sigma_gpd": 0.0,
                "threshold": 0.0,
                "prob_tail": 0.0,
            }

        # Cuerpo: percentil 10%-90% ajustado a normal
        central = lr[(lr >= np.percentile(lr, 10)) & (lr <= np.percentile(lr, 90))]
        mu_body = float(np.mean(central))
        sigma_body = float(np.std(central, ddof=1)) or 1e-6

        # Colas: EVT sobre ambas colas (simétrico por simplicidad)
        # Usamos la cola negativa (pérdidas) como referencia
        losses = -lr[lr < 0]
        if len(losses) < 10:
            return {
                "type": "normal",
                "mu": float(np.mean(lr)),
                "sigma": float(np.std(lr, ddof=1)) or 1e-6,
                "xi": 0.0,
                "sigma_gpd": 0.0,
                "threshold": 0.0,
                "prob_tail": 0.0,
            }

        threshold = float(np.percentile(losses, threshold_pct * 100))
        excesses = losses[losses > threshold] - threshold
        n_excess = len(excesses)
        prob_tail = n_excess / n  # Probabilidad sobre TODOS los retornos, no solo negativos

        if n_excess < 5:
            xi = sigma_gpd = 0.0
        else:
            xi, sigma_gpd, _ = math_engine.ExtremeValueEngine.gpd_fit_mle(excesses)
            if np.isnan(xi) or np.isnan(sigma_gpd):
                xi = sigma_gpd = 0.0

        return {
            "type": "mixture",
            "mu": mu_body,
            "sigma": sigma_body,
            "xi": float(xi),
            "sigma_gpd": float(sigma_gpd),
            "threshold": threshold,
            "prob_tail": prob_tail,
            "n_excess": n_excess,
        }

    @staticmethod
    def _simulate_returns(params: dict, steps: int, n_paths: int) -> np.ndarray:
        """
        Genera matriz (n_paths, steps) de retornos log simulados.
        Mixture: con probabilidad (1-prob_tail) → normal, con prob_tail → GPD.
        """
        returns = np.random.normal(params["mu"], params["sigma"], size=(n_paths, steps))

        if params["type"] == "mixture" and params["prob_tail"] > 0 and params["xi"] != 0:
            # Máscara de cola
            tail_mask = np.random.random(size=(n_paths, steps)) < params["prob_tail"]
            n_tail = int(np.sum(tail_mask))
            if n_tail > 0:
                # Simular excesos GPD vía transformación inversa
                # CDF GPD: F(y) = 1 - (1 + ξ y/σ)^(-1/ξ)
                # Inversa: y = (σ/ξ) * [(1-U)^(-ξ) - 1]
                u = np.random.random(size=n_tail)
                xi = params["xi"]
                sigma_gpd = params["sigma_gpd"]
                if abs(xi) < 1e-8:
                    excesses = sigma_gpd * np.log(1.0 - u)
                else:
                    excesses = (sigma_gpd / xi) * ((1.0 - u) ** (-xi) - 1.0)

                # Cola bidireccional: 50% positiva, 50% negativa
                signs = np.random.choice([-1.0, 1.0], size=n_tail)
                returns[tail_mask] = signs * (params["threshold"] + excesses)

        return returns

    def run_simulation(
        self,
        log_returns: np.ndarray,
        entry_price: float,
        funding_rate: float,
        leverage: float = 5.0,
        hold_hours: float = 8.0,
        steps_per_hour: int = 4,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> dict[str, Any]:
        """
        Simula n_paths trayectorias de P&L para una posición LONG.

        Args:
            log_returns: histórico de retornos log
            entry_price: precio de entrada
            funding_rate: funding rate a cobrar (positivo = cobro)
            leverage: apalancamiento
            hold_hours: horas de hold simuladas
            steps_per_hour: granularidad (4 = cada 15 min)
            stop_loss_price: precio de stop (opcional)
            take_profit_price: precio de TP (opcional)

        Retorna dict con métricas de riesgo y sizing óptimo.
        """
        steps = int(hold_hours * steps_per_hour)
        if steps < 1:
            steps = 1

        params = self._fit_mixture(log_returns)
        sim_returns = self._simulate_returns(params, steps, self.n_paths)

        # Precios acumulados: P_t = P_0 * exp(Σ r_i)
        log_prices = np.cumsum(sim_returns, axis=1)
        prices = entry_price * np.exp(log_prices)

        # P&L por path (considerando funding cobrado al final)
        # P&L% = (P_final - P_entry) / P_entry * leverage + abs(funding_rate)
        # Nota: funding_rate negativo en exchange = cobro para longs
        price_changes = (prices[:, -1] - entry_price) / entry_price
        pnl_pct = price_changes * leverage + abs(funding_rate)

        # Aplicar stops si se definen
        if stop_loss_price is not None and stop_loss_price > 0:
            hit_sl = np.any(prices <= stop_loss_price, axis=1)
            sl_pct = (stop_loss_price - entry_price) / entry_price * leverage
            pnl_pct = np.where(hit_sl, sl_pct, pnl_pct)

        if take_profit_price is not None and take_profit_price > 0:
            hit_tp = np.any(prices >= take_profit_price, axis=1)
            tp_pct = (take_profit_price - entry_price) / entry_price * leverage
            pnl_pct = np.where(hit_tp & (~hit_sl if stop_loss_price else True), tp_pct, pnl_pct)

        # Métricas
        var_95 = float(np.percentile(pnl_pct, 5))
        var_99 = float(np.percentile(pnl_pct, 1))
        es_95 = float(np.mean(pnl_pct[pnl_pct <= var_95]))
        es_99 = float(np.mean(pnl_pct[pnl_pct <= var_99]))
        prob_ruin = float(np.mean(pnl_pct <= -0.20))  # Ruina = pérdida > 20% del margen
        prob_profit = float(np.mean(pnl_pct > 0))
        expected_pnl = float(np.mean(pnl_pct))
        worst_case = float(np.min(pnl_pct))

        # Kelly fraction modificado para colas pesadas
        # Kelly clásico: f* = μ/σ². Con colas pesadas, usamos half-Kelly con ajuste por ES.
        # f_adj = (μ - λ * ES) / (σ² + κ * ES²) donde λ y κ penalizan colas.
        mu_pnl = float(np.mean(pnl_pct))
        var_pnl = float(np.var(pnl_pct))
        if var_pnl > 1e-12:
            kelly_raw = mu_pnl / var_pnl
            # Penalización por colas pesadas: si ES es muy negativo, reducir Kelly
            tail_penalty = 1.0
            if es_95 < -0.05:
                tail_penalty = max(0.1, 1.0 + es_95 * 3.0)  # es_95 = -0.10 → penalty = 0.7
            if es_99 < -0.10:
                tail_penalty *= max(0.1, 1.0 + es_99 * 1.5)
            kelly_fraction = kelly_raw * tail_penalty * 0.5  # Half-Kelly conservador
        else:
            kelly_fraction = 0.0

        # Tamaño óptimo como % del capital
        # Limitar a [0, 0.30] (30% máximo del capital en una posición)
        optimal_size_pct = float(np.clip(kelly_fraction, 0.0, 0.30))

        result = {
            "n_paths": self.n_paths,
            "horizon_hours": hold_hours,
            "var_95": var_95,
            "var_99": var_99,
            "expected_shortfall_95": es_95,
            "expected_shortfall_99": es_99,
            "prob_ruin": prob_ruin,
            "prob_profit": prob_profit,
            "expected_pnl": expected_pnl,
            "worst_case": worst_case,
            "kelly_fraction": kelly_fraction,
            "optimal_size_pct": optimal_size_pct,
            "gpd_xi": params["xi"],
            "gpd_sigma": params["sigma_gpd"],
            "powerlaw_alpha": None,
        }

        # Ajustar powerlaw alpha si hay datos suficientes
        if len(log_returns) >= 50:
            try:
                tail_info = math_engine.PowerLawTail.tail_risk_index(log_returns)
                result["powerlaw_alpha"] = tail_info.get("alpha")
            except Exception:
                pass

        return result

    def recommend_position_size(
        self,
        log_returns: np.ndarray,
        capital: float,
        funding_rate: float,
        leverage: float = 5.0,
    ) -> dict[str, Any]:
        """
        Retorna tamaño recomendado en USDT basado en simulación MC.
        """
        sim = self.run_simulation(
            log_returns=log_returns,
            entry_price=1.0,  # irrelevante para sizing relativo
            funding_rate=funding_rate,
            leverage=leverage,
            hold_hours=8.0,
        )
        optimal_fraction = sim["optimal_size_pct"]
        recommended_usdt = capital * optimal_fraction

        return {
            **sim,
            "capital": capital,
            "recommended_usdt": recommended_usdt,
            "max_acceptable_usdt": capital * 0.30,  # hard cap 30%
        }
