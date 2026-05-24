"""
math_engine.py — Arsenal Estadístico-Matemático de Nivel Quant.

Módulo puro, sin side-effects. Cada función está documentada con la referencia
al método/paper correspondiente.

Referencias principales:
    - Clauset, Shalizi, Newman (2009), SIAM Review 51(4):661-703 — Power Law MLE
    - McNeil, Frey, Embrechts (2015) — EVT / GPD / POT
    - ECB Working Paper 3166 (2025) — Score-driven EVT for crypto
    - Cornell (2022) — Hurst Trading strategy
    - MDPI (2024/2026) — Hurst in pairs trading / Entropy in green finance
    - Samara AM (2025) — Navigating chaos with Hurst in crypto
    - Schreiber (2000), Phys. Rev. Lett. 85(2):461 — Transfer Entropy
    - Andersen et al. (2001), Econometrica — Realized Variance
    - Parkinson (1980) — High-low volatility estimator
    - Garman & Klass (1980) — O-H-L-C volatility estimator
    - Kantelhardt et al. (2002), Physica A — MF-DFA
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from scipy import optimize, special, stats

logger = logging.getLogger("math")

# ──────────────────────────────────────────────────────────────────────────────
# A. Leyes de Potencia y Colas Pesadas — Clauset-Shalizi-Newman (2009)
# ──────────────────────────────────────────────────────────────────────────────


class PowerLawTail:
    """
    Implementación del método Clauset-Shalizi-Newman para ajustar leyes de
    potencia a datos empíricos vía MLE, con estimación de x_min por KS y
    goodness-of-fit test sintético.

    Fórmula MLE continua: α̂ = 1 + n / Σ ln(x_i / x_min)
    """

    @staticmethod
    def _power_law_mle_continuous(x: np.ndarray, x_min: float) -> float:
        """MLE de α para datos continuos (x_i ≥ x_min)."""
        data = x[x >= x_min]
        n = len(data)
        if n < 3:
            return float("nan")
        alpha = 1.0 + n / np.sum(np.log(data / x_min))
        # Corrección de sesgo de finita muestra (Clauset et al. nota)
        # Para n pequeño, α suele estar ligeramente sesgado hacia abajo.
        # No aplicamos corrección explícita para mantener consistencia con CSM.
        return alpha

    @staticmethod
    def _power_law_mle_discrete(x: np.ndarray, x_min: float) -> float:
        """MLE de α para datos discretos vía maximización numérica con zeta."""
        data = x[x >= x_min]
        n = len(data)
        if n < 3:
            return float("nan")

        def neg_log_likelihood(a: float) -> float:
            if a <= 1.0:
                return 1e18
            # log-likelihood discreta: -n ln ζ(α, x_min) - α Σ ln(x_i)
            # scipy.special.zeta(α, q) = Σ (k+q)^(-α) para k=0..∞
            z = special.zeta(a, x_min)
            if z <= 0 or np.isinf(z) or np.isnan(z):
                return 1e18
            return float(n * np.log(z) + a * np.sum(np.log(data)))

        # Búsqueda acotada entre 1.01 y 10.0
        result = optimize.minimize_scalar(
            neg_log_likelihood, bounds=(1.01, 10.0), method="bounded"
        )
        return float(result.x) if result.success else float("nan")

    @staticmethod
    def _cdf_powerlaw(x: np.ndarray, alpha: float, x_min: float) -> np.ndarray:
        """CDF teórica de power law continua: 1 - (x/x_min)^(-(α-1))."""
        return 1.0 - (x / x_min) ** (-(alpha - 1.0))

    @staticmethod
    def _ks_statistic(data: np.ndarray, alpha: float, x_min: float) -> float:
        """Kolmogorov-Smirnov statistic entre datos empíricos y power law."""
        filtered = np.sort(data[data >= x_min])
        n = len(filtered)
        if n == 0:
            return float("inf")
        empirical_cdf = np.arange(1, n + 1) / n
        theoretical_cdf = PowerLawTail._cdf_powerlaw(filtered, alpha, x_min)
        return float(np.max(np.abs(empirical_cdf - theoretical_cdf)))

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        discrete: bool = False,
        x_min_candidates: int | None = None,
    ) -> dict[str, float]:
        """
        Ajusta ley de potencia completa:
          1. Grid search sobre candidatos de x_min.
          2. Para cada x_min, MLE de α.
          3. Seleccionar x_min que minimice KS.
        Retorna dict con alpha, x_min, ks_statistic.
        """
        x = np.asarray(x, dtype=float)
        x = x[x > 0]
        if len(x) < 10:
            return {"alpha": float("nan"), "x_min": float("nan"), "ks": float("nan")}

        # Candidatos de x_min: percentiles 5% a 95% de los datos
        if x_min_candidates is None:
            x_min_candidates = min(50, max(10, len(x) // 5))
        candidates = np.percentile(x, np.linspace(5, 95, x_min_candidates))
        candidates = np.unique(candidates)

        best_ks = float("inf")
        best_alpha = float("nan")
        best_x_min = float("nan")

        for xm in candidates:
            data_above = x[x >= xm]
            if len(data_above) < 5:
                continue
            if discrete:
                alpha = cls._power_law_mle_discrete(x, xm)
            else:
                alpha = cls._power_law_mle_continuous(x, xm)
            if np.isnan(alpha) or alpha <= 1.0:
                continue
            ks = cls._ks_statistic(x, alpha, xm)
            if ks < best_ks:
                best_ks = ks
                best_alpha = alpha
                best_x_min = xm

        return {
            "alpha": best_alpha,
            "x_min": best_x_min,
            "ks": best_ks,
        }

    @classmethod
    def goodness_of_fit(
        cls,
        x: np.ndarray,
        x_min: float,
        alpha: float,
        n_synth: int = 500,
        discrete: bool = False,
    ) -> float:
        """
        Test de bondad de ajuste sintético (Clauset et al. 2009).
        Genera n_synth muestras de power law con mismos parámetros y longitud.
        p-value = fracción de KS_synth > KS_empirical.
        Si p > 0.1, no se rechaza hipótesis de power law.
        """
        x = np.asarray(x, dtype=float)
        empirical_ks = cls._ks_statistic(x, alpha, x_min)
        n_above = int(np.sum(x >= x_min))
        if n_above < 5:
            return 0.0

        count_better = 0
        for _ in range(n_synth):
            # Muestreo por transformación inversa de CDF
            u = np.random.uniform(0, 1, n_above)
            synth = x_min * (1.0 - u) ** (-1.0 / (alpha - 1.0))
            # Ajustar la muestra sintética
            fit_synth = cls.fit(synth, discrete=discrete, x_min_candidates=20)
            if np.isnan(fit_synth["alpha"]):
                continue
            ks_synth = cls._ks_statistic(synth, fit_synth["alpha"], fit_synth["x_min"])
            if ks_synth > empirical_ks:
                count_better += 1

        return count_better / n_synth

    @classmethod
    def tail_risk_index(cls, returns: np.ndarray) -> dict[str, float]:
        """
        Índice compuesto de riesgo de cola basado en ley de potencia.
        Ajusta power law a retornos negativos (valor absoluto).

        α < 2.0  → media infinita      → risk = 1.0
        2 ≤ α < 3 → varianza infinita  → risk = 0.8–1.0
        3 ≤ α < 4 → colas pesadas      → risk = 0.0–0.3
        α ≥ 4     → colas ligeras      → risk = 0.0
        """
        returns = np.asarray(returns, dtype=float)
        if len(returns) < 20:
            return {"alpha": float("nan"), "pvalue": 0.0, "risk": 0.5, "hill_xi": float("nan")}

        neg_returns = np.abs(returns[returns < 0])
        if len(neg_returns) < 10:
            return {"alpha": float("nan"), "pvalue": 0.0, "risk": 0.5, "hill_xi": float("nan")}

        fit = cls.fit(neg_returns, discrete=False)
        alpha = fit["alpha"]
        x_min = fit["x_min"]

        if np.isnan(alpha):
            return {"alpha": float("nan"), "pvalue": 0.0, "risk": 0.5, "hill_xi": float("nan")}

        pvalue = cls.goodness_of_fit(neg_returns, x_min, alpha, n_synth=200)

        # Estimador de Hill como verificación cruzada
        sorted_neg = np.sort(neg_returns)[::-1]
        k = max(5, int(0.1 * len(sorted_neg)))
        hill_xi = 1.0 / cls._hill_estimator(sorted_neg, k)

        if alpha < 2.0:
            risk = 1.0
        elif alpha < 3.0:
            risk = 0.2 + (3.0 - alpha) * 0.8
        elif alpha < 4.0:
            risk = (4.0 - alpha) * 0.3
        else:
            risk = 0.0

        # Si pvalue < 0.1, power law es dudosa → usar percentil empírico como fallback
        if pvalue < 0.1:
            # Cola pesada empírica: percentil 95 de pérdidas
            empirical_tail = np.percentile(np.abs(returns), 95)
            risk = max(risk, min(1.0, empirical_tail / 0.05))

        return {"alpha": alpha, "pvalue": pvalue, "risk": risk, "hill_xi": hill_xi}

    @staticmethod
    def _hill_estimator(sorted_descending: np.ndarray, k: int) -> float:
        """Estimador de Hill para el índice de cola α."""
        if k < 2 or len(sorted_descending) < k + 1:
            return float("nan")
        top_k = sorted_descending[:k]
        x_k = sorted_descending[k]
        if x_k <= 0:
            return float("nan")
        alpha = 1.0 / (np.mean(np.log(top_k / x_k)))
        return alpha


# ──────────────────────────────────────────────────────────────────────────────
# B. EVT — Extreme Value Theory (POT + GPD)
# ──────────────────────────────────────────────────────────────────────────────


class ExtremeValueEngine:
    """
    EVT para stops dinámicos y medidas de riesgo de cola.
    Teorema de Pickands-Balkema-de Haan: excesos sobre umbral → GPD.
    """

    @staticmethod
    def gpd_fit_mle(excesses: np.ndarray) -> tuple[float, float, float]:
        """
        Ajusta GPD vía MLE. Retorna (shape ξ, scale σ, log_likelihood).
        Log-likelihood GPD:
            L = -n ln σ - (1 + 1/ξ) Σ ln(1 + ξ x_i/σ)    para ξ ≠ 0
            L = -n ln σ - (1/σ) Σ x_i                    para ξ = 0 (exponencial)
        """
        excesses = np.asarray(excesses, dtype=float)
        excesses = excesses[excesses >= 0]
        n = len(excesses)
        if n < 5:
            return float("nan"), float("nan"), float("nan")

        # Momentos iniciales para σ y ξ
        m1 = np.mean(excesses)
        m2 = np.var(excesses, ddof=0)
        if m2 == 0:
            return 0.0, m1, float("nan")

        # Estimador inicial por método de momentos
        xi0 = 0.5 * (m2 / m1**2 - 1.0)
        sigma0 = m1 * (1.0 + xi0)

        def neg_ll(params: np.ndarray) -> float:
            xi, sigma = params
            if sigma <= 1e-12:
                return 1e18
            # Condición: 1 + ξ x_i/σ > 0 para todos los datos
            z = 1.0 + xi * excesses / sigma
            if np.any(z <= 0):
                return 1e18
            if abs(xi) < 1e-8:
                # Límite exponencial
                return float(n * np.log(sigma) + np.sum(excesses) / sigma)
            return float(n * np.log(sigma) + (1.0 + 1.0 / xi) * np.sum(np.log(z)))

        bounds = [(-0.5, 1.5), (1e-6, max(10.0 * m1, 1.0))]
        result = optimize.minimize(
            neg_ll,
            x0=[max(-0.4, min(1.0, xi0)), max(1e-6, sigma0)],
            bounds=bounds,
            method="L-BFGS-B",
        )

        if result.success:
            xi, sigma = result.x
            ll = -neg_ll(result.x)
            return float(xi), float(sigma), float(ll)
        return float("nan"), float("nan"), float("nan")

    @staticmethod
    def evt_var_es(
        returns: np.ndarray,
        confidence: float = 0.99,
        threshold_pct: float = 0.90,
    ) -> tuple[float, float, float, float]:
        """
        Peaks Over Threshold (POT) con GPD.
        Retorna (VaR, ES, xi, sigma).

        VaR_p = u + (σ/ξ) * [((N/N_u)*(1-p))^(-ξ) - 1]
        ES_p  = (VaR_p + σ - ξ*u) / (1 - ξ)     para ξ < 1
        """
        returns = np.asarray(returns, dtype=float)
        if len(returns) < 30:
            # Fallback a percentil empírico
            var = float(np.percentile(returns, confidence * 100))
            es = float(np.percentile(returns, min(99.9, confidence * 100 + 0.5)))
            return var, es, float("nan"), float("nan")

        u = float(np.percentile(returns, threshold_pct * 100))
        excesses = returns[returns > u] - u
        n_u = len(excesses)
        n = len(returns)

        if n_u < 5:
            var = float(np.percentile(returns, confidence * 100))
            es = float(np.percentile(returns, min(99.9, confidence * 100 + 0.5)))
            return var, es, float("nan"), float("nan")

        xi, sigma, _ = ExtremeValueEngine.gpd_fit_mle(excesses)
        if np.isnan(xi) or np.isnan(sigma):
            var = float(np.percentile(returns, confidence * 100))
            es = float(np.percentile(returns, min(99.9, confidence * 100 + 0.5)))
            return var, es, float("nan"), float("nan")

        # VaR bajo GPD
        prob = (n / n_u) * (1.0 - confidence)
        if prob <= 0:
            prob = 1e-12

        if abs(xi) < 1e-8:
            var = u + sigma * (-np.log(prob))
        else:
            var = u + (sigma / xi) * (prob ** (-xi) - 1.0)

        # ES bajo GPD
        if xi >= 1.0:
            es = var  # ES infinito para ξ ≥ 1
        elif abs(xi) < 1e-8:
            es = var + sigma
        else:
            es = (var + sigma - xi * u) / (1.0 - xi)

        return float(var), float(es), float(xi), float(sigma)

    @staticmethod
    def dynamic_stop_loss_logspace(
        entry_price: float,
        log_returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """
        Stop-loss dinámico en espacio logarítmico.
        Calcula VaR EVT sobre log-returns y aplica:
            stop_price = entry_price * exp(VaR_log)
        """
        if len(log_returns) < 20 or entry_price <= 0:
            # Fallback: stop del 5% en espacio log
            return entry_price * math.exp(-0.05)

        # Usar EVT sobre log-returns negativos (pérdidas)
        losses = -log_returns[log_returns < 0]
        if len(losses) < 10:
            return entry_price * math.exp(-0.05)

        try:
            var_95, _, xi, sigma = ExtremeValueEngine.evt_var_es(
                losses, confidence=confidence, threshold_pct=0.85
            )
            # var_95 es positivo (magnitud de pérdida)
            stop_log = -var_95
            stop_price = entry_price * math.exp(stop_log)
            # Sanity check: no permitir stops > 50% ni < 0.5%
            min_stop = entry_price * 0.5
            max_stop = entry_price * 0.995
            return float(np.clip(stop_price, min_stop, max_stop))
        except Exception:
            return entry_price * math.exp(-0.05)

    @staticmethod
    def evt_crowding_detection(oi_changes: np.ndarray, current_change: float) -> bool:
        """
        Detecta crowding anómalo vía POT: si current_change excede VaR_95
        de la distribución histórica de cambios de OI.
        """
        if len(oi_changes) < 20:
            return False
        try:
            var_95, _, _, _ = ExtremeValueEngine.evt_var_es(
                oi_changes, confidence=0.95, threshold_pct=0.90
            )
            return current_change > var_95
        except Exception:
            return False


# ──────────────────────────────────────────────────────────────────────────────
# C. Hurst y Análisis de Persistencia (R/S + DFA)
# ──────────────────────────────────────────────────────────────────────────────


class PersistenceAnalyzer:
    """
    Hurst exponent vía Rescaled Range (R/S) y Detrended Fluctuation Analysis (DFA).
    DFA es más robusto que R/S para series con tendencias locales.
    """

    @staticmethod
    def hurst_rs(ts: np.ndarray, max_lag: int = 100) -> float:
        """
        Rescaled Range analysis clásica.
        Regresión log-log de R/S vs lag → pendiente = H.
        """
        ts = np.asarray(ts, dtype=float)
        ts = ts[~np.isnan(ts)]
        n = len(ts)
        if n < max_lag * 2:
            max_lag = max(10, n // 4)
        if n < 20:
            return 0.5

        lags = range(10, max_lag + 1, max(1, (max_lag - 10) // 20))
        rs_values = []
        lag_values = []

        for lag in lags:
            chunks = n // lag
            if chunks < 2:
                continue
            rs_chunk = []
            for i in range(chunks):
                chunk = ts[i * lag : (i + 1) * lag]
                mean_c = np.mean(chunk)
                # Serie acumulada desviada de la media
                y = np.cumsum(chunk - mean_c)
                r = np.max(y) - np.min(y)
                s = np.std(chunk, ddof=0)
                if s > 1e-12:
                    rs_chunk.append(r / s)
            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
                lag_values.append(lag)

        if len(lag_values) < 3:
            return 0.5

        log_lags = np.log(lag_values)
        log_rs = np.log(rs_values)
        H, _, _, _, _ = stats.linregress(log_lags, log_rs)
        return float(np.clip(H, 0.0, 1.0))

    @staticmethod
    def hurst_dfa(
        ts: np.ndarray,
        min_scale: int = 4,
        max_scale: int | None = None,
        n_scales: int = 20,
        order: int = 1,
    ) -> float:
        """
        Detrended Fluctuation Analysis (DFA).
        Pendiente de log F(s) vs log s = H.
        """
        ts = np.asarray(ts, dtype=float)
        ts = ts[~np.isnan(ts)]
        n = len(ts)
        if n < min_scale * 4:
            return 0.5

        # Integrar serie
        mean_ts = np.mean(ts)
        y = np.cumsum(ts - mean_ts)

        if max_scale is None:
            max_scale = n // 4
        scales = np.unique(
            np.logspace(
                np.log10(min_scale), np.log10(max_scale), n_scales
            ).astype(int)
        )
        scales = scales[scales >= min_scale]

        fluctuations = []
        valid_scales = []

        for s in scales:
            n_segments = n // s
            if n_segments < 2:
                continue
            f2 = []
            for v in range(n_segments):
                segment = y[v * s : (v + 1) * s]
                x = np.arange(s)
                # Ajuste polinomial de orden `order`
                coeffs = np.polyfit(x, segment, order)
                trend = np.polyval(coeffs, x)
                f2.append(np.mean((segment - trend) ** 2))
            # Considerar también los segmentos overlap (opcional, aquí non-overlap)
            fluctuations.append(np.sqrt(np.mean(f2)))
            valid_scales.append(s)

        if len(valid_scales) < 3:
            return 0.5

        log_s = np.log(valid_scales)
        log_f = np.log(fluctuations)
        H, _, _, _, _ = stats.linregress(log_s, log_f)
        return float(np.clip(H, 0.0, 1.0))

    @classmethod
    def fr_regime_signal(cls, fr_history: np.ndarray, use_dfa: bool = True) -> dict[str, Any]:
        """
        Clasifica régimen del funding rate según Hurst.
        Retorna dict con H, regime, mean_fr.
        """
        fr = np.asarray(fr_history, dtype=float)
        if len(fr) < 20:
            return {"H": 0.5, "regime": "random_walk", "mean_fr": 0.0}

        if use_dfa:
            H = cls.hurst_dfa(fr)
        else:
            H = cls.hurst_rs(fr)

        mean_fr = float(np.mean(fr))

        if H > 0.58:
            regime = "persistent_negative" if mean_fr < 0 else "persistent_positive"
        elif H < 0.42:
            regime = "mean_reverting"
        else:
            regime = "random_walk"

        return {"H": H, "regime": regime, "mean_fr": mean_fr}


# ──────────────────────────────────────────────────────────────────────────────
# D. Entropía, Información y Predictibilidad
# ──────────────────────────────────────────────────────────────────────────────


class InformationMetrics:
    """
    Shannon entropy, Mutual Information, Transfer Entropy, Conditional Entropy.
    """

    @staticmethod
    def shannon_entropy_normalized(
        series: np.ndarray, bins: str = "fd"
    ) -> float:
        """
        Entropía de Shannon normalizada a [0, 1].
        H_norm = H_empirical / H_max
        bins: 'fd' (Freedman-Diaconis), 'scott', 'sturges', o entero.
        """
        series = np.asarray(series, dtype=float)
        series = series[~np.isnan(series)]
        if len(series) < 5:
            return 1.0  # Máxima incertidumbre por defecto

        if isinstance(bins, str):
            if bins == "fd":
                # Freedman-Diaconis rule: h = 2*IQR / n^(1/3)
                iqr = np.percentile(series, 75) - np.percentile(series, 25)
                if iqr > 0 and len(series) > 0:
                    h = 2.0 * iqr / (len(series) ** (1.0 / 3.0))
                    n_bins = max(3, int((np.max(series) - np.min(series)) / h))
                else:
                    n_bins = max(3, int(np.sqrt(len(series))))
            elif bins == "scott":
                h = 3.5 * np.std(series, ddof=1) / (len(series) ** (1.0 / 3.0))
                n_bins = max(3, int((np.max(series) - np.min(series)) / h)) if h > 0 else 10
            elif bins == "sturges":
                n_bins = max(3, int(np.ceil(np.log2(len(series)) + 1)))
            else:
                n_bins = 10
        else:
            n_bins = int(bins)

        n_bins = min(max(n_bins, 3), 50)  # bounds
        counts, _ = np.histogram(series, bins=n_bins)
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]

        H = -np.sum(probs * np.log(probs))
        H_max = np.log(n_bins)
        if H_max <= 0:
            return 1.0
        return float(np.clip(H / H_max, 0.0, 1.0))

    @staticmethod
    def mutual_information(
        x: np.ndarray, y: np.ndarray, n_bins: int = 20
    ) -> float:
        """Mutual Information estimado vía histograma 2D."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) < 10:
            return 0.0

        # Joint histogram
        joint, _, _ = np.histogram2d(x, y, bins=n_bins)
        joint /= np.sum(joint)

        # Marginals
        px = np.sum(joint, axis=1)
        py = np.sum(joint, axis=0)

        # MI
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))
        return float(mi)

    @staticmethod
    def transfer_entropy_knn(
        x: np.ndarray,
        y: np.ndarray,
        lag: int = 1,
        k: int = 4,
    ) -> float:
        """
        Transfer Entropy estimado vía k-NN simplificado (Kraskov-Stögbauer-Grassberger).
        TE_{Y→X} = I(X_t ; Y_{t-lag} | X_{t-1})

        Implementación simplificada basada en bins discretos (más robusta
        para series cortas que k-NN puro).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        if n < lag + 10 or len(y) < lag + 10:
            return 0.0

        # Alinear series
        x_t = x[lag:]
        x_tm1 = x[:-lag]
        y_tmlag = y[:-lag]

        # Discretizar a bins para estimar entropías
        n_bins = min(15, max(5, int(n ** 0.4)))

        def discretize(z):
            z_min, z_max = np.min(z), np.max(z)
            if z_max == z_min:
                return np.zeros_like(z, dtype=int)
            return np.clip(
                ((z - z_min) / (z_max - z_min) * n_bins).astype(int),
                0, n_bins - 1,
            )

        dx_t = discretize(x_t)
        dx_tm1 = discretize(x_tm1)
        dy_tmlag = discretize(y_tmlag)

        # TE ≈ H(X_t | X_{t-1}) - H(X_t | X_{t-1}, Y_{t-lag})
        # = H(X_t, X_{t-1}) + H(X_{t-1}, Y_{t-lag}) - H(X_{t-1}) - H(X_t, X_{t-1}, Y_{t-lag})
        def joint_entropy(*arrays):
            combos = np.stack(arrays, axis=1)
            unique, counts = np.unique(combos, axis=0, return_counts=True)
            probs = counts / np.sum(counts)
            return -np.sum(probs * np.log(probs + 1e-12))

        h_xt_xtm1 = joint_entropy(dx_t, dx_tm1)
        h_xtm1_y = joint_entropy(dx_tm1, dy_tmlag)
        h_xtm1 = joint_entropy(dx_tm1)
        h_xt_xtm1_y = joint_entropy(dx_t, dx_tm1, dy_tmlag)

        te = h_xt_xtm1 + h_xtm1_y - h_xtm1 - h_xt_xtm1_y
        return float(max(0.0, te))

    @staticmethod
    def conditional_entropy_fr_oi(
        fr_changes: np.ndarray, oi_changes: np.ndarray, n_bins: int = 15
    ) -> float:
        """H(FR|OI) = H(FR,OI) - H(OI)."""
        fr = np.asarray(fr_changes, dtype=float)
        oi = np.asarray(oi_changes, dtype=float)
        mask = ~(np.isnan(fr) | np.isnan(oi))
        fr, oi = fr[mask], oi[mask]
        if len(fr) < 10:
            return float("nan")

        def discretize(z, nb):
            z_min, z_max = np.min(z), np.max(z)
            if z_max == z_min:
                return np.zeros_like(z, dtype=int)
            return np.clip(((z - z_min) / (z_max - z_min) * nb).astype(int), 0, nb - 1)

        dfr = discretize(fr, n_bins)
        doi = discretize(oi, n_bins)

        def entropy_1d(arr):
            _, counts = np.unique(arr, return_counts=True)
            probs = counts / np.sum(counts)
            return -np.sum(probs * np.log(probs + 1e-12))

        def joint_entropy_2d(a, b):
            joint, _, _ = np.histogram2d(a, b, bins=(n_bins, n_bins))
            joint /= np.sum(joint)
            probs = joint[joint > 0]
            return -np.sum(probs * np.log(probs + 1e-12))

        h_oi = entropy_1d(doi)
        h_fr_oi = joint_entropy_2d(dfr, doi)
        return float(max(0.0, h_fr_oi - h_oi))


# ──────────────────────────────────────────────────────────────────────────────
# E. Volatilidad Logarítmica y Realized Measures
# ──────────────────────────────────────────────────────────────────────────────


class LogVolatilityEngine:
    """
    Volatilidad en espacio logarítmico. RV, Parkinson, Garman-Klass, log-ATR.
    """

    @staticmethod
    def realized_variance(log_returns: np.ndarray) -> float:
        """RV = Σ r_t²."""
        lr = np.asarray(log_returns, dtype=float)
        lr = lr[~np.isnan(lr)]
        if len(lr) < 2:
            return 0.0
        return float(np.sum(lr**2))

    @staticmethod
    def realized_volatility_annualized(
        log_returns: np.ndarray, periods_per_year: float = 365 * 24
    ) -> float:
        """σ_anual = sqrt(RV_total / N * periods_per_year)."""
        rv = LogVolatilityEngine.realized_variance(log_returns)
        n = len(log_returns)
        if n < 2:
            return 0.0
        return float(np.sqrt(rv / n * periods_per_year))

    @staticmethod
    def parkinson_volatility(high: np.ndarray, low: np.ndarray) -> float:
        """
        σ²_p = (1/(4N ln 2)) Σ (ln(H_i/L_i))²
        Eficiencia relativa ≈ 5.2x vs close-close.
        """
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        mask = (h > 0) & (l > 0) & (h >= l)
        h, l = h[mask], l[mask]
        n = len(h)
        if n < 2:
            return 0.0
        return float(np.sqrt(np.sum(np.log(h / l) ** 2) / (4.0 * n * np.log(2.0))))

    @staticmethod
    def garman_klass_volatility(
        open_p: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> float:
        """
        σ²_gk = 0.5(ln(H/L))² - (2ln2 - 1)(ln(C/O))²
        Eficiencia relativa ≈ 7.4x.
        """
        o = np.asarray(open_p, dtype=float)
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)
        mask = (o > 0) & (h > 0) & (l > 0) & (c > 0)
        o, h, l, c = o[mask], h[mask], l[mask], c[mask]
        n = len(o)
        if n < 2:
            return 0.0
        term1 = 0.5 * np.sum(np.log(h / l) ** 2)
        term2 = (2.0 * np.log(2.0) - 1.0) * np.sum(np.log(c / o) ** 2)
        return float(np.sqrt((term1 - term2) / n))

    @staticmethod
    def log_atr(prices: np.ndarray, window: int = 14) -> float:
        """
        ATR en espacio logarítmico.
        ATR_log = EMA(|ln(C_t) - ln(C_{t-1})|, window)
        """
        p = np.asarray(prices, dtype=float)
        p = p[p > 0]
        if len(p) < 2:
            return 0.0
        log_diffs = np.abs(np.diff(np.log(p)))
        if len(log_diffs) < window:
            return float(np.mean(log_diffs))
        # EMA simple
        alpha = 2.0 / (window + 1.0)
        ema = log_diffs[0]
        for val in log_diffs[1:]:
            ema = alpha * val + (1.0 - alpha) * ema
        return float(ema)

    @staticmethod
    def volatility_regime(
        log_returns: np.ndarray,
        window: int = 30,
        extreme_pct: float = 0.90,
        calm_pct: float = 0.25,
    ) -> tuple[str, float, float]:
        """
        Clasifica régimen de volatilidad basado en RV móvil vs histórica.
        Retorna (regime, rv_current, rv_percentile).
        """
        lr = np.asarray(log_returns, dtype=float)
        lr = lr[~np.isnan(lr)]
        if len(lr) < window + 5:
            return "normal", 0.0, 0.5

        # RV móvil en ventanas
        rv_windows = np.array([
            np.sum(lr[i : i + window] ** 2)
            for i in range(len(lr) - window + 1)
        ])
        rv_current = rv_windows[-1]
        rv_all = rv_windows[:-1]  # histórico excluyendo actual
        if len(rv_all) < 5:
            return "normal", float(rv_current), 0.5

        percentile = float(stats.percentileofscore(rv_all, rv_current, kind="rank") / 100.0)

        if percentile >= extreme_pct:
            regime = "extreme"
        elif percentile <= calm_pct:
            regime = "calm"
        else:
            regime = "normal"

        return regime, float(rv_current), percentile


# ──────────────────────────────────────────────────────────────────────────────
# F. Multifractalidad — MF-DFA
# ──────────────────────────────────────────────────────────────────────────────


class MultifractalSpectrum:
    """
    Multifractal Detrended Fluctuation Analysis (MF-DFA).
    """

    @staticmethod
    def mf_dfa(
        ts: np.ndarray,
        q_values: np.ndarray | None = None,
        scales: np.ndarray | None = None,
        order: int = 1,
    ) -> dict[str, np.ndarray]:
        """
        MF-DFA estándar.
        Para cada q y escala s: F_q(s) = (1/N_s Σ F²(s,ν)^(q/2))^(1/q)
        Regresión log-log: F_q(s) ~ s^{h(q)}.
        """
        ts = np.asarray(ts, dtype=float)
        ts = ts[~np.isnan(ts)]
        n = len(ts)
        if n < 64:
            return {"h_q": np.array([0.5]), "q": np.array([0.0])}

        if q_values is None:
            q_values = np.concatenate([
                np.linspace(-5, -0.5, 10),
                np.linspace(0.5, 5, 10),
            ])
        q_values = q_values[q_values != 0]  # q=0 requiere tratamiento especial

        if scales is None:
            scales = np.unique(np.logspace(np.log10(4), np.log10(n // 4), 20).astype(int))
            scales = scales[scales >= 4]

        y = np.cumsum(ts - np.mean(ts))
        h_q = np.zeros(len(q_values))

        for qi, q in enumerate(q_values):
            f_q = []
            valid_scales = []
            for s in scales:
                n_segments = n // s
                if n_segments < 2:
                    continue
                f2 = []
                for v in range(n_segments):
                    segment = y[v * s : (v + 1) * s]
                    x = np.arange(s)
                    coeffs = np.polyfit(x, segment, order)
                    trend = np.polyval(coeffs, x)
                    f2.append(np.mean((segment - trend) ** 2))
                if q > 0:
                    f_q_val = (np.mean(np.array(f2) ** (q / 2.0))) ** (1.0 / q)
                else:
                    # q < 0: usa mínimo
                    f_q_val = np.exp(0.5 * np.mean(np.log(np.array(f2) + 1e-12)))
                f_q.append(f_q_val)
                valid_scales.append(s)

            if len(valid_scales) >= 3:
                slope, _, _, _, _ = stats.linregress(
                    np.log(valid_scales), np.log(f_q)
                )
                h_q[qi] = slope
            else:
                h_q[qi] = 0.5

        return {"h_q": h_q, "q": q_values}

    @staticmethod
    def multifractal_width(h_q: np.ndarray, q_values: np.ndarray) -> float:
        """
        Ancho del espectro multifractal Δα = max(α) - min(α).
        τ(q) = q*h(q) - 1
        α = dτ/dq ≈ derivada numérica de τ(q)
        """
        if len(h_q) < 4:
            return 0.0
        tau = q_values * h_q - 1.0
        # Derivada numérica centrada
        alpha = np.gradient(tau, q_values)
        delta_alpha = float(np.max(alpha) - np.min(alpha))
        return max(0.0, delta_alpha)

    @staticmethod
    def width_penalty(delta_alpha: float, threshold: float = 0.50) -> float:
        """Penalización para sizing basada en ancho multifractal."""
        if delta_alpha <= threshold:
            return 0.0
        return float(min(1.0, (delta_alpha - threshold) / threshold))


# ──────────────────────────────────────────────────────────────────────────────
# G. Score Integrador Matemático
# ──────────────────────────────────────────────────────────────────────────────


def power_score(
    funding_rate: float,
    next_funding_rate: float,
    fr_history: np.ndarray,
    log_returns: np.ndarray,
    oi_changes: np.ndarray,
    price_history: np.ndarray,
    volume_24h: float,
    interval_hours: float = 8.0,
    min_volume: float = 70_000_000.0,
) -> dict[str, float]:
    """
    Score integrado 0.0–1.0 basado en arsenal matemático completo.
    Retorna dict con score final y todas las métricas intermedias.
    """
    result: dict[str, float] = {}

    # 1. Edge base por funding rate
    edge = abs(funding_rate) / 0.003
    edge_score = min(edge, 1.0)
    result["edge_score"] = edge_score

    # 2. Persistencia (Hurst sobre FR)
    hurst = 0.5
    hurst_multiplier = 1.0
    if len(fr_history) >= 20:
        hurst = PersistenceAnalyzer.hurst_dfa(fr_history)
        result["hurst"] = hurst
        if funding_rate < 0:
            if hurst > 0.58:
                hurst_multiplier = 1.0 + (hurst - 0.58) * 1.5
            elif hurst < 0.42:
                hurst_multiplier = max(0.5, 1.0 - (0.42 - hurst))
        result["hurst_multiplier"] = hurst_multiplier
    else:
        result["hurst"] = 0.5
        result["hurst_multiplier"] = 1.0

    # 3. Tail risk
    tail_penalty = 0.0
    alpha_val = float("nan")
    if len(log_returns) >= 20:
        tail_info = PowerLawTail.tail_risk_index(log_returns)
        alpha_val = tail_info["alpha"]
        tail_penalty = tail_info["risk"] ** 2
        result["tail_alpha"] = alpha_val
        result["tail_risk"] = tail_info["risk"]
    result["tail_penalty"] = tail_penalty

    # 4. Volatilidad log
    vol_penalty = 0.0
    vol_bonus = 0.0
    if len(log_returns) >= 30:
        regime, rv, pct = LogVolatilityEngine.volatility_regime(log_returns)
        result["vol_regime"] = {"calm": 0, "normal": 1, "extreme": 2}.get(regime, 1)
        result["vol_percentile"] = pct
        if regime == "extreme":
            vol_penalty = 0.5 + 0.5 * max(0.0, (pct - 0.9) / 0.1)
        elif regime == "calm":
            vol_bonus = 0.1
    result["vol_penalty"] = vol_penalty
    result["vol_bonus"] = vol_bonus

    # 5. Entropía del funding
    entropy_penalty = 0.0
    entropy_bonus = 0.0
    if len(fr_history) >= 10:
        h_norm = InformationMetrics.shannon_entropy_normalized(fr_history, bins="fd")
        result["entropy_fr"] = h_norm
        if h_norm > 0.8:
            entropy_penalty = (h_norm - 0.8) * 2.5
        elif h_norm < 0.4:
            entropy_bonus = 0.15
    result["entropy_penalty"] = entropy_penalty
    result["entropy_bonus"] = entropy_bonus

    # 6. Transfer Entropy OI → FR
    te_weight = 0.5
    if len(oi_changes) >= 20 and len(fr_history) >= 20:
        # Alinear longitudes
        min_len = min(len(oi_changes), len(fr_history))
        oi_aligned = np.asarray(oi_changes)[-min_len:]
        fr_aligned = np.asarray(fr_history)[-min_len:]
        try:
            te = InformationMetrics.transfer_entropy_knn(oi_aligned, fr_aligned, lag=1)
            result["te_oi_to_fr"] = te
            te_weight = 0.5 + 0.5 * min(1.0, te * 5.0)
        except Exception:
            te_weight = 0.5
    result["te_weight"] = te_weight

    # 7. Multifractal
    mf_penalty = 0.0
    if len(log_returns) >= 64:
        try:
            mf = MultifractalSpectrum.mf_dfa(log_returns)
            delta_alpha = MultifractalSpectrum.multifractal_width(mf["h_q"], mf["q"])
            result["mf_delta_alpha"] = delta_alpha
            mf_penalty = MultifractalSpectrum.width_penalty(delta_alpha, threshold=0.50)
        except Exception:
            result["mf_delta_alpha"] = 0.0
    result["mf_penalty"] = mf_penalty

    # 8. Liquidez (volumen)
    liq_score = min(volume_24h / (min_volume * 5.0), 1.0)
    result["liq_score"] = liq_score

    # 9. Frecuencia
    freq_boost = min(2.0, 8.0 / interval_hours) if interval_hours > 0 else 1.0
    result["freq_boost"] = freq_boost

    # Score final
    score = (
        edge_score * hurst_multiplier * freq_boost
        - tail_penalty
        - vol_penalty
        + vol_bonus
        - entropy_penalty
        + entropy_bonus
        - mf_penalty
    ) * te_weight

    # Boost por liquidez
    score = score * (0.7 + 0.3 * liq_score)

    # Penalizar si next funding no es negativo (tesis débil)
    if next_funding_rate > funding_rate * 0.5:
        score *= 0.7

    result["score"] = float(np.clip(score, 0.0, 1.0))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# H. Helpers de conveniencia
# ──────────────────────────────────────────────────────────────────────────────


def log_return(prices: np.ndarray) -> np.ndarray:
    """Retornos logarítmicos: r_t = ln(P_t / P_{t-1})."""
    p = np.asarray(prices, dtype=float)
    p = p[p > 0]
    if len(p) < 2:
        return np.array([])
    return np.diff(np.log(p))
