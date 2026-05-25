"""
microstructure_engine.py — Métricas de microestructura de mercado.

Análisis de order book, bid-ask dynamics, y proxies de toxicity flow
usando solo datos de ticker (bid, ask, volume) — sin necesidad de
order book completo (L2) que consume muchos rate-limits.

Métricas:
  - Bid-Ask Imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
  - VPIN proxy: Volume-synchronized probability of informed trading
  - Effective Spread: 2 * |last - mid| / mid
  - Price Impact proxy: Δprice / √volume

Referencias:
    - Easley, López de Prado, O'Hara (2012) — VPIN
    - Kyle (1985) — Price impact λ
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

import config

logger = logging.getLogger("micro")


class MicrostructureAnalyzer:
    """
    Analizador de microestructura basado en datos de ticker.
    """

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self._volume_buckets: deque[float] = deque(maxlen=window)
        self._buy_volume_buckets: deque[float] = deque(maxlen=window)
        self._sell_volume_buckets: deque[float] = deque(maxlen=window)
        self._last_price: float = 0.0
        self._last_returns: deque[float] = deque(maxlen=window)

    def feed_ticker(self, ticker: dict) -> dict[str, Any]:
        """
        Procesa un ticker y retorna métricas de microestructura.

        ticker debe tener (de ccxt fetch_ticker):
            bid, ask, last, quoteVolume, baseVolume, vwap
        """
        bid = ticker.get("bid") or 0.0
        ask = ticker.get("ask") or 0.0
        last = ticker.get("last") or ticker.get("close") or 0.0
        quote_vol = ticker.get("quoteVolume") or 0.0
        base_vol = ticker.get("baseVolume") or 0.0
        vwap = ticker.get("vwap") or 0.0

        if bid <= config.FLOAT_EPSILON or ask <= config.FLOAT_EPSILON or last <= config.FLOAT_EPSILON:
            return {"valid": False}

        mid = (bid + ask) / 2.0
        spread = (ask - bid) / mid
        effective_spread = 2.0 * abs(last - mid) / mid

        # Price impact proxy (Kyle's lambda simplificado)
        # λ ≈ |Δprice| / √volume
        price_impact = 0.0
        if self._last_price > config.FLOAT_EPSILON and base_vol > config.FLOAT_EPSILON:
            ret = abs(last - self._last_price) / self._last_price
            price_impact = ret / math.sqrt(base_vol + 1.0)
            self._last_returns.append(ret)
        self._last_price = last

        # Bid-Ask Imbalance (proxy de dirección del flow)
        # Como no tenemos L2, usamos vwap vs mid como proxy
        imbalance = 0.0
        if vwap > config.FLOAT_EPSILON and mid > config.FLOAT_EPSILON:
            imbalance = (vwap - mid) / mid

        # Bucket volume para VPIN
        self._volume_buckets.append(base_vol)

        # Classify volume como buy/sell usando tick rule
        # Si precio subió → más buy volume, si bajó → más sell volume
        if len(self._last_returns) >= 2:
            recent_ret = self._last_returns[-1] if self._last_returns else 0.0
            if recent_ret > 0:
                buy_vol = base_vol * 0.6
                sell_vol = base_vol * 0.4
            elif recent_ret < 0:
                buy_vol = base_vol * 0.4
                sell_vol = base_vol * 0.6
            else:
                buy_vol = sell_vol = base_vol * 0.5
        else:
            buy_vol = sell_vol = base_vol * 0.5

        self._buy_volume_buckets.append(buy_vol)
        self._sell_volume_buckets.append(sell_vol)

        # VPIN proxy
        vpin = 0.0
        if len(self._volume_buckets) >= 10:
            total_vol = sum(self._volume_buckets)
            if total_vol > config.FLOAT_EPSILON:
                vpin = sum(
                    abs(b - s) for b, s in zip(self._buy_volume_buckets, self._sell_volume_buckets)
                ) / total_vol

        return {
            "valid": True,
            "mid": mid,
            "spread": spread,
            "effective_spread": effective_spread,
            "imbalance": imbalance,
            "price_impact": price_impact,
            "vpin_proxy": vpin,
        }

    def toxicity_signal(self) -> dict[str, Any]:
        """
        Señal de toxicidad basada en histórico acumulado.
        """
        if len(self._volume_buckets) < 10:
            return {"toxic": False, "level": 0.0, "reason": "insufficient_data"}

        recent_vpin = sum(
            abs(b - s) for b, s in zip(
                list(self._buy_volume_buckets)[-10:],
                list(self._sell_volume_buckets)[-10:]
            )
        ) / (sum(list(self._volume_buckets)[-10:]) + 1e-12)

        # VPIN > 0.6 → alto flujo informado, mercado tóxico
        toxic = recent_vpin > 0.6
        level = min(1.0, recent_vpin)

        return {
            "toxic": toxic,
            "level": float(level),
            "vpin": float(recent_vpin),
            "reason": "high_vpin" if toxic else "normal",
        }


import math
