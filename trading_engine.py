"""
trading_engine.py — Motor de ejecución v4 (KuCoin Futures).

Diferencias clave vs v3:
──────────────────────────────────────────────────────────────────
1. Bugs corregidos:
   - Price drift gate: ahora compara precio de scan vs ejecución.
   - Partial fill en entrada: margen real proporcional al filled.
   - Funding state reconciliado al reiniciar (inferido desde exchange).
   - Slippage gate: alerta si fill price se desvía > umbral.

2. Seguridad de capital:
   - Global margin cap + max open positions.
   - Daily drawdown circuit breaker (P&L, no solo errores técnicos).
   - Liquidity gate: spread bid-ask máximo permitido.

3. Estrategia mejorada:
   - Opportunity ranking: solo entra en las TOP_N mejores.
   - Predicted funding sanity check (nextFundingRate).
   - Adaptive position sizing por calidad del edge.
   - Post-funding hold extension si el siguiente FR sigue negativo.

4. Operativa:
   - Dry-run / paper trading mode.
   - Webhook notifications para trades y alertas críticas.
   - Graceful shutdown configurable (hold vs close).

Regla de oro: el bot existe para COBRAR el funding fee y SALIR.
Cualquier segundo post-cobro es exposición sin tesis.

Todas las operaciones son LONG-only en KuCoin USD-M Swap.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import ccxt.async_support as ccxt_async

import auto_learning
import config
import db_local
import kalman_engine
import math_engine
import microstructure_engine
import monte_carlo_engine
import notifications
import regime_engine

logger = logging.getLogger("engine")


# ── Tipos auxiliares ──────────────────────────────────────────────

class OISnapshot(NamedTuple):
    timestamp: float   # time.time()
    value: float       # OI en USDT


@dataclass
class FundingTiming:
    """Datos temporales del funding extraídos de fetch_funding_rate."""
    funding_rate: float = 0.0
    next_funding_rate: float = 0.0       # predicted para el siguiente ciclo
    next_funding_ts_ms: int = 0          # ms — próximo snapshot
    last_funding_ts_ms: int = 0          # ms — último snapshot
    minutes_to_next: float = float("inf")
    minutes_since_last: float = float("inf")
    interval_hours: float = 8.0          # ciclo del par (8h, 4h, 1h)
    # Campos adicionales de KuCoin
    funding_rate_cap: float = 0.003      # límite superior del FR
    funding_rate_floor: float = -0.003   # límite inferior del FR
    scan_price: float = 0.0              # precio al momento del scan

    @property
    def entry_window_minutes(self) -> float:
        """Ventana de entrada en minutos, escalada al ciclo del par."""
        raw = self.interval_hours * 60 * config.ENTRY_WINDOW_FRACTION
        return max(
            config.ENTRY_WINDOW_MINUTES_MIN,
            min(config.ENTRY_WINDOW_MINUTES_MAX, raw),
        )

    @property
    def blindfold_minutes(self) -> float:
        """Ventana ciega post-snapshot, escalada al ciclo."""
        raw = self.interval_hours * 60 * config.POST_FUNDING_BLINDFOLD_FRACTION
        return max(
            config.POST_FUNDING_BLINDFOLD_MINUTES_MIN,
            min(config.POST_FUNDING_BLINDFOLD_MINUTES_MAX, raw),
        )

    @property
    def net_fr_after_fees(self) -> float:
        """FR neto después de fees de cierre (0.06% taker).
        Positivo = cobro neto real al cerrar post-snapshot."""
        return abs(self.funding_rate) - 0.0006  # 0.06% taker close

    @property
    def annualized_fr_pct(self) -> float:
        """FR anualizado para comparar oportunidades entre ciclos."""
        if self.interval_hours <= 0:
            return 0.0
        cycles_per_year = (365 * 24) / self.interval_hours
        return self.funding_rate * cycles_per_year * 100


@dataclass
class CoinState:
    """Estado en memoria por moneda monitoreada."""
    last_entry_price: float = 0.0
    total_margin_used: float = 0.0
    last_order_ts: float = 0.0
    oi_history: deque = field(default_factory=lambda: deque(maxlen=120))
    funding_prev: float | None = None

    # Tracking de snapshot
    funding_collected: bool = False
    funding_collected_ts: float = 0.0
    last_known_next_funding_ms: int = 0

    # Price drift gate
    scan_price: float = 0.0

    # ── Históricos para análisis matemático ─────────────────────────
    fr_history: deque = field(default_factory=lambda: deque(maxlen=config.MATH_HISTORY_WINDOW))
    log_return_history: deque = field(default_factory=lambda: deque(maxlen=config.MATH_HISTORY_WINDOW))
    oi_change_history: deque = field(default_factory=lambda: deque(maxlen=config.MATH_HISTORY_WINDOW))
    price_history: deque = field(default_factory=lambda: deque(maxlen=config.MATH_HISTORY_WINDOW))

    # Métricas matemáticas cacheadas (actualizadas cada ciclo)
    last_tail_alpha: float | None = None
    last_tail_pvalue: float | None = None
    last_tail_risk: float | None = None
    last_hurst_fr: float | None = None
    last_fr_regime: str = "unknown"
    last_entropy_fr: float | None = None
    last_te_oi_to_fr: float | None = None
    last_te_fr_to_oi: float | None = None
    last_vol_regime: str = "unknown"
    last_rv_annualized: float | None = None
    last_evt_stop_price: float = 0.0
    last_evt_xi: float | None = None
    last_evt_sigma: float | None = None
    last_multifractal_width: float | None = None
    last_power_score: float = 0.0

    # Nuevos motores avanzados
    kalman_filter: kalman_engine.KalmanFundingFilter | None = None
    last_kalman_filtered_fr: float | None = None
    last_kalman_predicted_fr: float | None = None
    last_regime_confidence: float = 0.0
    last_micro_imbalance: float = 0.0
    last_micro_vpin: float = 0.0
    last_mc_optimal_size: float = 0.0
    last_mc_prob_ruin: float = 0.0
    last_learning_multiplier: float = 1.0
    micro_analyzer: microstructure_engine.MicrostructureAnalyzer | None = None


# ── Retry con backoff exponencial ─────────────────────────────────

async def _retry(coro_factory, description: str = "API call"):
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            return await coro_factory()
        except (
            ccxt_async.RateLimitExceeded,
            ccxt_async.RequestTimeout,
            ccxt_async.NetworkError,
        ) as exc:
            wait = config.BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning(
                "%s intento %d/%d falló (%s). Esperando %.1fs",
                description, attempt, config.MAX_RETRIES, exc, wait,
            )
            if attempt == config.MAX_RETRIES:
                raise
            await asyncio.sleep(wait)


# ── Motor principal ───────────────────────────────────────────────

class TradingEngine:
    def __init__(self) -> None:
        self.exchange: ccxt_async.kucoinfutures | None = None
        self.states: dict[str, CoinState] = defaultdict(CoinState)
        self._running: bool = False
        self._configured_symbols: set[str] = set()
        self._consecutive_errors: int = 0
        self._circuit_open: bool = False
        self._circuit_open_ts: float = 0.0

        # Riesgo y capital
        self._start_balance: float = 0.0
        self._peak_balance: float = 0.0
        self._last_heartbeat_ts: float = 0.0
        self._daily_drawdown_triggered: bool = False
        self._daily_drawdown_ts: float = 0.0
        # Diagnóstico acumulado entre heartbeats
        self._scan_cycles: int = 0
        self._best_fr_seen: float = 0.0      # FR más negativo visto desde el último heartbeat
        self._best_fr_symbol: str = ""
        self.learning: auto_learning.AutoLearningEngine | None = (
            auto_learning.AutoLearningEngine(db_local.DB_PATH)
            if config.ENABLE_AUTO_LEARNING
            else None
        )

    # ── Inicialización ────────────────────────────────────────────

    async def start(self) -> None:
        """Crea la conexión asíncrona a KuCoin Futures y reconcilia estado."""
        self.exchange = ccxt_async.kucoinfutures({
            "apiKey": config.KUCOIN_API_KEY,
            "secret": config.KUCOIN_SECRET,
            "password": config.KUCOIN_PASSPHRASE,   # Obligatorio en KuCoin
            "enableRateLimit": True,
        })
        await _retry(lambda: self.exchange.load_markets(), "load_markets")
        swap_count = sum(
            1 for m in self.exchange.markets.values()
            if m.get("swap") and m.get("linear")
        )
        logger.info("KuCoin Futures conectado. %d mercados swap lineales.", swap_count)

        # Reconciliar estado desde posiciones reales antes de operar
        await self._reconcile_state()

        # Capturar balance inicial para drawdown
        if not config.DRY_RUN:
            try:
                balance = await _retry(lambda: self.exchange.fetch_balance(), "fetch_balance")
                await asyncio.sleep(config.RATE_LIMIT_PAUSE)
                total = float(
                    balance.get("total", {}).get("USDT")
                    or balance.get("total", {}).get("USD")
                    or 0
                )
                self._start_balance = total
                self._peak_balance = total
                logger.info("BALANCE INICIAL: %.2f USDT", total)
            except Exception as exc:
                logger.error("No se pudo leer balance inicial: %s", exc)

        await self._refresh_learning(force=True)
        self._running = True

    async def stop(self) -> None:
        self._running = False
        if self.exchange:
            await self.exchange.close()
            logger.info("Conexión a KuCoin Futures cerrada.")

    # ── Reconciliación de estado (crítico para reinicios) ─────────

    async def _reconcile_state(self) -> None:
        """
        Al arrancar, reconstruye total_margin_used desde las posiciones
        reales del exchange.

        Sin esto, un reinicio con posiciones abiertas hace que el bot
        ignore el MAX_MARGIN_PER_COIN porque cree que está flat.
        """
        try:
            positions_raw = await _retry(
                lambda: self.exchange.fetch_positions(),
                "reconcile_fetch_positions",
            )
            await asyncio.sleep(config.RATE_LIMIT_PAUSE)

            reconciled = 0
            for pos in positions_raw:
                sym = pos.get("symbol")
                side = pos.get("side")
                contracts = abs(float(pos.get("contracts") or 0))
                initial_margin = float(pos.get("initialMargin") or 0)
                entry_price = float(pos.get("entryPrice") or 0)

                if sym and side == "long" and contracts > 0:
                    state = self.states[sym]
                    state.total_margin_used = initial_margin
                    state.last_entry_price = entry_price
                    reconciled += 1
                    logger.info(
                        "RECONCILE %s | Margen: %.2f USDT | Entry: %.4f",
                        sym, initial_margin, entry_price,
                    )

                    # Inferir funding_collected desde el timing del exchange
                    try:
                        fr_data = await _retry(
                            lambda s=sym: self.exchange.fetch_funding_rate(s),
                            f"reconcile_fr({sym})",
                        )
                        ft = self._parse_funding_timing(fr_data, sym)
                        await asyncio.sleep(config.RATE_LIMIT_PAUSE)

                        # Si el último snapshot fue hace poco y todavía estamos
                        # dentro del blindfold, asumimos que cobramos
                        if ft.minutes_since_last <= ft.blindfold_minutes:
                            state.funding_collected = True
                            state.funding_collected_ts = time.time() - (ft.minutes_since_last * 60)
                            state.last_known_next_funding_ms = ft.next_funding_ts_ms
                            logger.info(
                                "RECONCILE %s | Funding cobrado inferido (%.1f min post-snapshot).",
                                sym, ft.minutes_since_last,
                            )
                        else:
                            state.last_known_next_funding_ms = ft.next_funding_ts_ms
                    except Exception as exc:
                        logger.debug("Reconcile FR skip %s: %s", sym, exc)

            if reconciled == 0:
                logger.info("RECONCILE: Sin posiciones abiertas. Estado limpio.")
            else:
                logger.info("RECONCILE: %d posiciones cargadas.", reconciled)

        except Exception as exc:
            logger.error("Error en reconciliación de estado: %s. Arrancando con estado vacío.", exc)

    # ── Capital y riesgo helpers ──────────────────────────────────

    def _total_margin_used(self) -> float:
        """Suma total de margen usado en todas las posiciones."""
        return sum(s.total_margin_used for s in self.states.values())

    def _open_position_count(self) -> int:
        """Número de posiciones abiertas (margen > 0)."""
        return sum(1 for s in self.states.values() if s.total_margin_used > 0)

    def _check_global_margin(self, margin_usdt: float, symbol: str | None = None) -> bool:
        """Retorna True si hay capacidad global para esta orden."""
        current = self._total_margin_used()
        if current + margin_usdt > config.MAX_TOTAL_MARGIN:
            logger.info(
                "GLOBAL MARGIN CAP | Actual: %.2f + Nuevo: %.2f > Límite: %.2f",
                current, margin_usdt, config.MAX_TOTAL_MARGIN,
            )
            return False
        existing_state = self.states.get(symbol) if symbol else None
        is_new_position = existing_state is None or existing_state.total_margin_used <= 0
        if (
            is_new_position
            and self._open_position_count() >= config.MAX_OPEN_POSITIONS
            and margin_usdt > 0
        ):
            logger.info(
                "MAX POSITIONS | Actual: %d >= Límite: %d",
                self._open_position_count(), config.MAX_OPEN_POSITIONS,
            )
            return False
        return True

    async def _check_drawdown(self) -> bool:
        """
        Daily drawdown circuit breaker.
        Retorna True si el drawdown excede el umbral y debemos parar.
        """
        if config.DRY_RUN or self._start_balance <= 0:
            return False

        # Si ya disparado, mantener pausa hasta el día siguiente (o 1h)
        if self._daily_drawdown_triggered:
            elapsed = time.time() - self._daily_drawdown_ts
            if elapsed < 3600:  # 1h de pausa mínima
                logger.warning("Daily drawdown pausa activa. %.0fs restantes.", 3600 - elapsed)
                return True
            self._daily_drawdown_triggered = False
            logger.info("Daily drawdown pausa finalizada. Reanudando operaciones.")

        try:
            balance = await _retry(lambda: self.exchange.fetch_balance(), "fetch_balance")
            await asyncio.sleep(config.RATE_LIMIT_PAUSE)
            total_equity = float(
                balance.get("total", {}).get("USDT")
                or balance.get("total", {}).get("USD")
                or 0
            )
            if total_equity <= 0:
                return False

            self._peak_balance = max(self._peak_balance, total_equity)

            dd_from_start = (total_equity - self._start_balance) / self._start_balance * 100
            dd_from_peak = (total_equity - self._peak_balance) / self._peak_balance * 100

            if dd_from_start <= config.MAX_DAILY_DRAWDOWN_PCT or dd_from_peak <= config.MAX_DAILY_DRAWDOWN_PCT:
                logger.critical(
                    "DAILY DRAWDOWN | Equity: %.2f | Start: %.2f | Peak: %.2f | "
                    "DD start: %.2f%% | DD peak: %.2f%% | OPERACIONES PAUSADAS 1h",
                    total_equity, self._start_balance, self._peak_balance,
                    dd_from_start, dd_from_peak,
                )
                await notifications.notify_alert(
                    "drawdown", "GLOBAL",
                    f"Equity {total_equity:.2f} USDT | DD start {dd_from_start:.2f}% | DD peak {dd_from_peak:.2f}%",
                )
                self._daily_drawdown_triggered = True
                self._daily_drawdown_ts = time.time()
                return True

            return False
        except Exception as exc:
            logger.error("Error check_drawdown: %s", exc)
            return False

    async def _refresh_learning(self, force: bool = False) -> None:
        """Refresca ajustes aprendidos sin bloquear el event loop."""
        if self.learning is None:
            return
        try:
            await asyncio.to_thread(self.learning.refresh, force)
        except Exception as exc:
            logger.debug("Autoaprendizaje refresh skip: %s", exc)

    def _apply_learning_score(self, symbol: str, base_score: float) -> float:
        if self.learning is None:
            return base_score
        adjustment = self.learning.get_adjustment(symbol)
        self.states[symbol].last_learning_multiplier = adjustment.score_multiplier
        return max(0.0, min(1.0, base_score * adjustment.score_multiplier))

    async def _check_spread(self, symbol: str) -> bool:
        """Retorna True si el spread es aceptable."""
        try:
            ticker = await _retry(
                lambda: self.exchange.fetch_ticker(symbol),
                f"spread({symbol})",
            )
            await asyncio.sleep(config.RATE_LIMIT_PAUSE)
            bid = ticker.get("bid")
            ask = ticker.get("ask")
            if bid and ask and bid > 0:
                spread = (ask - bid) / bid
                if spread > config.MAX_SPREAD_PCT:
                    logger.info(
                        "SPREAD GATE %s | %.4f%% > %.4f%%",
                        symbol, spread * 100, config.MAX_SPREAD_PCT * 100,
                    )
                    return False
            return True
        except Exception as exc:
            logger.debug("Spread check fail %s: %s", symbol, exc)
            return True  # Si falla, no bloquear por defecto

    # ── Circuit breaker ───────────────────────────────────────────

    def _record_success(self) -> None:
        self._consecutive_errors = 0
        if self._circuit_open:
            logger.info("Circuit breaker: CERRADO. Operación normal reanudada.")
            self._circuit_open = False

    def _record_error(self) -> None:
        self._consecutive_errors += 1
        if self._consecutive_errors >= config.CIRCUIT_BREAKER_MAX_ERRORS:
            if not self._circuit_open:
                logger.critical(
                    "CIRCUIT BREAKER ABIERTO: %d errores consecutivos. "
                    "Pausa de %ds antes de reintentar.",
                    self._consecutive_errors, config.CIRCUIT_BREAKER_PAUSE_SECONDS,
                )
                self._circuit_open = True
                self._circuit_open_ts = time.time()

    def _is_circuit_open(self) -> bool:
        if not self._circuit_open:
            return False
        elapsed = time.time() - self._circuit_open_ts
        if elapsed >= config.CIRCUIT_BREAKER_PAUSE_SECONDS:
            logger.info("Circuit breaker: tiempo de pausa cumplido. Reintentando.")
            self._circuit_open = False
            self._consecutive_errors = 0
            return False
        remaining = config.CIRCUIT_BREAKER_PAUSE_SECONDS - elapsed
        logger.warning("Circuit breaker ABIERTO. %.0fs restantes.", remaining)
        return True

    # ── Configuración por símbolo ─────────────────────────────────

    async def _ensure_config(self, symbol: str) -> None:
        """
        Configura margin mode y leverage para el símbolo.

        FIX KuCoin #330005: set_margin_mode y set_leverage deben
        ejecutarse con los params correctos o la orden falla con
        "margin mode does not match". Se pasan coordinados.
        """
        if symbol in self._configured_symbols:
            return

        try:
            await _retry(
                lambda: self.exchange.set_margin_mode(
                    config.MARGIN_MODE,
                    symbol,
                    params={"leverage": config.LEVERAGE},
                ),
                f"set_margin_mode({symbol})",
            )
        except ccxt_async.ExchangeError as exc:
            logger.debug("set_margin_mode %s: %s (posiblemente ya configurado)", symbol, exc)

        try:
            await _retry(
                lambda: self.exchange.set_leverage(
                    config.LEVERAGE,
                    symbol,
                    params={"marginMode": config.MARGIN_MODE},
                ),
                f"set_leverage({symbol})",
            )
        except ccxt_async.ExchangeError as exc:
            logger.debug("set_leverage %s: %s (posiblemente ya configurado)", symbol, exc)

        self._configured_symbols.add(symbol)
        await asyncio.sleep(config.RATE_LIMIT_PAUSE)

    # ── Funding: extracción de timing ─────────────────────────────

    def _extract_interval_hours(self, symbol: str) -> float:
        """
        Extrae el intervalo de funding del par desde los metadatos
        del mercado cargados en load_markets().

        KuCoin expone 'fundingInterval' en milisegundos en market['info'].
        Fallback a 8h si no está disponible.
        """
        market = self.exchange.markets.get(symbol, {})
        info = market.get("info", {})

        # KuCoin API: fundingInterval en ms
        interval_ms = info.get("fundingInterval") or info.get("settleFreqMs")
        if interval_ms:
            try:
                return float(interval_ms) / (1000 * 3600)  # ms → horas
            except (TypeError, ValueError):
                pass

        # Fallback por nombre de símbolo: BTC y ETH son 8h en KuCoin
        base = market.get("base", "")
        if base in ("BTC", "ETH", "XBT"):
            return 8.0

        return 4.0  # Default para altcoins en KuCoin

    def _parse_funding_timing(self, fr_data: dict, symbol: str) -> FundingTiming:
        """
        Extrae datos de timing de la respuesta de fetch_funding_rate.

        KuCoin retorna (vía ccxt):
            fundingRate          → ccxt['fundingRate']
            fundingTime          → ccxt['fundingTimestamp'] (ms, PRÓXIMO snapshot)
            nextFundingRate      → ccxt['nextFundingRate']
            fundingRateCap       → fr_data.get('info', {}).get('fundingRateCap')
            fundingRateFloor     → fr_data.get('info', {}).get('fundingRateFloor')

        DIFERENCIA CON MEXC:
        En KuCoin, 'fundingTimestamp' es el PRÓXIMO snapshot (no el último).
        El "último" se calcula restando el intervalo.
        """
        ft = FundingTiming()
        ft.interval_hours = self._extract_interval_hours(symbol)

        ft.funding_rate = float(fr_data.get("fundingRate") or 0.0)
        ft.next_funding_rate = float(fr_data.get("nextFundingRate") or 0.0)

        # En KuCoin via ccxt, fundingTimestamp suele representar el proximo
        # snapshot. Algunas versiones tambien exponen nextFundingTimestamp.
        ft.next_funding_ts_ms = int(
            fr_data.get("nextFundingTimestamp")
            or fr_data.get("fundingTimestamp")
            or fr_data.get("timestamp")
            or 0
        )

        # KuCoin no devuelve el último timestamp directamente.
        # Lo calculamos restando el intervalo al próximo.
        interval_ms = int(ft.interval_hours * 3600 * 1000)
        if ft.next_funding_ts_ms > 0:
            ft.last_funding_ts_ms = ft.next_funding_ts_ms - interval_ms
        else:
            ft.last_funding_ts_ms = 0

        # Cap/floor del FR para este par (desde info raw)
        info = fr_data.get("info", {})
        try:
            ft.funding_rate_cap = float(info.get("fundingRateCap", 0.003))
            ft.funding_rate_floor = float(info.get("fundingRateFloor", -0.003))
        except (TypeError, ValueError):
            pass

        now_ms = int(time.time() * 1000)

        if ft.next_funding_ts_ms > 0:
            ft.minutes_to_next = max(0.0, (ft.next_funding_ts_ms - now_ms) / 60_000)

        if ft.last_funding_ts_ms > 0:
            ft.minutes_since_last = max(0.0, (now_ms - ft.last_funding_ts_ms) / 60_000)

        return ft

    def _detect_funding_collected(self, symbol: str, ft: FundingTiming) -> None:
        """
        Detecta si un snapshot de funding acaba de ocurrir.

        Dos métodos (OR):
        1. nextFundingTimestamp avanzó → epoch completado.
        2. minutes_since_last ≤ blindfold_minutes → justo después del snapshot.
        """
        state = self.states[symbol]

        if state.total_margin_used <= 0:
            state.funding_collected = False
            return

        epoch_changed = (
            state.last_known_next_funding_ms > 0
            and ft.next_funding_ts_ms > 0
            and ft.next_funding_ts_ms > state.last_known_next_funding_ms
        )

        just_after = ft.minutes_since_last <= ft.blindfold_minutes

        if ft.next_funding_ts_ms > 0:
            state.last_known_next_funding_ms = ft.next_funding_ts_ms

        if not state.funding_collected and (epoch_changed or just_after):
            state.funding_collected = True
            state.funding_collected_ts = time.time()
            logger.info(
                "FUNDING COBRADO %s | FR: %.4f%% | Ciclo: %.0fh | "
                "Net post-fees: %.4f%% | Modo post-cobro activado.",
                symbol, ft.funding_rate * 100, ft.interval_hours,
                ft.net_fr_after_fees * 100,
            )

    def _is_in_blindfold(self, ft: FundingTiming) -> bool:
        """¿Estamos en la ventana ciega post-snapshot?"""
        return ft.minutes_since_last <= ft.blindfold_minutes

    # ── Scanner paralelo ──────────────────────────────────────────

    def _score_opportunity(self, symbol: str, ft: FundingTiming, ticker: dict) -> float:
        """
        Score 0.0–1.0 para rankear oportunidades.
        Si ENABLE_MATH_SCORING es True, usa power_score del arsenal matemático.
        Si no, fallback al scoring lineal legacy.
        """
        state = self.states[symbol]

        if not config.ENABLE_MATH_SCORING:
            # Legacy scoring
            annualized = abs(ft.annualized_fr_pct)
            fr_score = min(annualized / 200.0, 1.0)
            vol = ticker.get("quoteVolume") or 0.0
            liq_score = min(vol / config.VOLUME_LIQ_REFERENCE, 1.0)
            base_score = fr_score * config.SCORE_WEIGHT_FR + liq_score * config.SCORE_WEIGHT_LIQ
            return self._apply_learning_score(symbol, base_score)

        # ── Scoring matemático integrado ─────────────────────────────
        fr_arr = np.array(state.fr_history, dtype=float)
        lr_arr = np.array(state.log_return_history, dtype=float)
        oi_arr = np.array(state.oi_change_history, dtype=float)
        px_arr = np.array(state.price_history, dtype=float)
        vol = ticker.get("quoteVolume") or 0.0

        if len(fr_arr) < 10 or len(lr_arr) < 10:
            # Datos insuficientes → legacy con penalización
            annualized = abs(ft.annualized_fr_pct)
            fr_score = min(annualized / 200.0, 1.0)
            return self._apply_learning_score(symbol, fr_score * 0.5)

        try:
            score_result = math_engine.power_score(
                funding_rate=ft.funding_rate,
                next_funding_rate=ft.next_funding_rate,
                fr_history=fr_arr,
                log_returns=lr_arr,
                oi_changes=oi_arr,
                price_history=px_arr,
                volume_24h=vol,
                interval_hours=ft.interval_hours,
                min_volume=config.MIN_VOLUME_24H,
                liq_reference=config.VOLUME_LIQ_REFERENCE,
                entropy_max=config.ENTROPY_MAX_ENTRY,
            )
            state.last_power_score = score_result.get("score", 0.0)
            # Cachear métricas para uso posterior
            state.last_tail_alpha = score_result.get("tail_alpha")
            state.last_tail_risk = score_result.get("tail_risk")
            state.last_hurst_fr = score_result.get("hurst")
            state.last_fr_regime = "persistent_negative" if score_result.get("hurst", 0.5) > 0.58 and ft.funding_rate < 0 else "unknown"
            state.last_entropy_fr = score_result.get("entropy_fr")
            state.last_te_oi_to_fr = score_result.get("te_oi_to_fr")
            state.last_multifractal_width = score_result.get("mf_delta_alpha")
            return self._apply_learning_score(symbol, state.last_power_score)
        except Exception as exc:
            logger.debug("power_score error %s: %s", symbol, exc)
            # Fallback legacy
            annualized = abs(ft.annualized_fr_pct)
            return self._apply_learning_score(symbol, min(annualized / 200.0, 1.0))

    async def scan(self) -> list[tuple[str, FundingTiming]]:
        """
        Retorna (symbol, FundingTiming) que cumplen todos los gates:
          1. Volumen 24h > MIN_VOLUME_24H
          2. Funding Rate ≤ MAX_FUNDING_RATE
          3. Dentro de la ventana de entrada (fracción del ciclo)
          4. NO en blindfold post-snapshot
          5. Net FR después de fees de cierre > 0
          6. Predicted funding sanity check (opcional)
          7. Spread gate

        Optimización: fetch_funding_rate en paralelo con semáforo
        para evitar saturar el rate limit de KuCoin.
        """
        tickers: dict = await _retry(
            lambda: self.exchange.fetch_tickers(),
            "fetch_tickers",
        )
        await asyncio.sleep(config.RATE_LIMIT_PAUSE)

        # Gate 1: filtro de volumen + guardar precio de scan
        candidates: list[str] = []
        for symbol, ticker in tickers.items():
            market = self.exchange.markets.get(symbol)
            if not market or not market.get("swap") or not market.get("linear"):
                continue
            if not market.get("active", True):
                continue
            vol_quote = ticker.get("quoteVolume") or 0.0
            if vol_quote >= config.MIN_VOLUME_24H:
                candidates.append(symbol)
                # Guardar precio de scan para price drift gate
                self.states[symbol].scan_price = float(
                    ticker.get("last") or ticker.get("close") or 0
                )

        logger.info("SCAN: %d candidatos por volumen (>= %.0f USDT).", len(candidates), config.MIN_VOLUME_24H)

        # Fetch funding rates en paralelo (semáforo = 5 concurrentes)
        semaphore = asyncio.Semaphore(5)
        qualified: list[tuple[str, FundingTiming, float]] = []  # (sym, ft, score)
        lock = asyncio.Lock()

        # Contadores diagnóstico (thread-safe porque se usan con asyncio.Lock)
        diag: dict[str, int | float | str] = {
            "fr_neg": 0,        # tienen FR negativo (cualquier valor)
            "fr_ok": 0,         # FR <= MAX_FUNDING_RATE
            "window_ok": 0,     # dentro de la ventana de entrada
            "passed_all": 0,    # pasaron todos los gates
            "best_fr": 0.0,     # FR más negativo visto en este scan
            "best_sym": "",     # símbolo con FR más negativo
            "best_min": 9999.0, # minutos al snapshot del mejor FR
        }
        diag_lock = asyncio.Lock()

        async def _check_symbol(symbol: str) -> None:
            async with semaphore:
                try:
                    fr_data = await _retry(
                        lambda s=symbol: self.exchange.fetch_funding_rate(s),
                        f"fetch_funding_rate({symbol})",
                    )
                    await asyncio.sleep(config.RATE_LIMIT_PAUSE)

                    state = self.states[symbol]
                    ft = self._parse_funding_timing(fr_data, symbol)
                    ft.scan_price = state.scan_price

                    # ── Actualizar históricos matemáticos ─────────────────
                    state.fr_history.append(ft.funding_rate)
                    # Log return: precio actual vs último precio histórico
                    prev_price = state.price_history[-1] if state.price_history else 0.0
                    if prev_price > 0 and ft.scan_price > 0:
                        log_ret = math.log(ft.scan_price / prev_price)
                        state.log_return_history.append(log_ret)
                    if ft.scan_price > 0:
                        state.price_history.append(ft.scan_price)

                    # ── Tracking diagnóstico ──────────────────────────────
                    if ft.funding_rate < 0:
                        async with diag_lock:
                            diag["fr_neg"] += 1
                            if ft.funding_rate < diag["best_fr"]:
                                diag["best_fr"] = ft.funding_rate
                                diag["best_sym"] = symbol
                                diag["best_min"] = ft.minutes_to_next

                    # ── Gates baratos primero (early-exit antes del cómputo caro) ─

                    # Gate 1: FR suficientemente negativo
                    if ft.funding_rate > config.MAX_FUNDING_RATE:
                        logger.debug(
                            "SCAN SKIP %s | FR %.4f%% > umbral %.4f%% (no suficientemente negativo).",
                            symbol, ft.funding_rate * 100, config.MAX_FUNDING_RATE * 100,
                        )
                        return

                    async with diag_lock:
                        diag["fr_ok"] += 1

                    # Gate 2: dentro de la ventana de entrada
                    if ft.minutes_to_next > ft.entry_window_minutes:
                        logger.debug(
                            "SCAN SKIP %s | FR: %.4f%% OK | "
                            "Faltan %.1f min (ventana %.1f min, ciclo %.0fh)",
                            symbol, ft.funding_rate * 100,
                            ft.minutes_to_next, ft.entry_window_minutes,
                            ft.interval_hours,
                        )
                        return

                    async with diag_lock:
                        diag["window_ok"] += 1

                    # Gate 3: no en blindfold post-snapshot
                    if self._is_in_blindfold(ft):
                        logger.debug(
                            "SCAN SKIP %s | Blindfold activo (%.1f min post-snapshot)",
                            symbol, ft.minutes_since_last,
                        )
                        return

                    # Gate 4: net FR positivo después de fee de cierre (0.06% taker)
                    if ft.net_fr_after_fees <= 0:
                        logger.debug(
                            "SCAN SKIP %s | FR %.4f%% no cubre fee de cierre 0.06%%",
                            symbol, ft.funding_rate * 100,
                        )
                        return

                    # Gate 5: predicted funding sanity check
                    # IMPORTANTE: solo aplicar si KuCoin proveyó el dato.
                    # Cuando nextFundingRate está ausente, ccxt devuelve 0.0,
                    # y 0.0 >= 0.0 bloquearía todas las entradas.
                    # Usamos EXIT_FUNDING_RATE (0.0): solo descartar si el próximo
                    # ciclo el FR ya se volvió positivo (sin tesis de cobro).
                    if config.USE_PREDICTED_FUNDING_CHECK and ft.next_funding_rate != 0.0:
                        if ft.next_funding_rate >= config.EXIT_FUNDING_RATE:
                            logger.debug(
                                "SCAN SKIP %s | Next FR %.4f%% ya es positivo — sin tesis.",
                                symbol, ft.next_funding_rate * 100,
                            )
                            return

                    # Gate 6: spread bid-ask (calculado inline desde tickers del scan)
                    # Evita un fetch_ticker() adicional por símbolo candidato.
                    _ticker_snap = tickers.get(symbol, {})
                    _bid = _ticker_snap.get("bid")
                    _ask = _ticker_snap.get("ask")
                    if _bid and _ask and _bid > 0:
                        _spread = (_ask - _bid) / _bid
                        if _spread > config.MAX_SPREAD_PCT:
                            logger.info(
                                "SPREAD GATE %s | %.4f%% > %.4f%%",
                                symbol, _spread * 100, config.MAX_SPREAD_PCT * 100,
                            )
                            return

                    # ── Gates matemáticos (hard constraints, cómputo intensivo) ─
                    # Solo se ejecutan para símbolos que pasaron los 6 gates baratos.

                    # Tail Risk Gate
                    if config.ENABLE_TAIL_GATES and len(state.log_return_history) >= 20:
                        try:
                            tail_info = math_engine.PowerLawTail.tail_risk_index(
                                np.array(state.log_return_history, dtype=float)
                            )
                            state.last_tail_alpha = tail_info["alpha"]
                            state.last_tail_pvalue = tail_info["pvalue"]
                            state.last_tail_risk = tail_info["risk"]
                            if tail_info["risk"] > config.TAIL_RISK_MAX_ENTRY:
                                logger.info(
                                    "MATH GATE BLOCK %s | Tail risk %.2f > %.2f (α=%.2f). Entrada abortada.",
                                    symbol, tail_info["risk"], config.TAIL_RISK_MAX_ENTRY, tail_info["alpha"]
                                )
                                return
                        except Exception as exc:
                            logger.debug("tail_gate skip %s: %s", symbol, exc)

                    # Entropy Gate
                    if config.ENABLE_ENTROPY_GATES and len(state.fr_history) >= 10:
                        try:
                            h_norm = math_engine.InformationMetrics.shannon_entropy_normalized(
                                np.array(state.fr_history, dtype=float), bins="fd"
                            )
                            state.last_entropy_fr = h_norm
                            if h_norm > config.ENTROPY_MAX_ENTRY:
                                logger.info(
                                    "MATH GATE BLOCK %s | Entropy %.3f > %.2f (ruido puro). Entrada abortada.",
                                    symbol, h_norm, config.ENTROPY_MAX_ENTRY
                                )
                                return
                        except Exception as exc:
                            logger.debug("entropy_gate skip %s: %s", symbol, exc)

                    # Multifractal Gate
                    if config.ENABLE_MULTIFRACTAL_GATES and len(state.log_return_history) >= 64:
                        try:
                            mf = math_engine.MultifractalSpectrum.mf_dfa(
                                np.array(state.log_return_history, dtype=float)
                            )
                            delta_alpha = math_engine.MultifractalSpectrum.multifractal_width(
                                mf["h_q"], mf["q"]
                            )
                            state.last_multifractal_width = delta_alpha
                            if delta_alpha > config.MULTIFRACTAL_WIDTH_ABORT:
                                logger.info(
                                    "MATH GATE BLOCK %s | Δα %.3f > %.2f (multifractalidad extrema). Entrada abortada.",
                                    symbol, delta_alpha, config.MULTIFRACTAL_WIDTH_ABORT
                                )
                                return
                        except Exception as exc:
                            logger.debug("mf_gate skip %s: %s", symbol, exc)

                    # ── Microestructura ─────────────────────────────────
                    ticker = tickers.get(symbol, {})
                    state = self.states[symbol]
                    if state.micro_analyzer is None:
                        state.micro_analyzer = microstructure_engine.MicrostructureAnalyzer()
                    micro = state.micro_analyzer.feed_ticker(ticker)
                    if micro.get("valid"):
                        state.last_micro_imbalance = micro.get("imbalance", 0.0)
                        state.last_micro_vpin = micro.get("vpin_proxy", 0.0)
                        tox = state.micro_analyzer.toxicity_signal()
                        if tox.get("toxic"):
                            logger.info(
                                "MICRO GATE BLOCK %s | VPIN %.3f (toxic). Entrada abortada.",
                                symbol, tox.get("vpin", 0.0)
                            )
                            return

                    # Score
                    score = self._score_opportunity(symbol, ft, ticker)

                    async with lock:
                        qualified.append((symbol, ft, score))
                    async with diag_lock:
                        diag["passed_all"] += 1

                    logger.info(
                        "SCAN OK %s | FR: %.4f%% | Ciclo: %.0fh | "
                        "Net: %.4f%% | Snapshot: %.1f min | Score: %.3f",
                        symbol, ft.funding_rate * 100,
                        ft.interval_hours, ft.net_fr_after_fees * 100,
                        ft.minutes_to_next, score,
                    )

                except Exception as exc:
                    logger.debug("Funding skip %s: %s", symbol, exc)

        await asyncio.gather(*[_check_symbol(s) for s in candidates])

        # ── Resumen diagnóstico del scan (INFO level, visible siempre) ─
        best_fr_pct = diag["best_fr"] * 100
        best_info = (
            f" | Mejor FR: {best_fr_pct:.4f}% ({diag['best_sym']}, {diag['best_min']:.0f} min al snapshot)"
            if diag["best_sym"]
            else " | Sin FRs negativos detectados"
        )
        logger.info(
            "SCAN RESUMEN | Vol OK: %d | FR neg: %d | FR <= %.2f%%: %d | "
            "En ventana: %d | Calificados: %d%s",
            len(candidates),
            diag["fr_neg"],
            config.MAX_FUNDING_RATE * 100,
            diag["fr_ok"],
            diag["window_ok"],
            diag["passed_all"],
            best_info,
        )
        # Alimentar stats de heartbeat
        self._scan_cycles += 1
        if diag["best_fr"] < self._best_fr_seen:
            self._best_fr_seen = diag["best_fr"]
            self._best_fr_symbol = diag["best_sym"]

        # Ranking: solo las TOP_N mejores
        qualified.sort(key=lambda x: x[2], reverse=True)
        top_n = qualified[: config.TOP_N_OPPORTUNITIES]

        if len(qualified) > config.TOP_N_OPPORTUNITIES:
            logger.info(
                "RANKING: %d calificaron, top %d seleccionadas.",
                len(qualified), config.TOP_N_OPPORTUNITIES,
            )

        return [(sym, ft) for sym, ft, _ in top_n]

    # ── OI: captura y análisis ────────────────────────────────────

    async def _record_oi(self, symbol: str) -> None:
        try:
            oi_data = await _retry(
                lambda: self.exchange.fetch_open_interest(symbol),
                f"fetch_oi({symbol})",
            )
            await asyncio.sleep(config.RATE_LIMIT_PAUSE)

            # openInterestValue = USDT-denominado (confiable)
            # openInterestAmount = en contratos (ambiguo, ignorar)
            oi_value = oi_data.get("openInterestValue") or 0.0
            if oi_value > 0:
                state = self.states[symbol]
                state.oi_history.append(
                    OISnapshot(timestamp=time.time(), value=oi_value)
                )
                # Actualizar oi_change_history (cambio porcentual vs último valor)
                if state.oi_history:
                    prev = list(state.oi_history)[-2] if len(state.oi_history) >= 2 else None
                    if prev is not None and prev.value > 0:
                        change_pct = (oi_value - prev.value) / prev.value
                        state.oi_change_history.append(change_pct)
        except Exception as exc:
            logger.debug("OI fetch skip %s: %s", symbol, exc)

    def _classify_oi(self, symbol: str) -> str | None:
        """Clasifica tendencia de OI: 'falling' | 'lateral' | 'rising_strong' | None."""
        change = self._oi_change_pct(symbol)
        if change is None:
            return None
        if change < config.OI_FALLING_THRESHOLD:
            return "falling"
        elif config.OI_LATERAL_LOW <= change <= config.OI_LATERAL_HIGH:
            return "lateral"
        elif change > config.OI_RISING_STRONG_THRESHOLD:
            return "rising_strong"
        else:
            return "lateral"

    def _oi_change_pct(self, symbol: str) -> float | None:
        """Cambio % crudo de OI en la ventana de 5 min."""
        history = self.states[symbol].oi_history
        if len(history) < 2:
            return None

        now = time.time()
        cutoff = now - config.OI_WINDOW_SECONDS

        oldest: OISnapshot | None = None
        for snap in history:
            if snap.timestamp >= cutoff:
                oldest = snap
                break

        if oldest is None or oldest.value == 0:
            return None

        latest = history[-1]
        return (latest.value - oldest.value) / oldest.value

    def _is_oi_crowded(self, symbol: str) -> bool:
        """
        Detector de saturación de bots vía EVT + z-score fallback.

        Primero intenta detectar crowding anómalo con EVT (POT/GPD),
        que es más apropiado para colas pesadas que un z-score gaussiano.
        Si no hay datos suficientes para EVT, fallback al z-score.
        """
        state = self.states[symbol]
        history = state.oi_history
        if len(history) < config.OI_CROWDING_WINDOW + 2:
            return False

        # Calcular cambios consecutios de OI
        values = [snap.value for snap in list(history)[-config.OI_CROWDING_WINDOW:]]
        if len(values) < 2:
            return False

        changes = np.diff(values) / np.array(values[:-1])
        if len(changes) < 2:
            return False

        current_change = changes[-1]

        # ── Método 1: EVT (preferido para colas pesadas) ────────────
        if len(changes) >= 20:
            try:
                is_evt_crowded = math_engine.ExtremeValueEngine.evt_crowding_detection(
                    changes, current_change
                )
                if is_evt_crowded:
                    logger.info(
                        "OI EVT CROWDING %s | Change: %.4f excede VaR_95 GPD | "
                        "Saturación anómala detectada.", symbol, current_change
                    )
                    return True
            except Exception as exc:
                logger.debug("EVT crowding skip %s: %s", symbol, exc)

        # ── Método 2: z-score gaussiano (fallback) ──────────────────
        mean = float(np.mean(changes[:-1]))  # excluir el actual
        std = float(np.std(changes[:-1], ddof=0))
        if std == 0:
            return False

        z_score = (current_change - mean) / std
        if z_score > config.OI_CROWDING_ZSCORE_THRESHOLD:
            logger.info(
                "OI CROWDING %s | z-score: %.2f (umbral: %.1f) | "
                "Saturación de bots detectada. Entrada abortada.",
                symbol, z_score, config.OI_CROWDING_ZSCORE_THRESHOLD,
            )
            return True

        return False

    # ── Cálculo de amount ─────────────────────────────────────────

    def _margin_to_amount(self, symbol: str, margin_usdt: float, price: float) -> float | None:
        """
        Convierte margen USDT a cantidad de contratos normalizada.
        amount = (margin * leverage) / (price * contractSize)
        """
        if price <= 0:
            return None

        market = self.exchange.markets.get(symbol)
        if not market:
            return None

        contract_size = market.get("contractSize", 1.0) or 1.0
        notional = margin_usdt * config.LEVERAGE
        raw_amount = notional / (price * contract_size)

        normalized = self.exchange.amount_to_precision(symbol, raw_amount)
        normalized_float = float(normalized)

        if normalized_float <= 0:
            logger.warning(
                "%s: amount normalizado a 0 (margin=%.2f, price=%.4f, cs=%.6f).",
                symbol, margin_usdt, price, contract_size,
            )
            return None

        # Validar contra límites del mercado
        min_amount = (market.get("limits") or {}).get("amount", {}).get("min", 0)
        if min_amount and normalized_float < min_amount:
            logger.warning(
                "%s: amount %.6f < mínimo %.6f del mercado.",
                symbol, normalized_float, min_amount,
            )
            return None

        return normalized_float

    def _adaptive_margin(self, base_margin: float, ft: FundingTiming, symbol: str = "") -> float:
        """
        Escala el margen base proporcional a la calidad del edge.
        FR más negativo y ciclo más corto = más tamaño.
        Ahora con ajustes matemáticos por tail risk, volatilidad, persistencia
        y multifractalidad.
        """
        if not config.ADAPTIVE_SIZING:
            return base_margin

        ratio = abs(ft.funding_rate) / abs(config.MAX_FUNDING_RATE)
        multiplier = min(ratio, config.ADAPTIVE_SIZING_MAX_MULTIPLIER)

        # Bonus por frecuencia (1h vs 8h = 8x más oportunidades)
        if ft.interval_hours > 0:
            freq_boost = 8.0 / ft.interval_hours
            multiplier *= min(freq_boost, 2.0)

        # ── Ajustes matemáticos ─────────────────────────────────────
        if symbol:
            state = self.states[symbol]

            # Penalización por tail risk (colas pesadas)
            if state.last_tail_alpha is not None and not math.isnan(state.last_tail_alpha):
                if state.last_tail_alpha < 2.5:
                    multiplier *= 0.3
                elif state.last_tail_alpha < 3.0:
                    multiplier *= 0.6
                elif state.last_tail_alpha < 3.5:
                    multiplier *= 0.85

            # Penalización / bonus por régimen de volatilidad
            if state.last_vol_regime == "extreme":
                multiplier *= 0.4
            elif state.last_vol_regime == "calm":
                multiplier *= 1.15

            # Bonus / penalización por persistencia (Hurst)
            if state.last_hurst_fr is not None and not math.isnan(state.last_hurst_fr):
                if state.last_hurst_fr > config.HURST_PERSISTENCE_THRESHOLD:
                    multiplier *= 1.0 + (state.last_hurst_fr - config.HURST_PERSISTENCE_THRESHOLD) * 1.5
                elif state.last_hurst_fr < config.HURST_MEANREVERSION_THRESHOLD:
                    multiplier *= max(0.5, 1.0 - (config.HURST_MEANREVERSION_THRESHOLD - state.last_hurst_fr))

            # Penalización multifractal
            if state.last_multifractal_width is not None and not math.isnan(state.last_multifractal_width):
                penalty = math_engine.MultifractalSpectrum.width_penalty(
                    state.last_multifractal_width, threshold=config.MULTIFRACTAL_WIDTH_THRESHOLD
                )
                multiplier *= (1.0 - penalty)

        if symbol and self.learning is not None:
            adjustment = self.learning.get_adjustment(symbol)
            multiplier *= adjustment.size_multiplier
            self.states[symbol].last_learning_multiplier = adjustment.size_multiplier

        return base_margin * multiplier

    # ── Ejecución de órdenes con fill verificado ──────────────────

    async def _confirm_fill(self, symbol: str, order_id: str, fallback_price: float) -> float:
        """
        Obtiene el precio de fill real vía fetch_order.
        Retorna average si existe, si no el fallback.

        CRÍTICO: En KuCoin, las órdenes de mercado pueden retornar
        sin 'average' en la respuesta inicial. fetch_order confirma
        el precio real una vez liquidada la orden.
        """
        try:
            await asyncio.sleep(0.3)  # Pequeña pausa para que el exchange liquide
            confirmed = await _retry(
                lambda oid=order_id: self.exchange.fetch_order(oid, symbol),
                f"fetch_order({order_id})",
            )
            await asyncio.sleep(config.RATE_LIMIT_PAUSE)
            avg = confirmed.get("average") or confirmed.get("price")
            if avg and float(avg) > 0:
                return float(avg)
        except Exception as exc:
            logger.warning("fetch_order %s/%s falló: %s. Usando precio fallback.", symbol, order_id, exc)
        return fallback_price

    async def _open_long(
        self, symbol: str, margin_usdt: float, price: float, rule: str, ft: FundingTiming | None = None
    ) -> bool:
        """Ejecuta Market Buy Long. Retorna True si se ejecutó."""
        state = self.states[symbol]

        # Global margin cap
        if not self._check_global_margin(margin_usdt, symbol):
            return False

        # Per-coin margin cap
        if state.total_margin_used + margin_usdt > config.MAX_MARGIN_PER_COIN:
            logger.info(
                "%s: margen (%.2f + %.2f) > límite %.2f. DCA detenido.",
                symbol, state.total_margin_used, margin_usdt, config.MAX_MARGIN_PER_COIN,
            )
            return False

        elapsed = time.time() - state.last_order_ts
        if elapsed < config.COOLDOWN_SECONDS:
            logger.debug(
                "%s: cooldown (%.0fs restantes).",
                symbol, config.COOLDOWN_SECONDS - elapsed,
            )
            return False

        # Adaptive sizing
        if ft is not None and config.ADAPTIVE_SIZING and rule in ("standard", "squeeze", "defense"):
            adjusted = self._adaptive_margin(margin_usdt, ft, symbol)
            # No exceder el margen disponible para esta moneda
            available = config.MAX_MARGIN_PER_COIN - state.total_margin_used
            margin_usdt = min(adjusted, available)
            if margin_usdt < 1.0:
                logger.debug("%s: adaptive sizing reduce margen a <1 USDT. Abortado.", symbol)
                return False

        amount = self._margin_to_amount(symbol, margin_usdt, price)
        if amount is None:
            return False

        # ── DRY-RUN ───────────────────────────────────────────────
        if config.DRY_RUN:
            order_id = f"dry-run-{int(time.time() * 1000)}"
            fill_price = price
            state.last_entry_price = fill_price
            state.total_margin_used += margin_usdt
            state.last_order_ts = time.time()
            logger.info(
                "[DRY-RUN] BUY %s | %s | $%.2f margin | %.6f amt | @%.6f | OID:%s",
                symbol, rule, margin_usdt, amount, fill_price, order_id,
            )
            await db_local.log_trade(
                symbol=symbol, side="buy", rule=rule,
                margin_usdt=margin_usdt, amount=amount,
                price=fill_price, order_id=order_id, status="dry_run",
                fill_confirmed=True,
            )
            await notifications.notify_trade(
                side="buy", symbol=symbol, rule=rule,
                margin_usdt=margin_usdt, amount=amount, price=fill_price,
            )
            return True

        # ── ORDEN REAL ────────────────────────────────────────────
        await self._ensure_config(symbol)

        order = await _retry(
            lambda: self.exchange.create_order(
                symbol, "market", "buy", amount,
                params={"marginMode": config.MARGIN_MODE, "leverage": config.LEVERAGE},
            ),
            f"buy({symbol})",
        )
        await asyncio.sleep(config.RATE_LIMIT_PAUSE)

        order_id = order.get("id", "unknown")
        filled = float(order.get("filled") or 0)
        if filled <= 0:
            logger.warning("BUY %s | Orden ejecutada pero filled=0. OID:%s", symbol, order_id)
            return False

        # Fill price verificado (no optimista)
        initial_price = order.get("average") or order.get("price") or price
        fill_price = await self._confirm_fill(symbol, order_id, float(initial_price))
        fill_confirmed = fill_price != float(initial_price)

        # Slippage gate
        expected_price = price
        if expected_price > 0:
            slippage = abs(fill_price - expected_price) / expected_price
            if slippage > config.MAX_SLIPPAGE_PCT:
                logger.critical(
                    "SLIPPAGE ALERT %s | %.2f%% > %.2f%% | Fill: %.6f vs Expected: %.6f",
                    symbol, slippage * 100, config.MAX_SLIPPAGE_PCT * 100,
                    fill_price, expected_price,
                )
                await notifications.notify_alert(
                    "slippage", symbol,
                    f"Slippage {slippage*100:.2f}% > {config.MAX_SLIPPAGE_PCT*100:.2f}%",
                )

        # Margen real proporcional al fill (partial fill protection)
        fill_ratio = filled / amount if amount > 0 else 1.0
        real_margin = margin_usdt * fill_ratio

        status = order.get("status", "unknown")
        state.last_entry_price = fill_price
        state.total_margin_used += real_margin
        state.last_order_ts = time.time()

        logger.info(
            "BUY %s | %s | $%.2f margin (real: %.2f) | %.6f amt (filled: %.6f) | @%.6f%s | OID:%s",
            symbol, rule, margin_usdt, real_margin, amount, filled, fill_price,
            " (confirmed)" if fill_confirmed else " (estimated)",
            order_id,
        )

        await db_local.log_trade(
            symbol=symbol, side="buy", rule=rule,
            margin_usdt=real_margin, amount=filled,
            price=fill_price, order_id=order_id, status=status,
            fill_confirmed=fill_confirmed,
        )
        await notifications.notify_trade(
            side="buy", symbol=symbol, rule=rule,
            margin_usdt=real_margin, amount=filled, price=fill_price,
        )
        return True

    async def _close_long(self, symbol: str, position_amt: float, rule: str) -> bool:
        """
        Cierra 100% de la posición Long a mercado.
        Verifica partial fill antes de resetear estado.
        """
        if position_amt <= 0:
            return False

        normalized = self.exchange.amount_to_precision(symbol, position_amt)
        normalized_float = float(normalized)
        if normalized_float <= 0:
            return False

        # ── DRY-RUN ───────────────────────────────────────────────
        if config.DRY_RUN:
            order_id = f"dry-run-close-{int(time.time() * 1000)}"
            logger.info(
                "[DRY-RUN] SELL %s | %s | %.6f amt | OID:%s",
                symbol, rule, normalized_float, order_id,
            )
            state = self.states[symbol]
            state.last_entry_price = 0.0
            state.total_margin_used = 0.0
            state.last_order_ts = time.time()
            state.funding_collected = False
            state.funding_collected_ts = 0.0
            state.last_known_next_funding_ms = 0
            await db_local.log_trade(
                symbol=symbol, side="sell", rule=rule,
                margin_usdt=0.0, amount=normalized_float,
                price=0.0, order_id=order_id, status="dry_run",
                fill_confirmed=True,
            )
            await notifications.notify_trade(
                side="sell", symbol=symbol, rule=rule,
                margin_usdt=0.0, amount=normalized_float, price=0.0,
            )
            return True

        # ── ORDEN REAL ────────────────────────────────────────────
        order = await _retry(
            lambda: self.exchange.create_order(
                symbol, "market", "sell", normalized_float,
                params={
                    "marginMode": config.MARGIN_MODE,
                    "leverage": config.LEVERAGE,
                    "reduceOnly": True,
                },
            ),
            f"sell({symbol})",
        )
        await asyncio.sleep(config.RATE_LIMIT_PAUSE)

        order_id = order.get("id", "unknown")

        # Fill price verificado
        initial_price = order.get("average") or order.get("price") or 0.0
        fill_price = await self._confirm_fill(symbol, order_id, float(initial_price))
        fill_confirmed = fill_price != float(initial_price)

        status = order.get("status", "unknown")

        # Verificar partial fill: si filled < amount, posición huérfana
        filled = float(order.get("filled") or 0)
        if filled < normalized_float * 0.99:  # tolerancia 1%
            logger.warning(
                "PARTIAL FILL %s | Solicitado: %.6f | Ejecutado: %.6f | "
                "Posición parcialmente abierta. Requiere revisión manual.",
                symbol, normalized_float, filled,
            )
            await notifications.notify_alert(
                "partial_fill", symbol,
                f"Solicitado {normalized_float:.6f}, ejecutado {filled:.6f}",
            )
            # No resetear total_margin_used completamente
            fraction_closed = filled / normalized_float if normalized_float > 0 else 0
            state = self.states[symbol]
            state.total_margin_used *= (1 - fraction_closed)
        else:
            # Reset completo: fill correcto
            state = self.states[symbol]
            state.last_entry_price = 0.0
            state.total_margin_used = 0.0
            state.last_order_ts = time.time()
            state.funding_collected = False
            state.funding_collected_ts = 0.0
            state.last_known_next_funding_ms = 0

        logger.info(
            "SELL %s | %s | %.6f amt | @%.6f%s | OID:%s",
            symbol, rule, normalized_float, fill_price,
            " (confirmed)" if fill_confirmed else " (estimated)",
            order_id,
        )

        await db_local.log_trade(
            symbol=symbol, side="sell", rule=rule,
            margin_usdt=0.0, amount=normalized_float,
            price=fill_price, order_id=order_id, status=status,
            fill_confirmed=fill_confirmed,
        )
        await notifications.notify_trade(
            side="sell", symbol=symbol, rule=rule,
            margin_usdt=0.0, amount=normalized_float, price=fill_price,
        )
        return True

    # ── Evaluación de posición existente ──────────────────────────

    async def _evaluate_position(
        self, symbol: str, position: dict, ft: FundingTiming
    ) -> None:
        """
        Aplica lógica de salida y DCA sobre una posición abierta.

        CAPA 0 — Stop-loss absoluto (siempre primero):
            ROE ≤ MAX_LOSS_ROE_PCT → cerrar incondicionalmente.

        CAPA 1 — Salidas universales:
            TP estándar (ROE ≥ 15%), funding positivo.

        CAPA 2 — Modo post-cobro (tesis agotada) O hold extension:
            Si nextFundingRate sigue muy negativo, la tesis sigue viva.
            Si no:
            a. Blindfold → solo observar.
            b. TP reducido (ROE ≥ 3%).
            c. OI dump → salir.
            d. Timeout → cortar.
            e. Esperar.

        CAPA 3 — DCA pre-funding (tesis activa):
            Reglas 1-3 según OI/funding.
        """
        contracts = abs(float(position.get("contracts") or 0))
        if contracts <= 0:
            return

        entry_price = float(position.get("entryPrice") or 0)
        mark_price = float(position.get("markPrice") or 0)
        unrealized_pnl = float(position.get("unrealizedPnl") or 0)
        initial_margin = float(position.get("initialMargin") or 0)

        if entry_price <= 0 or mark_price <= 0:
            return

        roe_pct = (unrealized_pnl / initial_margin * 100) if initial_margin > 0 else 0.0
        state = self.states[symbol]
        funding_rate = ft.funding_rate
        blindfold_active = self._is_in_blindfold(ft)

        self._detect_funding_collected(symbol, ft)

        # Snapshot PNL con annualized FR
        await db_local.log_pnl(
            symbol=symbol,
            entry_price=entry_price,
            mark_price=mark_price,
            position_amt=contracts,
            unrealized_pnl=unrealized_pnl,
            roe_pct=roe_pct,
            margin_used=state.total_margin_used,
            funding_rate=funding_rate,
            funding_interval_hours=ft.interval_hours,
        )

        # ── Cálculo/caché de métricas EVT para stops ────────────────
        if config.ENABLE_EVT_STOPS and len(state.log_return_history) >= 20:
            try:
                lr = np.array(state.log_return_history, dtype=float)
                losses = -lr[lr < 0]
                if len(losses) >= 10:
                    var_95, es_99, xi, sigma = math_engine.ExtremeValueEngine.evt_var_es(
                        losses, confidence=0.99, threshold_pct=config.EVT_THRESHOLD_PCT
                    )
                    state.last_evt_xi = xi
                    state.last_evt_sigma = sigma
            except Exception as exc:
                logger.debug("EVT cache error %s: %s", symbol, exc)

        # ══════════════════════════════════════════════════════════
        # CAPA 0: Stop-loss (EVT dinámico > legacy fijo)
        # ══════════════════════════════════════════════════════════
        state = self.states[symbol]
        log_returns = np.array(state.log_return_history, dtype=float)

        # Stop-loss EVT en espacio logarítmico
        if (
            config.ENABLE_EVT_STOPS
            and len(log_returns) >= 30
            and entry_price > 0
        ):
            try:
                evt_sl_price = math_engine.ExtremeValueEngine.dynamic_stop_loss_logspace(
                    entry_price, log_returns, confidence=config.EVT_CONFIDENCE_SL
                )
                state.last_evt_stop_price = evt_sl_price

                if mark_price <= evt_sl_price:
                    logger.critical(
                        "EVT STOP LOSS %s | Price: %.4f ≤ Stop: %.4f | "
                        "ROE: %.2f%% | GPD ξ=%.3f σ=%.3f",
                        symbol, mark_price, evt_sl_price, roe_pct,
                        state.last_evt_xi or 0.0, state.last_evt_sigma or 0.0,
                    )
                    await notifications.notify_alert(
                        "stop_loss", symbol,
                        f"EVT Stop {mark_price:.4f} ≤ {evt_sl_price:.4f} | ROE {roe_pct:.2f}%",
                    )
                    await self._close_long(symbol, contracts, "evt_stop_loss")
                    return
            except Exception as exc:
                logger.debug("EVT stop error %s: %s", symbol, exc)

        # Fallback a stop-loss legacy
        if roe_pct <= config.MAX_LOSS_ROE_PCT:
            logger.critical(
                "STOP LOSS %s | ROE: %.2f%% ≤ %.2f%% | "
                "Cerrando incondicionalmente.",
                symbol, roe_pct, config.MAX_LOSS_ROE_PCT,
            )
            await notifications.notify_alert(
                "stop_loss", symbol,
                f"ROE {roe_pct:.2f}% ≤ {config.MAX_LOSS_ROE_PCT:.2f}%",
            )
            await self._close_long(symbol, contracts, "stop_loss")
            return

        # ══════════════════════════════════════════════════════════
        # CAPA 1: Salidas universales
        # ══════════════════════════════════════════════════════════
        if roe_pct >= config.TAKE_PROFIT_ROE_PCT:
            logger.info("TP %s | ROE: %.2f%%", symbol, roe_pct)
            await self._close_long(symbol, contracts, "tp")
            return

        if funding_rate >= config.EXIT_FUNDING_RATE:
            logger.info(
                "FR EXIT %s | FR: %.4f%% >= 0%% | Tesis de cobro invertida.",
                symbol, funding_rate * 100,
            )
            await self._close_long(symbol, contracts, "funding_exit")
            return

        # ══════════════════════════════════════════════════════════
        # CAPA 2: Modo post-cobro (tesis agotada) O hold extension
        # ══════════════════════════════════════════════════════════
        if state.funding_collected:

            # HOLD EXTENSION: si el siguiente funding sigue muy negativo,
            # la tesis NO está agotada. Saltar a lógica pre-funding.
            if ft.next_funding_rate <= config.MAX_FUNDING_RATE:
                # Bonus de confianza por persistencia (Hurst)
                extension_msg = ""
                effective_timeout = config.POST_FUNDING_MAX_HOLD_MINUTES
                if (
                    config.ENABLE_HURST_FILTER
                    and state.last_hurst_fr is not None
                    and state.last_hurst_fr > config.HURST_PERSISTENCE_THRESHOLD
                ):
                    extension_bonus = (state.last_hurst_fr - config.HURST_PERSISTENCE_THRESHOLD) * 2.0
                    effective_timeout = int(
                        config.POST_FUNDING_MAX_HOLD_MINUTES * (1.0 + extension_bonus)
                    )
                    extension_msg = f" | H={state.last_hurst_fr:.3f} (ext. bonus {extension_bonus:.2f}x, timeout {effective_timeout}m)"

                logger.info(
                    "HOLD EXTENSION %s | Next FR: %.4f%% sigue negativo. "
                    "Manteniendo lógica pre-funding.%s",
                    symbol, ft.next_funding_rate * 100, extension_msg,
                )
                # Caer a CAPA 3 (pre-funding) sin pasar por post-cobro
                # Timeout efectivo se usa en la evaluación post-cobro más abajo
                pass  # continuar a CAPA 3
            else:
                # Modo post-cobro agresivo (tesis realmente agotada)

                # 2a. Blindfold: dump post-snapshot en curso
                if blindfold_active:
                    logger.info(
                        "BLINDFOLD %s | %.1f min post-snapshot (blindfold=%.1f min) | "
                        "ROE: %.2f%% | DCA BLOQUEADO",
                        symbol, ft.minutes_since_last, ft.blindfold_minutes, roe_pct,
                    )
                    return

                hold_minutes = (time.time() - state.funding_collected_ts) / 60

                # 2b. TP reducido
                if roe_pct >= config.POST_FUNDING_REDUCED_TP_ROE:
                    logger.info(
                        "TP POST-COBRO %s | ROE: %.2f%% >= %.2f%% | Hold: %.1f min",
                        symbol, roe_pct, config.POST_FUNDING_REDUCED_TP_ROE, hold_minutes,
                    )
                    await self._close_long(symbol, contracts, "tp_post_funding")
                    return

                # 2c. OI dump
                oi_change = self._oi_change_pct(symbol)
                if oi_change is not None and oi_change <= config.POST_FUNDING_OI_DUMP_THRESHOLD:
                    logger.info(
                        "OI DUMP %s | OI: %.2f%% ≤ %.2f%% | ROE: %.2f%%",
                        symbol, oi_change * 100,
                        config.POST_FUNDING_OI_DUMP_THRESHOLD * 100, roe_pct,
                    )
                    await self._close_long(symbol, contracts, "oi_dump_exit")
                    return

                # 2d. Timeout
                if hold_minutes >= config.POST_FUNDING_MAX_HOLD_MINUTES:
                    logger.info(
                        "TIMEOUT %s | %.1f min >= %d min | ROE: %.2f%%",
                        symbol, hold_minutes, config.POST_FUNDING_MAX_HOLD_MINUTES, roe_pct,
                    )
                    await self._close_long(symbol, contracts, "timeout_exit")
                    return

                # 2e. Esperar
                logger.debug(
                    "%s: post-cobro %.1f min, ROE: %.2f%%. Esperando salida.",
                    symbol, hold_minutes, roe_pct,
                )
                return

        # ══════════════════════════════════════════════════════════
        # CAPA 3: DCA pre-funding (tesis activa)
        # ══════════════════════════════════════════════════════════

        # Blindfold sin cobro confirmado = posición abierta justo antes
        if blindfold_active:
            logger.debug(
                "%s: blindfold pre-cobro (%.1f min). DCA bloqueado.",
                symbol, ft.minutes_since_last,
            )
            return

        # Evaluar caída de precio desde última entrada
        ref_price = state.last_entry_price if state.last_entry_price > 0 else entry_price
        drop_pct = (ref_price - mark_price) / ref_price if ref_price > 0 else 0.0

        if drop_pct < config.DCA_MIN_DROP_PCT or drop_pct > config.DCA_MAX_DROP_PCT:
            return

        # Clasificar OI
        oi_trend = self._classify_oi(symbol)
        if oi_trend is None:
            logger.debug("%s: sin datos OI suficientes para DCA.", symbol)
            return

        # Detectar funding empeorando
        funding_worsening = False
        if state.funding_prev is not None:
            funding_worsening = funding_rate < state.funding_prev
        state.funding_prev = funding_rate

        # ── REGLA 3: Squeeze ──────────────────────────────────────
        # Solo ejecutar si OI crowding NO está activo
        if funding_worsening and oi_trend == "rising_strong":
            if self._is_oi_crowded(symbol):
                logger.info(
                    "SQUEEZE ABORTADO %s | OI crowding detectado (bot saturation).",
                    symbol,
                )
                return
            logger.info(
                "R3 SQUEEZE %s | Drop: %.3f%% | OI: %s | FR empeora",
                symbol, drop_pct * 100, oi_trend,
            )
            await self._open_long(symbol, config.SQUEEZE_MARGIN, mark_price, "squeeze", ft)
            return

        # ── REGLA 1: Defensa ──────────────────────────────────────
        if oi_trend == "falling":
            logger.info(
                "R1 DEFENSE %s | Drop: %.3f%% | OI: %s",
                symbol, drop_pct * 100, oi_trend,
            )
            await self._open_long(symbol, config.DEFENSE_MARGIN, mark_price, "defense", ft)
            return

        # ── REGLA 2: DCA Estándar ─────────────────────────────────
        if oi_trend == "lateral":
            logger.info(
                "R2 DCA %s | Drop: %.3f%% | OI: %s",
                symbol, drop_pct * 100, oi_trend,
            )
            await self._open_long(symbol, config.STANDARD_MARGIN, mark_price, "standard", ft)
            return

    # ── Limpieza de memoria ───────────────────────────────────────

    def _cleanup_stale_states(self) -> None:
        """
        Purga estados vacíos para evitar memory leak en operación 24/7.
        El scanner crea un CoinState por cada símbolo que toca.
        Sin esta limpieza, tras días de operación el dict acumula
        cientos de estados con deques de OI vacíos.
        """
        now = time.time()
        stale_cutoff = 3600  # 1 hora sin actividad
        to_delete = [
            sym for sym, state in self.states.items()
            if state.total_margin_used == 0.0
            and state.last_order_ts < now - stale_cutoff
            and len(state.oi_history) == 0
            and len(state.fr_history) < 10  # preservar si acumuló histórico útil para math gates
        ]
        for sym in to_delete:
            del self.states[sym]
        if to_delete:
            logger.debug("Cleanup: %d estados vacíos purgados.", len(to_delete))

    # ── Ciclo principal ───────────────────────────────────────────

    async def run_cycle(self) -> None:
        """
        Un ciclo completo:
          scan paralelo → posiciones → OI → evaluación/entrada

        El scan devuelve monedas para ENTRADA INICIAL (dentro de ventana).
        Posiciones abiertas se evalúan SIEMPRE, sin restricción temporal.
        """
        # Circuit breaker: si está abierto, no operar
        if self._is_circuit_open():
            return

        # Daily drawdown breaker
        if await self._check_drawdown():
            return

        await self._refresh_learning()

        try:
            # 1. Scan paralelo para entradas iniciales
            qualified_pairs = await self.scan()
            qualified_map: dict[str, FundingTiming] = {
                sym: ft for sym, ft in qualified_pairs
            }
            self._record_success()

        except Exception as exc:
            logger.error("Error en scan: %s", exc)
            self._record_error()
            return

        try:
            # 2. Posiciones abiertas
            positions_raw: list[dict] = await _retry(
                lambda: self.exchange.fetch_positions(),
                "fetch_positions",
            )
            await asyncio.sleep(config.RATE_LIMIT_PAUSE)
            self._record_success()

        except Exception as exc:
            logger.error("Error en fetch_positions: %s", exc)
            self._record_error()
            return

        open_positions: dict[str, dict] = {}
        for pos in positions_raw:
            sym = pos.get("symbol")
            side = pos.get("side")
            contracts = abs(float(pos.get("contracts") or 0))
            if sym and side == "long" and contracts > 0:
                open_positions[sym] = pos

        # Unión: calificadas + posiciones abiertas
        symbols_to_process = set(qualified_map.keys()) | set(open_positions.keys())

        for symbol in symbols_to_process:
            if not self._running:
                break

            state = self.states[symbol]

            # 3. Capturar OI
            await self._record_oi(symbol)

            # ── Calcular métricas matemáticas avanzadas ──────────────────
            math_gates_passed = True
            gates_blocked_by = None
            tail_info: dict[str, float] = {}
            var_95 = es_99 = xi = sigma = None
            mc_result: dict[str, Any] = {}
            try:
                # 1. Entropía
                if len(state.fr_history) >= 10:
                    state.last_entropy_fr = math_engine.InformationMetrics.shannon_entropy_normalized(
                        np.array(state.fr_history, dtype=float), bins="fd"
                    )

                # 2. Tail risk + EVT + Vol
                if len(state.log_return_history) >= 20:
                    tail_info = math_engine.PowerLawTail.tail_risk_index(
                        np.array(state.log_return_history, dtype=float)
                    )
                    state.last_tail_alpha = tail_info.get("alpha")
                    state.last_tail_risk = tail_info.get("risk")

                    losses = -np.array(state.log_return_history, dtype=float)
                    losses = losses[losses > 0]
                    if len(losses) >= 5:
                        var_95, es_99, xi, sigma = math_engine.ExtremeValueEngine.evt_var_es(
                            losses, confidence=0.99, threshold_pct=config.EVT_THRESHOLD_PCT
                        )
                        state.last_evt_xi = xi
                        state.last_evt_sigma = sigma

                    regime, rv, pct = math_engine.LogVolatilityEngine.volatility_regime(
                        np.array(state.log_return_history, dtype=float),
                        extreme_pct=config.LOG_VOL_EXTREME_PERCENTILE,
                        calm_pct=config.LOG_VOL_CALM_PERCENTILE,
                    )
                    state.last_vol_regime = regime
                    state.last_rv_annualized = math_engine.LogVolatilityEngine.realized_volatility_annualized(
                        np.array(state.log_return_history, dtype=float)
                    )

                # 3. Hurst
                if len(state.fr_history) >= 20:
                    state.last_hurst_fr = math_engine.PersistenceAnalyzer.hurst_dfa(
                        np.array(state.fr_history, dtype=float)
                    )

                # 4. Kalman filter (suavización + predicción de FR)
                if len(state.fr_history) >= 5:
                    if state.kalman_filter is None:
                        # Estimar Q y R de los datos iniciales
                        noise_est = kalman_engine.KalmanFundingFilter.estimate_noise_parameters(
                            np.array(state.fr_history, dtype=float)
                        )
                        state.kalman_filter = kalman_engine.KalmanFundingFilter(
                            process_variance=noise_est["Q"],
                            measurement_variance=noise_est["R"],
                            initial_estimate=float(state.fr_history[-1]),
                        )
                    k_res = state.kalman_filter.update(float(state.fr_history[-1]))
                    state.last_kalman_filtered_fr = k_res["filtered"]
                    state.last_kalman_predicted_fr = state.kalman_filter.predict(steps=1)

                # 5. Regime-switching
                if len(state.fr_history) >= 15:
                    r_res = regime_engine.FundingRegimeModel.classify(
                        np.array(state.fr_history, dtype=float),
                        hurst=state.last_hurst_fr,
                        vol_regime=state.last_vol_regime,
                    )
                    state.last_fr_regime = r_res["regime"]
                    state.last_regime_confidence = r_res["confidence"]

                # 6. Transfer entropy OI → FR
                if len(state.oi_change_history) >= 10 and len(state.fr_history) >= 10:
                    min_len = min(len(state.oi_change_history), len(state.fr_history))
                    oi_aligned = np.array(state.oi_change_history)[-min_len:]
                    fr_aligned = np.array(state.fr_history)[-min_len:]
                    state.last_te_oi_to_fr = math_engine.InformationMetrics.transfer_entropy_knn(
                        oi_aligned, fr_aligned, lag=1
                    )

                # 7. Monte Carlo sizing (cada 5 ciclos para no saturar CPU)
                if (
                    len(state.log_return_history) >= 50
                    and (len(state.fr_history) % 5 == 0)
                ):
                    mc_engine = monte_carlo_engine.MonteCarloRiskEngine(n_paths=5_000)
                    mc_result = mc_engine.recommend_position_size(
                        log_returns=np.array(state.log_return_history, dtype=float),
                        capital=config.MAX_TOTAL_MARGIN,
                        funding_rate=state.fr_history[-1] if state.fr_history else 0.0,
                        leverage=config.LEVERAGE,
                    )
                    state.last_mc_optimal_size = mc_result.get("recommended_usdt", 0.0)
                    state.last_mc_prob_ruin = mc_result.get("prob_ruin", 1.0)
                    await db_local.log_monte_carlo({"symbol": symbol, **mc_result})

                # 8. Loguear snapshot matemático
                await db_local.log_math_snapshot(
                    symbol=symbol,
                    tail_alpha=state.last_tail_alpha,
                    tail_pvalue=state.last_tail_pvalue,
                    tail_risk_index=state.last_tail_risk,
                    hill_xi=tail_info.get("hill_xi"),
                    evt_var_95=var_95,
                    evt_var_99=None,
                    evt_es_99=es_99,
                    evt_xi=state.last_evt_xi,
                    evt_sigma=state.last_evt_sigma,
                    evt_stop_price=state.last_evt_stop_price,
                    hurst_fr=state.last_hurst_fr,
                    hurst_method=config.HURST_CALCULATION_METHOD,
                    fr_regime=state.last_fr_regime,
                    shannon_entropy_fr=state.last_entropy_fr,
                    te_oi_to_fr=state.last_te_oi_to_fr,
                    te_fr_to_oi=state.last_te_fr_to_oi,
                    mi_fr_oi=None,
                    log_rv_annualized=state.last_rv_annualized,
                    log_vol_regime=state.last_vol_regime,
                    log_atr_14=None,
                    mf_width_delta_alpha=state.last_multifractal_width,
                    power_score=state.last_power_score,
                    math_gates_passed=math_gates_passed,
                    gates_blocked_by=gates_blocked_by,
                )
            except Exception as exc:
                logger.debug("Math snapshot error %s: %s", symbol, exc)

            # 4. Funding timing (reusar del scan si disponible)
            ft = qualified_map.get(symbol)
            if ft is None:
                ft = FundingTiming()
                try:
                    fr_data = await _retry(
                        lambda s=symbol: self.exchange.fetch_funding_rate(s),
                        f"fetch_fr({symbol})",
                    )
                    ft = self._parse_funding_timing(fr_data, symbol)
                    await asyncio.sleep(config.RATE_LIMIT_PAUSE)
                except Exception as exc:
                    logger.debug("FR fetch fail %s: %s", symbol, exc)

            # ── Actualizar históricos matemáticos para posiciones abiertas ─
            if ft.scan_price <= 0 and symbol in open_positions:
                ft.scan_price = float(open_positions[symbol].get("markPrice") or 0.0)
            if symbol not in qualified_map and (ft.funding_rate != 0.0 or ft.scan_price != 0.0):
                state.fr_history.append(ft.funding_rate)
                prev_price = state.price_history[-1] if state.price_history else 0.0
                if prev_price > 0 and ft.scan_price > 0:
                    state.log_return_history.append(math.log(ft.scan_price / prev_price))
                if ft.scan_price > 0:
                    state.price_history.append(ft.scan_price)

            # 5. Posición abierta → evaluar siempre
            if symbol in open_positions:
                try:
                    await self._evaluate_position(symbol, open_positions[symbol], ft)
                    self._record_success()
                except Exception as exc:
                    logger.error("Error evaluando %s: %s", symbol, exc)
                    self._record_error()
                continue

            # 6. Sin posición + califica → entrada inicial con price drift gate
            if symbol in qualified_map:
                try:
                    ticker = await _retry(
                        lambda s=symbol: self.exchange.fetch_ticker(s),
                        f"fetch_ticker({symbol})",
                    )
                    await asyncio.sleep(config.RATE_LIMIT_PAUSE)

                    price = float(ticker.get("last") or ticker.get("close") or 0)
                    if price <= 0:
                        continue

                    state = self.states[symbol]
                    scan_price = state.scan_price

                    # Price drift gate: precio no puede haberse movido mucho
                    # desde que el scanner lo calificó
                    if scan_price > 0:
                        drift = abs(price - scan_price) / scan_price
                        if drift > config.PRICE_DRIFT_MAX_PCT:
                            logger.info(
                                "DRIFT GATE %s | Drift: %.4f%% > %.4f%% | Entrada cancelada.",
                                symbol, drift * 100, config.PRICE_DRIFT_MAX_PCT * 100,
                            )
                            continue
                    else:
                        logger.warning("DRIFT GATE %s | Sin scan_price disponible. Usando precio actual.", symbol)

                    scan_ft = qualified_map[symbol]

                    logger.info(
                        "ENTRADA %s | FR: %.4f%% | Ciclo: %.0fh | "
                        "@%.6f | Snapshot: %.1f min",
                        symbol, scan_ft.funding_rate * 100,
                        scan_ft.interval_hours, price,
                        scan_ft.minutes_to_next,
                    )
                    await self._open_long(
                        symbol, config.INITIAL_ENTRY_MARGIN, price, "initial", scan_ft
                    )
                    self._record_success()

                except Exception as exc:
                    logger.error("Error en entrada %s: %s", symbol, exc)
                    self._record_error()

        # Heartbeat
        now = time.time()
        if now - self._last_heartbeat_ts >= config.HEARTBEAT_INTERVAL_SECONDS:
            self._last_heartbeat_ts = now
            total_margin = self._total_margin_used()
            open_count = self._open_position_count()
            best_fr_info = (
                f" | Mejor FR ({self._best_fr_symbol}): {self._best_fr_seen*100:.4f}%"
                if self._best_fr_symbol
                else " | Sin FRs negativos en este periodo"
            )
            logger.info(
                "HEARTBEAT | Posiciones: %d | Margen: %.2f/%.2f USDT | "
                "Ciclos scan: %d | Circuit: %s | DD: %s%s",
                open_count, total_margin, config.MAX_TOTAL_MARGIN,
                self._scan_cycles,
                "OPEN" if self._circuit_open else "ok",
                "ACTIVO" if self._daily_drawdown_triggered else "ok",
                best_fr_info,
            )
            # Reset stats de periodo
            self._scan_cycles = 0
            self._best_fr_seen = 0.0
            self._best_fr_symbol = ""

        # Limpiar estados vacíos
        self._cleanup_stale_states()

    def status_snapshot(self) -> dict[str, Any]:
        learning_state = "off"
        if self.learning is not None:
            learning_state = f"on ({len(self.learning.stats)} simbolos)"
        return {
            "running": self._running,
            "dry_run": config.DRY_RUN,
            "open_positions": self._open_position_count(),
            "total_margin": self._total_margin_used(),
            "max_total_margin": config.MAX_TOTAL_MARGIN,
            "circuit": "OPEN" if self._circuit_open else "closed",
            "drawdown": "TRIGGERED" if self._daily_drawdown_triggered else "ok",
            "consecutive_errors": self._consecutive_errors,
            "learning": learning_state,
        }

    @property
    def is_running(self) -> bool:
        return self._running
