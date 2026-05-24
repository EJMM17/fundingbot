"""
config.py — Configuración centralizada para KuCoin Funding Fee Scalper.

Todas las credenciales se cargan desde .env vía python-dotenv.
Ningún secreto toca este archivo.

─────────────────────────────────────────────────────────────────
MODELO ECONÓMICO (por qué estos números):

KuCoin Futures USD-M:
  Taker fee:  0.06% por lado  → trade completo (buy + sell) = 0.12%
  Ciclos:     BTC/ETH = 8h | mayoría altcoins = 4h | algunos = 1h
  FR cap/floor: dinámico por par, típicamente ±0.375% para contratos 5x

Umbral mínimo de entrada (MAX_FUNDING_RATE = -0.20%):
  FR cobrado:        ≥ +0.20% del notional por ciclo
  Fee round-trip:      0.12% del notional
  Net mínimo esperado: 0.08% por ciclo → positivo incluso con slippage

  NOTA: -0.20% es MUCHO más estricto que MEXC (-1.0%).
  En KuCoin los FR negativos profundos son menos frecuentes pero
  el floor del exchange protege contra sorpresas extremas.
  Si quieres más oportunidades: ajusta a -0.15%.
  Si quieres más seguridad: deja en -0.20%.

Stop-loss absoluto (MAX_LOSS_ROE_PCT = -20.0%):
  Con 5x leverage, -20% ROE = -4% movimiento adverse en precio.
  Protege contra black swans que quiebren la tesis.
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Cargar .env desde la raíz del proyecto ────────────────────────
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)


def _require_env(key: str) -> str:
    """Aborta si falta una variable de entorno obligatoria."""
    val = os.getenv(key)
    if not val:
        sys.exit(f"[FATAL] Variable de entorno '{key}' no definida. Revisa .env")
    return val


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        sys.exit(f"[FATAL] Variable de entorno '{key}' debe ser un entero.")


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        sys.exit(f"[FATAL] Variable de entorno '{key}' debe ser numerica.")


# ── Credenciales KuCoin (nunca hardcodeadas) ──────────────────────
KUCOIN_API_KEY: str = _require_env("KUCOIN_API_KEY")
KUCOIN_SECRET: str = _require_env("KUCOIN_SECRET")
KUCOIN_PASSPHRASE: str = _require_env("KUCOIN_PASSPHRASE")  # Obligatorio en KuCoin

# ── Exchange ──────────────────────────────────────────────────────
# KuCoin Futures es una clase SEPARADA en ccxt: 'kucoinfutures'
# NO es ccxt.kucoin con defaultType='swap'. Son clases distintas.
EXCHANGE_ID: str = "kucoinfutures"
LEVERAGE: int = _env_int("LEVERAGE", 5)
MARGIN_MODE: str = os.getenv("MARGIN_MODE", "cross")

# ── Escáner ───────────────────────────────────────────────────────
# Piso de volumen 24h en USDT. Permisivo a propósito (5M) para operar
# muchas más altcoins: la liquidez REAL ya no se controla aquí sino vía
# el spread gate (MAX_SPREAD_PCT) y el score de liquidez (VOLUME_LIQ_REFERENCE).
MIN_VOLUME_24H: float = 5_000_000.0         # USDT

# Volumen de referencia al que liq_score satura en 1.0. Desacoplado del
# piso de entrada: aunque MIN_VOLUME_24H baje, el ranking sigue premiando
# coins gruesas sobre finas (antes estaba atado a MIN_VOLUME_24H * 5).
VOLUME_LIQ_REFERENCE: float = 50_000_000.0  # USDT

# Umbral de FR para entrada.
# Lógica económica: fee round-trip KuCoin = 0.12% notional.
# Relajado a -0.15% para ganar flexibilidad en alts de bajo volumen.
# net_fr_after_fees resta el 0.06% de cierre → 0.15% - 0.06% = 0.09% neto > 0.
# Margen sobre fees deliberadamente más ajustado (el spread gate cubre el resto).
MAX_FUNDING_RATE: float = -0.0015           # ≤ -0.15%

# ── Límites de riesgo ─────────────────────────────────────────────
# Margen por moneda bajado a 100: con coins de bajo volumen (más riesgo de
# cola) conviene menos concentración por símbolo y más diversificación.
MAX_MARGIN_PER_COIN: float = _env_float("MAX_MARGIN_PER_COIN", 100.0)   # USDT máximo por moneda
MAX_TOTAL_MARGIN: float = _env_float("MAX_TOTAL_MARGIN", 500.0)         # USDT máximo en TODAS las posiciones (capital total acotado)
MAX_OPEN_POSITIONS: int = _env_int("MAX_OPEN_POSITIONS", 8)             # Máximo de posiciones simultáneas (más candidatos al bajar el piso)
COOLDOWN_SECONDS: int = 60                  # Entre órdenes de la misma moneda
INITIAL_ENTRY_MARGIN: float = 5.0           # Primera entrada en USDT

# ── Stop-loss absoluto (ausente en v2, crítico para producción) ────
# Si ROE cae a este nivel → cerrar incondicionalmente.
# Con 5x leverage: -20% ROE = -4% movimiento en precio.
MAX_LOSS_ROE_PCT: float = -20.0             # ROE ≤ -20% → stop loss

# ── DCA — Umbrales de caída de precio ─────────────────────────────
# Zona de DCA: caída entre 0.25% y 0.50% desde última entrada.
# Fuera de esta zona: ni muy poco (ruido) ni demasiado (tendencia bajista).
DCA_MIN_DROP_PCT: float = 0.0025            # 0.25%
DCA_MAX_DROP_PCT: float = 0.0050            # 0.50%

# ── DCA — Margen por regla (USDT) ─────────────────────────────────
DEFENSE_MARGIN: float = 5.0                 # Regla 1: OI cae
STANDARD_MARGIN: float = 10.0              # Regla 2: OI lateral
SQUEEZE_MARGIN: float = 50.0               # Regla 3: squeeze genuino

# ── OI — Detección de tendencia ───────────────────────────────────
OI_WINDOW_SECONDS: int = 300                # Ventana de comparación: 5 min
OI_FALLING_THRESHOLD: float = -0.02        # < -2% → cayendo
OI_LATERAL_LOW: float = -0.02              # ≥ -2%
OI_LATERAL_HIGH: float = 0.02             # ≤ +2% → lateral
OI_RISING_STRONG_THRESHOLD: float = 0.05  # > +5% → subida fuerte

# ── OI Crowding — Detección de saturación de bots ─────────────────
# Si demasiados bots entran al mismo tiempo, el OI sube de forma
# anómala. Un z-score > umbral indica saturación → abortar entrada.
OI_CROWDING_ZSCORE_THRESHOLD: float = 2.5  # σ sobre la media histórica
OI_CROWDING_WINDOW: int = 20               # N snapshots para calcular σ

# ── Ventana temporal de entrada (Time-Targeting) ──────────────────
# CRÍTICO: KuCoin tiene ciclos de 8h (BTC/ETH) y 4h (altcoins).
# La ventana se escala dinámicamente en el engine según el ciclo.
# Este valor es la FRACCIÓN del ciclo que se permite como ventana.
# 0.125 = 12.5% del ciclo:
#   → ciclo 8h: ventana = 60 min
#   → ciclo 4h: ventana = 30 min
#   → ciclo 1h: ventana = 7.5 min
ENTRY_WINDOW_FRACTION: float = 0.125       # 12.5% del ciclo de funding
ENTRY_WINDOW_MINUTES_MAX: int = 60         # Tope absoluto en minutos
ENTRY_WINDOW_MINUTES_MIN: int = 5          # Mínimo absoluto en minutos

# ── Price drift gate (entrada inicial) ────────────────────────────
# Máximo desvío aceptable entre el precio del scan y el precio
# de ejecución. Protege contra entradas en precios stale.
# Relajado a 0.40%: las alts finas se mueven más entre scan y ejecución.
PRICE_DRIFT_MAX_PCT: float = 0.0040        # 0.40%

# ── Slippage gate (ejecución de órdenes) ──────────────────────────
# Máximo desvío aceptable entre el precio esperado y el fill real.
# Subido a 0.70%: coins de bajo volumen rellenan con más slippage;
# se mantiene acotado para no comerse el edge del funding.
MAX_SLIPPAGE_PCT: float = 0.0070           # 0.70%

# ── Liquidity gate (spread) ───────────────────────────────────────
# Máximo spread bid-ask aceptable al momento de la entrada.
# Subido a 0.15%: con el piso de volumen en 5M, ESTE es el verdadero
# filtro de liquidez. Trade-off: spread mayor = más coste de slippage
# en market orders, por eso no se relaja más allá de 0.15%.
MAX_SPREAD_PCT: float = 0.0015             # 0.15%

# ── Blindfold post-funding (anti-dump) ────────────────────────────
# Después del snapshot, scalpers cierran masivamente causando un dump.
# Durante este periodo: DCA DESHABILITADO, solo evaluación de salida.
# Se escala igual que ENTRY_WINDOW: fracción del ciclo.
POST_FUNDING_BLINDFOLD_FRACTION: float = 0.0625   # 6.25% del ciclo
POST_FUNDING_BLINDFOLD_MINUTES_MAX: int = 10       # Tope absoluto
POST_FUNDING_BLINDFOLD_MINUTES_MIN: int = 3        # Mínimo absoluto

# ── Salida inteligente post-cobro ─────────────────────────────────
# Una vez cobrado el funding, la tesis está AGOTADA.
# Ajustado para cubrir el taker fee de cierre (0.06%):
# TP reducido en ROE términos, no en precio.
POST_FUNDING_REDUCED_TP_ROE: float = 3.0   # ROE ≥ 3% post-cobro
                                            # (vs 2% en MEXC, sube por fees)
POST_FUNDING_OI_DUMP_THRESHOLD: float = -0.08  # OI cae >8% → salir
POST_FUNDING_MAX_HOLD_MINUTES: int = 30    # Timeout post-cobro

# ── Salida universal ──────────────────────────────────────────────
TAKE_PROFIT_ROE_PCT: float = 15.0          # ROE ≥ 15% → cerrar
EXIT_FUNDING_RATE: float = 0.0             # FR ≥ 0% → cerrar

# ── Scoring y ranking de oportunidades ────────────────────────────
# El bot evalúa TODOS los pares que califican y solo entra en las
# TOP_N mejores según score ponderado.
TOP_N_OPPORTUNITIES: int = 4               # Máximo de entradas nuevas por ciclo
SCORE_WEIGHT_FR: float = 0.6               # Peso del funding rate anualizado
SCORE_WEIGHT_LIQ: float = 0.3              # Peso de la liquidez (volumen normalizado)
SCORE_WEIGHT_OI: float = 0.1               # Peso de la tendencia de OI

# ── Predicted funding sanity check ────────────────────────────────
# Si True, el bot valida que nextFundingRate también sea negativo
# antes de entrar. Evita entrar cuando el funding ya giró positivo.
USE_PREDICTED_FUNDING_CHECK: bool = True

# ── Adaptive position sizing ──────────────────────────────────────
# Si True, escala el margen DCA proporcional a la calidad del edge:
# más margen para FR más negativos / ciclos más cortos.
ADAPTIVE_SIZING: bool = True
ADAPTIVE_SIZING_MAX_MULTIPLIER: float = 2.0  # Tope de multiplicador sobre el base

# ── Daily drawdown circuit breaker ────────────────────────────────
# Si el P&L total cae más de este % desde el balance inicial del día,
# el bot pausa operaciones hasta el día siguiente.
MAX_DAILY_DRAWDOWN_PCT: float = -10.0      # -10% del capital

# ── Timing del loop ───────────────────────────────────────────────
SCAN_INTERVAL_SECONDS: int = 30
RATE_LIMIT_PAUSE: float = 0.20             # KuCoin permite más throughput
MAX_RETRIES: int = 5
BACKOFF_BASE: float = 1.0

# ── Circuit breaker ───────────────────────────────────────────────
# Si ocurren demasiados errores consecutivos → pausa automática.
CIRCUIT_BREAKER_MAX_ERRORS: int = 10       # Errores consecutivos para disparar
CIRCUIT_BREAKER_PAUSE_SECONDS: int = 300   # 5 min de pausa antes de reintentar

# ── Dry-run / Paper trading ──────────────────────────────────────
# Si True, el bot ejecuta TODO el ciclo, loguea las órdenes que
# "enviaría" y guarda en DB, pero NUNCA envía órdenes reales.
DRY_RUN: bool = _env_bool("DRY_RUN", True)

# ── Graceful shutdown ─────────────────────────────────────────────
# Si True, al recibir SIGINT/SIGTERM cierra todas las posiciones.
# Si False, solo deja de hacer nuevas entradas y mantiene lo abierto.
SHUTDOWN_CLOSE_POSITIONS: bool = _env_bool("SHUTDOWN_CLOSE_POSITIONS", False)

# ── Webhook notifications ─────────────────────────────────────────
# URL opcional para notificaciones (Discord/Slack/Telegram).
# Dejar en blanco ("" ) para desactivar.
WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")

# Telegram tracking
# Activo si hay token + chat_id, salvo que TELEGRAM_ENABLED=false.
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED: bool = _env_bool(
    "TELEGRAM_ENABLED",
    bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
)
TELEGRAM_COMMANDS_ENABLED: bool = _env_bool("TELEGRAM_COMMANDS_ENABLED", True)
TELEGRAM_POLL_SECONDS: int = _env_int("TELEGRAM_POLL_SECONDS", 5)
TELEGRAM_REQUEST_TIMEOUT_SECONDS: int = _env_int("TELEGRAM_REQUEST_TIMEOUT_SECONDS", 15)
TELEGRAM_STATUS_INTERVAL_SECONDS: int = _env_int("TELEGRAM_STATUS_INTERVAL_SECONDS", 300)

# Autoaprendizaje conservador
ENABLE_AUTO_LEARNING: bool = _env_bool("ENABLE_AUTO_LEARNING", True)
AUTO_LEARNING_REFRESH_SECONDS: int = _env_int("AUTO_LEARNING_REFRESH_SECONDS", 900)
AUTO_LEARNING_MIN_TRADES: int = _env_int("AUTO_LEARNING_MIN_TRADES", 5)
AUTO_LEARNING_MAX_BOOST: float = _env_float("AUTO_LEARNING_MAX_BOOST", 0.15)
AUTO_LEARNING_MAX_PENALTY: float = _env_float("AUTO_LEARNING_MAX_PENALTY", 0.40)

# ── Arsenal Matemático ──────────────────────────────────────────────
# Ventana de históricos subida a 150: estimadores de cola (CSN), Hurst-DFA
# y multifractal (MF-DFA) son más estables con más muestras y se mantiene
# el mínimo ≥64 del multifractal con margen.
MATH_HISTORY_WINDOW: int = 150          # Ventana para históricos

# EVT (Extreme Value Theory)
ENABLE_EVT_STOPS: bool = True           # Usar stops basados en GPD
EVT_CONFIDENCE_SL: float = 0.95         # Confianza para stop-loss EVT
EVT_CONFIDENCE_VAR: float = 0.99        # Confianza para VaR/ES
EVT_THRESHOLD_PCT: float = 0.90         # Percentil para umbral POT (90% = top 10%)

# Leyes de Potencia / Tail Risk
ENABLE_TAIL_GATES: bool = True          # Gate de cola pesada
TAIL_ALPHA_MIN_ENTRY: float = 2.5       # α mínima para entrar (α < 2.5 = cola catastrófica)
TAIL_ALPHA_WARNING: float = 3.0         # α de advertencia

# Hurst / Persistencia
ENABLE_HURST_FILTER: bool = True
HURST_PERSISTENCE_THRESHOLD: float = 0.58
HURST_MEANREVERSION_THRESHOLD: float = 0.42
HURST_CALCULATION_METHOD: str = "dfa"   # "rs" o "dfa" (DFA recomendado)

# Entropía / Teoría de la Información
ENABLE_ENTROPY_GATES: bool = True
ENTROPY_MAX_ENTRY: float = 0.75         # Entropía normalizada máxima para entrar
ENTROPY_MIN_BONUS: float = 0.40         # Entropía para bonus de predictibilidad
TE_MIN_FOR_OI_WEIGHT: float = 0.01      # Transfer entropy mínimo para confiar en OI

# Volatilidad Logarítmica
ENABLE_LOG_VOL_FILTER: bool = True
LOG_VOL_EXTREME_PERCENTILE: float = 0.90
LOG_VOL_CALM_PERCENTILE: float = 0.25

# Multifractalidad
ENABLE_MULTIFRACTAL_GATES: bool = True
MULTIFRACTAL_WIDTH_THRESHOLD: float = 0.50  # Δα umbral para penalización
MULTIFRACTAL_WIDTH_ABORT: float = 0.70      # Δα para abortar entrada

# Scoring
ENABLE_MATH_SCORING: bool = True        # Reemplazar scoring simple por power_score

# ── Heartbeat ─────────────────────────────────────────────────────
# Intervalo en segundos para loguear heartbeat (estado resumido).
HEARTBEAT_INTERVAL_SECONDS: int = 300      # 5 minutos
