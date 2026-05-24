"""
main.py — Punto de entrada del KuCoin Funding Fee Scalper v4.

Responsabilidades:
    - Configurar logging (stdout + archivo rotativo).
    - Manejar señales SIGINT/SIGTERM para shutdown graceful.
    - Ejecutar el loop con intervalos controlados.
    - Reportar estado del circuit breaker y drawdown en cada ciclo.
    - Opcionalmente cerrar posiciones al shutdown (configurable).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from logging.handlers import RotatingFileHandler

import config
from telegram_tracker import TelegramTracker
from trading_engine import TradingEngine

# ── Logging ───────────────────────────────────────────────────────
def _setup_logging() -> None:
    fmt = logging.Formatter(
        "%(asctime)s | %(name)-8s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Archivo rotativo: 10 MB × 5 archivos
    fh = RotatingFileHandler(
        "bot.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


_setup_logging()
logger = logging.getLogger("main")


# ── Loop principal ────────────────────────────────────────────────

async def main() -> None:
    engine = TradingEngine()
    telegram = TelegramTracker(engine)
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _handle_signal(sig: signal.Signals) -> None:
        logger.info("Señal %s recibida. Iniciando shutdown graceful…", sig.name)
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal, sig)
        except NotImplementedError:
            signal.signal(sig, lambda _signum, _frame, s=sig: _handle_signal(s))

    try:
        await engine.start()
        await telegram.start()
        logger.info("═" * 60)
        logger.info("KuCoin Funding Fee Scalper — ACTIVO")
        if config.DRY_RUN:
            logger.info("⚠️  MODO DRY-RUN ACTIVO — No se enviarán órdenes reales")
        logger.info("Exchange:      kucoinfutures (USD-M Perpetual Swap)")
        logger.info("Leverage:      %dx | Margin: %s", config.LEVERAGE, config.MARGIN_MODE)
        logger.info("Max/coin:      %.0f USDT", config.MAX_MARGIN_PER_COIN)
        logger.info("Global margin: %.0f USDT", config.MAX_TOTAL_MARGIN)
        logger.info("Max posiciones: %d", config.MAX_OPEN_POSITIONS)
        logger.info("Vol. mínimo:   %.0f USDT", config.MIN_VOLUME_24H)
        logger.info("FR umbral:     ≤ %.2f%%", config.MAX_FUNDING_RATE * 100)
        logger.info("Stop-loss:     ROE ≤ %.1f%%", config.MAX_LOSS_ROE_PCT)
        logger.info("Drawdown:      ≤ %.1f%% diario", config.MAX_DAILY_DRAWDOWN_PCT)
        logger.info("Taker fee:     0.06%% por lado (0.12%% round-trip)")
        logger.info(
            "Ventana:       %.0f%% del ciclo (%.0f–%.0f min)",
            config.ENTRY_WINDOW_FRACTION * 100,
            config.ENTRY_WINDOW_MINUTES_MIN,
            config.ENTRY_WINDOW_MINUTES_MAX,
        )
        logger.info(
            "Shutdown:      %s",
            "cerrar posiciones" if config.SHUTDOWN_CLOSE_POSITIONS else "mantener posiciones",
        )
        if config.WEBHOOK_URL:
            logger.info("Webhook:       %s", config.WEBHOOK_URL[:40] + "…")
        if telegram.enabled:
            logger.info("Telegram:      seguimiento activo")
        if config.ENABLE_AUTO_LEARNING:
            logger.info("Autolearn:     activo | min trades: %d", config.AUTO_LEARNING_MIN_TRADES)
        logger.info("═" * 60)

        cycle = 0
        while not shutdown_event.is_set():
            cycle += 1
            logger.info("── Ciclo %d ──", cycle)

            try:
                await engine.run_cycle()
            except Exception as exc:
                logger.exception("Error no capturado en ciclo %d: %s", cycle, exc)

            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=config.SCAN_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                pass  # Normal → siguiente ciclo

    except Exception as exc:
        logger.exception("Error fatal en inicialización: %s", exc)
    finally:
        # Graceful shutdown configurable
        if config.SHUTDOWN_CLOSE_POSITIONS and not config.DRY_RUN:
            logger.info("Shutdown: cerrando posiciones abiertas…")
            try:
                positions_raw = await engine.exchange.fetch_positions()
                for pos in positions_raw:
                    sym = pos.get("symbol")
                    side = pos.get("side")
                    contracts = abs(float(pos.get("contracts") or 0))
                    if sym and side == "long" and contracts > 0:
                        logger.info("Shutdown close %s | %.6f contratos", sym, contracts)
                        await engine._close_long(sym, contracts, "shutdown")
            except Exception as exc:
                logger.error("Error cerrando posiciones en shutdown: %s", exc)

        await telegram.stop()
        await engine.stop()
        logger.info("Bot detenido limpiamente. Adiós.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
