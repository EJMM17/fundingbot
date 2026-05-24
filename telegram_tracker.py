"""
telegram_tracker.py - Seguimiento operativo por Telegram.

Comandos de solo lectura:
    /status      estado general
    /positions   posiciones abiertas
    /risk        limites y breakers
    /learn       resumen de autoaprendizaje
    /help        ayuda
"""

from __future__ import annotations

import asyncio
import html
import logging
from typing import Any

import aiohttp

import config
import notifications

logger = logging.getLogger("telegram")


class TelegramTracker:
    def __init__(self, engine) -> None:
        self.engine = engine
        self._session: aiohttp.ClientSession | None = None
        self._poll_task: asyncio.Task | None = None
        self._status_task: asyncio.Task | None = None
        self._offset: int | None = None
        self._running = False

    @property
    def enabled(self) -> bool:
        return bool(
            config.TELEGRAM_ENABLED
            and config.TELEGRAM_BOT_TOKEN
            and config.TELEGRAM_CHAT_ID
        )

    async def start(self) -> None:
        if not self.enabled:
            return
        self._running = True
        self._session = aiohttp.ClientSession()
        await self._send("Seguimiento Telegram activo. Usa /help para ver comandos.")

        if config.TELEGRAM_COMMANDS_ENABLED:
            self._poll_task = asyncio.create_task(self._poll_loop(), name="telegram-poll")
        if config.TELEGRAM_STATUS_INTERVAL_SECONDS > 0:
            self._status_task = asyncio.create_task(
                self._periodic_status_loop(),
                name="telegram-status",
            )
        logger.info("Telegram tracker activo.")

    async def stop(self) -> None:
        if not self.enabled:
            return
        self._running = False
        for task in (self._poll_task, self._status_task):
            if task:
                task.cancel()
        await asyncio.gather(
            *[task for task in (self._poll_task, self._status_task) if task],
            return_exceptions=True,
        )
        await self._send("Bot detenido limpiamente.")
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("Telegram tracker detenido.")

    async def _api(self, method: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self._session:
            return None
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/{method}"
        try:
            async with self._session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=config.TELEGRAM_REQUEST_TIMEOUT_SECONDS),
            ) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400 or not data.get("ok", False):
                    logger.warning("Telegram %s fallo: HTTP %s | %s", method, resp.status, data)
                    return None
                return data
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Telegram %s excepcion: %s", method, exc)
            return None

    async def _send(self, message: str) -> None:
        if self._session:
            await notifications.send_telegram_message(message, session=self._session)
        else:
            await notifications.send_telegram_message(message)

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                payload: dict[str, Any] = {
                    "timeout": max(1, config.TELEGRAM_POLL_SECONDS),
                    "allowed_updates": ["message"],
                }
                if self._offset is not None:
                    payload["offset"] = self._offset
                data = await self._api("getUpdates", payload)
                if data:
                    for update in data.get("result", []):
                        self._offset = int(update["update_id"]) + 1
                        await self._handle_update(update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Telegram poll error: %s", exc)
                await asyncio.sleep(config.TELEGRAM_POLL_SECONDS)

    async def _periodic_status_loop(self) -> None:
        while self._running:
            await asyncio.sleep(config.TELEGRAM_STATUS_INTERVAL_SECONDS)
            if self._running:
                await self._send(self._format_status(periodic=True))

    async def _handle_update(self, update: dict[str, Any]) -> None:
        msg = update.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        if chat_id != str(config.TELEGRAM_CHAT_ID):
            logger.warning("Telegram chat no autorizado: %s", chat_id)
            return

        text = str(msg.get("text") or "").strip()
        if not text:
            return
        command = text.split()[0].split("@")[0].lower()

        if command in {"/start", "/help"}:
            await self._send(self._format_help())
        elif command in {"/ping"}:
            await self._send("pong")
        elif command in {"/status", "/estado"}:
            await self._send(self._format_status())
        elif command in {"/positions", "/posiciones"}:
            await self._send(await self._format_positions())
        elif command in {"/risk", "/riesgo"}:
            await self._send(self._format_risk())
        elif command in {"/learn", "/aprendizaje"}:
            await self._send(self._format_learning())
        else:
            await self._send("Comando no reconocido. Usa /help.")

    def _format_help(self) -> str:
        return (
            "<b>Comandos</b>\n"
            "/status - estado general\n"
            "/positions - posiciones abiertas\n"
            "/risk - limites y breakers\n"
            "/learn - autoaprendizaje\n"
            "/ping - prueba de conexion"
        )

    def _format_status(self, periodic: bool = False) -> str:
        snap = self.engine.status_snapshot()
        title = "Resumen periodico" if periodic else "Estado"
        return (
            f"<b>{title}</b>\n"
            f"Modo: {'DRY-RUN' if snap['dry_run'] else 'REAL'}\n"
            f"Running: {snap['running']}\n"
            f"Posiciones: {snap['open_positions']}\n"
            f"Margen: {snap['total_margin']:.2f} / {snap['max_total_margin']:.2f} USDT\n"
            f"Circuit: {snap['circuit']}\n"
            f"Drawdown: {snap['drawdown']}\n"
            f"Errores seguidos: {snap['consecutive_errors']}\n"
            f"Autoaprendizaje: {snap['learning']}"
        )

    def _format_risk(self) -> str:
        snap = self.engine.status_snapshot()
        return (
            "<b>Riesgo</b>\n"
            f"Leverage: {config.LEVERAGE}x\n"
            f"Margin mode: {html.escape(config.MARGIN_MODE)}\n"
            f"Max/coin: {config.MAX_MARGIN_PER_COIN:.2f} USDT\n"
            f"Max global: {config.MAX_TOTAL_MARGIN:.2f} USDT\n"
            f"Max posiciones: {config.MAX_OPEN_POSITIONS}\n"
            f"Stop-loss ROE: {config.MAX_LOSS_ROE_PCT:.2f}%\n"
            f"Drawdown diario: {config.MAX_DAILY_DRAWDOWN_PCT:.2f}%\n"
            f"Circuit: {snap['circuit']}"
        )

    def _format_learning(self) -> str:
        learning = getattr(self.engine, "learning", None)
        if learning is None:
            return "Autoaprendizaje desactivado."
        return "<pre>" + html.escape(learning.render_summary()) + "</pre>"

    async def _format_positions(self) -> str:
        if not getattr(self.engine, "exchange", None):
            return "Exchange no inicializado."
        try:
            positions = await self.engine.exchange.fetch_positions()
        except Exception as exc:
            return f"No se pudieron leer posiciones: {html.escape(str(exc))}"

        lines = ["<b>Posiciones abiertas</b>"]
        count = 0
        for pos in positions:
            side = pos.get("side")
            contracts = abs(float(pos.get("contracts") or 0.0))
            if side != "long" or contracts <= 0:
                continue
            count += 1
            symbol = html.escape(str(pos.get("symbol") or "?"))
            entry = float(pos.get("entryPrice") or 0.0)
            mark = float(pos.get("markPrice") or 0.0)
            pnl = float(pos.get("unrealizedPnl") or 0.0)
            margin = float(pos.get("initialMargin") or 0.0)
            roe = (pnl / margin * 100) if margin > 0 else 0.0
            lines.append(
                f"{symbol}: amt={contracts:.6f}, entry={entry:.6f}, "
                f"mark={mark:.6f}, PnL={pnl:.2f}, ROE={roe:.2f}%"
            )
            if count >= 10:
                lines.append("Mas de 10 posiciones; se muestra solo el inicio.")
                break

        if count == 0:
            lines.append("Sin posiciones long abiertas.")
        return "\n".join(lines)
