"""
notifications.py - Webhook y Telegram notifications para el bot.

Soporta:
    - WEBHOOK_URL compatible con Discord/Slack/Telegram HTTP API.
    - TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID para seguimiento directo.
"""

from __future__ import annotations

import logging

import aiohttp

import config

logger = logging.getLogger("notify")


async def _send_json(session: aiohttp.ClientSession, url: str, payload: dict) -> None:
    """Envia un POST JSON con timeout y sin levantar errores al caller."""
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status >= 400:
                logger.warning("Webhook respondio %d: %s", resp.status, await resp.text())
    except Exception as exc:
        logger.debug("Webhook fallo: %s", exc)


async def send_telegram_message(
    message: str,
    session: aiohttp.ClientSession | None = None,
) -> bool:
    """Envia un mensaje directo via Telegram Bot API si esta configurado."""
    if not (config.TELEGRAM_ENABLED and config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID):
        return False

    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    if session is not None:
        await _send_json(session, url, payload)
        return True

    async with aiohttp.ClientSession() as own_session:
        await _send_json(own_session, url, payload)
    return True


async def notify(message: str, level: str = "info") -> None:
    """Envia una notificacion si WEBHOOK_URL o Telegram estan configurados."""
    telegram_ready = bool(
        config.TELEGRAM_ENABLED
        and config.TELEGRAM_BOT_TOKEN
        and config.TELEGRAM_CHAT_ID
    )
    if not config.WEBHOOK_URL and not telegram_ready:
        return

    url = config.WEBHOOK_URL
    payload: dict = {}

    if url:
        if "discord.com" in url or "discordapp.com" in url:
            payload = {"content": message}
        elif "telegram.org" in url or "api.telegram.org" in url:
            payload = {"text": message, "parse_mode": "HTML"}
        else:
            payload = {"text": message}

    try:
        async with aiohttp.ClientSession() as session:
            if url:
                await _send_json(session, url, payload)

            webhook_is_telegram = "telegram.org" in url or "api.telegram.org" in url
            if not webhook_is_telegram:
                await send_telegram_message(message, session=session)
    except Exception as exc:
        logger.debug("notify() excepcion: %s", exc)


async def notify_trade(
    side: str,
    symbol: str,
    rule: str,
    margin_usdt: float,
    amount: float,
    price: float,
    roe_pct: float | None = None,
) -> None:
    """Notificacion formateada para trades ejecutados."""
    icon = "[BUY]" if side == "buy" else "[SELL]"
    msg = (
        f"{icon} <b>TRADE {side.upper()}</b>\n"
        f"<b>Par:</b> {symbol}\n"
        f"<b>Regla:</b> {rule}\n"
        f"<b>Margen:</b> {margin_usdt:.2f} USDT\n"
        f"<b>Cantidad:</b> {amount:.6f}\n"
        f"<b>Precio:</b> {price:.6f}"
    )
    if roe_pct is not None:
        msg += f"\n<b>ROE:</b> {roe_pct:.2f}%"
    await notify(msg)


async def notify_alert(alert_type: str, symbol: str, detail: str) -> None:
    """Notificacion para alertas criticas."""
    label_map = {
        "stop_loss": "[STOP]",
        "circuit_breaker": "[CIRCUIT]",
        "drawdown": "[DRAWDOWN]",
        "slippage": "[SLIPPAGE]",
        "partial_fill": "[PARTIAL]",
    }
    label = label_map.get(alert_type, "[INFO]")
    msg = (
        f"{label} <b>{alert_type.upper().replace('_', ' ')}</b>\n"
        f"<b>Par:</b> {symbol}\n"
        f"<b>Detalle:</b> {detail}"
    )
    await notify(msg)
