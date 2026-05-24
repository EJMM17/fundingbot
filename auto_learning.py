"""
auto_learning.py - Ajustes conservadores basados en resultados historicos.

El modulo no predice precios ni cambia la tesis base. Aprende de operaciones
cerradas en trade_logs y genera multiplicadores acotados por simbolo para:
    - score de oportunidades
    - tamano de DCA/adaptive sizing
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import config

logger = logging.getLogger("learn")


_LEARNING_SQL = """
CREATE TABLE IF NOT EXISTS learning_stats (
    symbol           TEXT PRIMARY KEY,
    sample_count     INTEGER NOT NULL,
    wins             INTEGER NOT NULL,
    losses           INTEGER NOT NULL,
    win_rate         REAL NOT NULL,
    avg_roe_pct      REAL NOT NULL,
    best_roe_pct     REAL NOT NULL,
    worst_roe_pct    REAL NOT NULL,
    avg_hold_minutes REAL,
    score_multiplier REAL NOT NULL,
    size_multiplier  REAL NOT NULL,
    updated_ts       TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class SymbolLearning:
    symbol: str
    sample_count: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_roe_pct: float = 0.0
    best_roe_pct: float = 0.0
    worst_roe_pct: float = 0.0
    avg_hold_minutes: float | None = None
    score_multiplier: float = 1.0
    size_multiplier: float = 1.0
    updated_ts: str = ""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None


class AutoLearningEngine:
    """Calcula ajustes por simbolo a partir de trades cerrados."""

    def __init__(
        self,
        db_path: Path,
        *,
        min_trades: int | None = None,
        refresh_seconds: int | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.min_trades = min_trades if min_trades is not None else config.AUTO_LEARNING_MIN_TRADES
        self.refresh_seconds = (
            refresh_seconds
            if refresh_seconds is not None
            else config.AUTO_LEARNING_REFRESH_SECONDS
        )
        self.stats: dict[str, SymbolLearning] = {}
        self.last_refresh_ts: float = 0.0

    def refresh(self, force: bool = False) -> bool:
        if not config.ENABLE_AUTO_LEARNING:
            return False

        now = time.time()
        if not force and now - self.last_refresh_ts < self.refresh_seconds:
            return False

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(_LEARNING_SQL)
            outcomes = self._load_closed_outcomes(conn)
            self.stats = self._aggregate(outcomes)
            self._persist(conn, self.stats.values())

        self.last_refresh_ts = now
        logger.info("AUTOLEARN refresh | simbolos=%d | cierres=%d", len(self.stats), len(outcomes))
        return True

    def get_adjustment(self, symbol: str) -> SymbolLearning:
        return self.stats.get(symbol, SymbolLearning(symbol=symbol))

    def render_summary(self, limit: int = 6) -> str:
        if not config.ENABLE_AUTO_LEARNING:
            return "Autoaprendizaje desactivado."
        if not self.stats:
            return "Autoaprendizaje activo, aun sin cierres suficientes."

        ranked = sorted(
            self.stats.values(),
            key=lambda s: (s.sample_count >= self.min_trades, s.avg_roe_pct),
            reverse=True,
        )[:limit]
        lines = [
            "Autoaprendizaje",
            f"Min cierres/simbolo: {self.min_trades}",
        ]
        for stat in ranked:
            lines.append(
                f"{stat.symbol}: n={stat.sample_count}, win={stat.win_rate*100:.0f}%, "
                f"avgROE={stat.avg_roe_pct:.2f}%, score x{stat.score_multiplier:.2f}, "
                f"size x{stat.size_multiplier:.2f}"
            )
        return "\n".join(lines)

    def _load_closed_outcomes(self, conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT id, ts, symbol, side, rule, amount, price
            FROM trade_logs
            WHERE side IN ('buy', 'sell')
              AND amount IS NOT NULL
              AND amount > 0
              AND price IS NOT NULL
              AND price > 0
            ORDER BY ts ASC, id ASC
            """
        ).fetchall()

        open_state: dict[str, dict[str, Any]] = {}
        outcomes: list[dict[str, Any]] = []

        for row in rows:
            symbol = row["symbol"]
            side = str(row["side"]).lower()
            amount = float(row["amount"] or 0.0)
            price = float(row["price"] or 0.0)
            ts = _parse_ts(row["ts"])
            if amount <= 0 or price <= 0:
                continue

            state = open_state.setdefault(
                symbol,
                {"qty": 0.0, "cost": 0.0, "opened_ts": None, "rules": []},
            )

            if side == "buy":
                state["qty"] += amount
                state["cost"] += amount * price
                state["opened_ts"] = state["opened_ts"] or ts
                if row["rule"]:
                    state["rules"].append(str(row["rule"]))
                continue

            if side != "sell" or state["qty"] <= 0 or state["cost"] <= 0:
                continue

            avg_entry = state["cost"] / state["qty"]
            if avg_entry <= 0:
                continue

            close_qty = min(amount, state["qty"])
            price_return = (price - avg_entry) / avg_entry
            roe_pct = price_return * config.LEVERAGE * 100

            hold_minutes = None
            opened_ts = state.get("opened_ts")
            if opened_ts and ts:
                hold_minutes = max(0.0, (ts - opened_ts).total_seconds() / 60)

            outcomes.append(
                {
                    "symbol": symbol,
                    "roe_pct": roe_pct,
                    "hold_minutes": hold_minutes,
                    "rule": state["rules"][-1] if state["rules"] else None,
                }
            )

            if close_qty >= state["qty"] * 0.99:
                state["qty"] = 0.0
                state["cost"] = 0.0
                state["opened_ts"] = None
                state["rules"] = []
            else:
                state["qty"] -= close_qty
                state["cost"] -= avg_entry * close_qty

        return outcomes

    def _aggregate(self, outcomes: list[dict[str, Any]]) -> dict[str, SymbolLearning]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for outcome in outcomes:
            grouped.setdefault(outcome["symbol"], []).append(outcome)

        stats: dict[str, SymbolLearning] = {}
        updated_ts = datetime.now(timezone.utc).isoformat()
        for symbol, items in grouped.items():
            roes = [float(item["roe_pct"]) for item in items]
            holds = [
                float(item["hold_minutes"])
                for item in items
                if item.get("hold_minutes") is not None
            ]
            sample_count = len(roes)
            wins = sum(1 for roe in roes if roe > 0)
            losses = sample_count - wins
            win_rate = wins / sample_count if sample_count else 0.0
            avg_roe = mean(roes) if roes else 0.0

            score_multiplier, size_multiplier = self._multipliers(
                sample_count=sample_count,
                win_rate=win_rate,
                avg_roe_pct=avg_roe,
                worst_roe_pct=min(roes) if roes else 0.0,
            )

            stats[symbol] = SymbolLearning(
                symbol=symbol,
                sample_count=sample_count,
                wins=wins,
                losses=losses,
                win_rate=win_rate,
                avg_roe_pct=avg_roe,
                best_roe_pct=max(roes) if roes else 0.0,
                worst_roe_pct=min(roes) if roes else 0.0,
                avg_hold_minutes=mean(holds) if holds else None,
                score_multiplier=score_multiplier,
                size_multiplier=size_multiplier,
                updated_ts=updated_ts,
            )
        return stats

    def _multipliers(
        self,
        *,
        sample_count: int,
        win_rate: float,
        avg_roe_pct: float,
        worst_roe_pct: float,
    ) -> tuple[float, float]:
        if sample_count < self.min_trades:
            return 1.0, 1.0

        max_boost = max(0.0, config.AUTO_LEARNING_MAX_BOOST)
        max_penalty = max(0.0, config.AUTO_LEARNING_MAX_PENALTY)
        low = max(0.1, 1.0 - max_penalty)

        edge_component = _clamp(avg_roe_pct / 25.0, -max_penalty, max_boost)
        win_component = _clamp((win_rate - 0.50) * 0.5, -max_penalty, max_boost)
        drawdown_component = _clamp(worst_roe_pct / 100.0, -max_penalty, 0.0)

        score_multiplier = 1.0 + edge_component + win_component + drawdown_component
        score_multiplier = _clamp(score_multiplier, low, 1.0 + max_boost)

        size_multiplier = 1.0 + (score_multiplier - 1.0) * 0.65
        size_multiplier = _clamp(size_multiplier, low, 1.0 + min(max_boost, 0.10))
        return score_multiplier, size_multiplier

    def _persist(
        self,
        conn: sqlite3.Connection,
        stats: Any,
    ) -> None:
        conn.execute("DELETE FROM learning_stats")
        conn.executemany(
            """
            INSERT INTO learning_stats
            (symbol, sample_count, wins, losses, win_rate, avg_roe_pct,
             best_roe_pct, worst_roe_pct, avg_hold_minutes, score_multiplier,
             size_multiplier, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    stat.symbol,
                    stat.sample_count,
                    stat.wins,
                    stat.losses,
                    stat.win_rate,
                    stat.avg_roe_pct,
                    stat.best_roe_pct,
                    stat.worst_roe_pct,
                    stat.avg_hold_minutes,
                    stat.score_multiplier,
                    stat.size_multiplier,
                    stat.updated_ts,
                )
                for stat in stats
            ],
        )
        conn.commit()
