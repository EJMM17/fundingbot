"""
db_local.py — Persistencia local vía SQLite (sin dependencias externas).

Reemplaza completamente a Supabase. Todas las escrituras corren en un
executor para no bloquear el event loop.

Tablas:
    - trade_logs: órdenes ejecutadas
    - pnl_snapshots: snapshots de PNL por posición
    - math_snapshots: métricas matemáticas del arsenal quant
    - monte_carlo_runs: resultados de simulaciones Monte Carlo

El archivo DB se crea en el directorio de trabajo: bot.db
"""

from __future__ import annotations

import asyncio
import functools
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("db")

DB_PATH: Path = Path(__file__).resolve().parent / "bot.db"

# ── Schema inicial ──────────────────────────────────────────────────────────

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS trade_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL DEFAULT (datetime('now')),
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    rule        TEXT,
    margin_usdt REAL,
    amount      REAL,
    price       REAL,
    order_id    TEXT,
    status      TEXT,
    fill_confirmed INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_trade_logs_symbol ON trade_logs(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_logs_ts ON trade_logs(ts DESC);

CREATE TABLE IF NOT EXISTS pnl_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL DEFAULT (datetime('now')),
    symbol          TEXT NOT NULL,
    entry_price     REAL,
    mark_price      REAL,
    position_amt    REAL,
    unrealized_pnl  REAL,
    roe_pct         REAL,
    margin_used     REAL,
    funding_rate    REAL,
    funding_interval_hours REAL,
    annualized_fr_pct      REAL
);

CREATE INDEX IF NOT EXISTS idx_pnl_snapshots_symbol ON pnl_snapshots(symbol);
CREATE INDEX IF NOT EXISTS idx_pnl_snapshots_ts ON pnl_snapshots(ts DESC);

CREATE TABLE IF NOT EXISTS math_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT NOT NULL DEFAULT (datetime('now')),
    symbol              TEXT NOT NULL,
    tail_alpha          REAL,
    tail_pvalue         REAL,
    tail_risk_index     REAL,
    hill_xi             REAL,
    evt_var_95          REAL,
    evt_var_99          REAL,
    evt_es_99           REAL,
    evt_xi              REAL,
    evt_sigma           REAL,
    evt_stop_price      REAL,
    hurst_fr            REAL,
    hurst_method        TEXT,
    fr_regime           TEXT,
    shannon_entropy_fr  REAL,
    te_oi_to_fr         REAL,
    te_fr_to_oi         REAL,
    mi_fr_oi            REAL,
    log_rv_annualized   REAL,
    log_vol_regime      TEXT,
    log_atr_14          REAL,
    mf_width_delta_alpha REAL,
    mf_spectrum_json    TEXT,
    power_score         REAL,
    math_gates_passed   INTEGER,
    gates_blocked_by    TEXT
);

CREATE INDEX IF NOT EXISTS idx_math_snapshots_symbol ON math_snapshots(symbol);
CREATE INDEX IF NOT EXISTS idx_math_snapshots_ts ON math_snapshots(ts DESC);

CREATE TABLE IF NOT EXISTS monte_carlo_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT NOT NULL DEFAULT (datetime('now')),
    symbol              TEXT NOT NULL,
    n_paths             INTEGER,
    horizon_hours       REAL,
    var_95              REAL,
    var_99              REAL,
    expected_shortfall_95 REAL,
    expected_shortfall_99 REAL,
    prob_ruin           REAL,
    prob_profit         REAL,
    expected_pnl        REAL,
    worst_case          REAL,
    kelly_fraction      REAL,
    optimal_size_pct    REAL,
    gpd_xi              REAL,
    gpd_sigma           REAL,
    powerlaw_alpha      REAL
);

CREATE INDEX IF NOT EXISTS idx_mc_symbol ON monte_carlo_runs(symbol);
CREATE INDEX IF NOT EXISTS idx_mc_ts ON monte_carlo_runs(ts DESC);

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


def _init_db() -> None:
    """Crea tablas si no existen."""
    try:
        with sqlite3.connect(str(DB_PATH), check_same_thread=False) as conn:
            conn.executescript(_INIT_SQL)
    except Exception as exc:
        logger.error("DB init error: %s", exc)


# Inicializar al importar el módulo
_init_db()


async def _run_sync(func, *args, **kwargs) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


# ── Writers ─────────────────────────────────────────────────────────────────


def _insert_trade(row: dict) -> None:
    with sqlite3.connect(str(DB_PATH), check_same_thread=False) as conn:
        conn.execute(
            """
            INSERT INTO trade_logs
            (ts, symbol, side, rule, margin_usdt, amount, price, order_id, status, fill_confirmed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("ts") or datetime.now(timezone.utc).isoformat(),
                row["symbol"],
                row["side"],
                row.get("rule"),
                row.get("margin_usdt"),
                row.get("amount"),
                row.get("price"),
                row.get("order_id"),
                row.get("status"),
                1 if row.get("fill_confirmed") else 0,
            ),
        )
        conn.commit()


async def log_trade(
    symbol: str,
    side: str,
    rule: str,
    margin_usdt: float,
    amount: float,
    price: float,
    order_id: str,
    status: str,
    fill_confirmed: bool = False,
) -> None:
    row = {
        "symbol": symbol,
        "side": side,
        "rule": rule,
        "margin_usdt": margin_usdt,
        "amount": amount,
        "price": price,
        "order_id": order_id,
        "status": status,
        "fill_confirmed": fill_confirmed,
    }
    try:
        await _run_sync(_insert_trade, row)
    except Exception as exc:
        logger.error("DB log_trade falló: %s", exc)


def _insert_pnl(row: dict) -> None:
    with sqlite3.connect(str(DB_PATH), check_same_thread=False) as conn:
        conn.execute(
            """
            INSERT INTO pnl_snapshots
            (ts, symbol, entry_price, mark_price, position_amt, unrealized_pnl,
             roe_pct, margin_used, funding_rate, funding_interval_hours, annualized_fr_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("ts") or datetime.now(timezone.utc).isoformat(),
                row["symbol"],
                row.get("entry_price"),
                row.get("mark_price"),
                row.get("position_amt"),
                row.get("unrealized_pnl"),
                row.get("roe_pct"),
                row.get("margin_used"),
                row.get("funding_rate"),
                row.get("funding_interval_hours"),
                row.get("annualized_fr_pct"),
            ),
        )
        conn.commit()


async def log_pnl(
    symbol: str,
    entry_price: float,
    mark_price: float,
    position_amt: float,
    unrealized_pnl: float,
    roe_pct: float,
    margin_used: float,
    funding_rate: float,
    funding_interval_hours: float = 8.0,
) -> None:
    cycles_per_year = (365 * 24) / funding_interval_hours if funding_interval_hours > 0 else 0
    annualized = funding_rate * cycles_per_year * 100

    row = {
        "symbol": symbol,
        "entry_price": entry_price,
        "mark_price": mark_price,
        "position_amt": position_amt,
        "unrealized_pnl": unrealized_pnl,
        "roe_pct": roe_pct,
        "margin_used": margin_used,
        "funding_rate": funding_rate,
        "funding_interval_hours": funding_interval_hours,
        "annualized_fr_pct": annualized,
    }
    try:
        await _run_sync(_insert_pnl, row)
    except Exception as exc:
        logger.error("DB log_pnl falló: %s", exc)


def _insert_math(row: dict) -> None:
    with sqlite3.connect(str(DB_PATH), check_same_thread=False) as conn:
        conn.execute(
            """
            INSERT INTO math_snapshots
            (ts, symbol, tail_alpha, tail_pvalue, tail_risk_index, hill_xi,
             evt_var_95, evt_var_99, evt_es_99, evt_xi, evt_sigma, evt_stop_price,
             hurst_fr, hurst_method, fr_regime, shannon_entropy_fr,
             te_oi_to_fr, te_fr_to_oi, mi_fr_oi, log_rv_annualized,
             log_vol_regime, log_atr_14, mf_width_delta_alpha,
             mf_spectrum_json, power_score, math_gates_passed, gates_blocked_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("ts") or datetime.now(timezone.utc).isoformat(),
                row["symbol"],
                row.get("tail_alpha"),
                row.get("tail_pvalue"),
                row.get("tail_risk_index"),
                row.get("hill_xi"),
                row.get("evt_var_95"),
                row.get("evt_var_99"),
                row.get("evt_es_99"),
                row.get("evt_xi"),
                row.get("evt_sigma"),
                row.get("evt_stop_price"),
                row.get("hurst_fr"),
                row.get("hurst_method"),
                row.get("fr_regime"),
                row.get("shannon_entropy_fr"),
                row.get("te_oi_to_fr"),
                row.get("te_fr_to_oi"),
                row.get("mi_fr_oi"),
                row.get("log_rv_annualized"),
                row.get("log_vol_regime"),
                row.get("log_atr_14"),
                row.get("mf_width_delta_alpha"),
                row.get("mf_spectrum_json"),
                row.get("power_score"),
                1 if row.get("math_gates_passed") else 0,
                row.get("gates_blocked_by"),
            ),
        )
        conn.commit()


async def log_math_snapshot(
    symbol: str,
    tail_alpha: float | None = None,
    tail_pvalue: float | None = None,
    tail_risk_index: float | None = None,
    hill_xi: float | None = None,
    evt_var_95: float | None = None,
    evt_var_99: float | None = None,
    evt_es_99: float | None = None,
    evt_xi: float | None = None,
    evt_sigma: float | None = None,
    evt_stop_price: float | None = None,
    hurst_fr: float | None = None,
    hurst_method: str | None = None,
    fr_regime: str | None = None,
    shannon_entropy_fr: float | None = None,
    te_oi_to_fr: float | None = None,
    te_fr_to_oi: float | None = None,
    mi_fr_oi: float | None = None,
    log_rv_annualized: float | None = None,
    log_vol_regime: str | None = None,
    log_atr_14: float | None = None,
    mf_width_delta_alpha: float | None = None,
    power_score: float | None = None,
    math_gates_passed: bool | None = None,
    gates_blocked_by: str | None = None,
) -> None:
    row = {
        "symbol": symbol,
        "tail_alpha": tail_alpha,
        "tail_pvalue": tail_pvalue,
        "tail_risk_index": tail_risk_index,
        "hill_xi": hill_xi,
        "evt_var_95": evt_var_95,
        "evt_var_99": evt_var_99,
        "evt_es_99": evt_es_99,
        "evt_xi": evt_xi,
        "evt_sigma": evt_sigma,
        "evt_stop_price": evt_stop_price,
        "hurst_fr": hurst_fr,
        "hurst_method": hurst_method,
        "fr_regime": fr_regime,
        "shannon_entropy_fr": shannon_entropy_fr,
        "te_oi_to_fr": te_oi_to_fr,
        "te_fr_to_oi": te_fr_to_oi,
        "mi_fr_oi": mi_fr_oi,
        "log_rv_annualized": log_rv_annualized,
        "log_vol_regime": log_vol_regime,
        "log_atr_14": log_atr_14,
        "mf_width_delta_alpha": mf_width_delta_alpha,
        "power_score": power_score,
        "math_gates_passed": math_gates_passed,
        "gates_blocked_by": gates_blocked_by,
    }
    try:
        await _run_sync(_insert_math, row)
    except Exception as exc:
        logger.error("DB log_math_snapshot falló: %s", exc)


def _insert_mc(row: dict) -> None:
    with sqlite3.connect(str(DB_PATH), check_same_thread=False) as conn:
        conn.execute(
            """
            INSERT INTO monte_carlo_runs
            (ts, symbol, n_paths, horizon_hours, var_95, var_99,
             expected_shortfall_95, expected_shortfall_99, prob_ruin,
             prob_profit, expected_pnl, worst_case, kelly_fraction,
             optimal_size_pct, gpd_xi, gpd_sigma, powerlaw_alpha)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("ts") or datetime.now(timezone.utc).isoformat(),
                row["symbol"],
                row.get("n_paths"),
                row.get("horizon_hours"),
                row.get("var_95"),
                row.get("var_99"),
                row.get("expected_shortfall_95"),
                row.get("expected_shortfall_99"),
                row.get("prob_ruin"),
                row.get("prob_profit"),
                row.get("expected_pnl"),
                row.get("worst_case"),
                row.get("kelly_fraction"),
                row.get("optimal_size_pct"),
                row.get("gpd_xi"),
                row.get("gpd_sigma"),
                row.get("powerlaw_alpha"),
            ),
        )
        conn.commit()


async def log_monte_carlo(row: dict) -> None:
    try:
        await _run_sync(_insert_mc, row)
    except Exception as exc:
        logger.error("DB log_monte_carlo falló: %s", exc)


# ── Readers ─────────────────────────────────────────────────────────────────


def _fetch_recent_math_sync(symbol: str, limit: int = 100):
    with sqlite3.connect(str(DB_PATH), check_same_thread=False) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM math_snapshots WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
            (symbol, limit),
        )
        return [dict(r) for r in cur.fetchall()]


async def fetch_recent_math(symbol: str, limit: int = 100) -> list[dict]:
    try:
        return await _run_sync(_fetch_recent_math_sync, symbol, limit)
    except Exception as exc:
        logger.error("DB fetch_recent_math falló: %s", exc)
        return []
