from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("KUCOIN_API_KEY", "dummy")
os.environ.setdefault("KUCOIN_SECRET", "dummy")
os.environ.setdefault("KUCOIN_PASSPHRASE", "dummy")

from auto_learning import AutoLearningEngine


class AutoLearningTests(unittest.TestCase):
    def _make_db(self) -> Path:
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        handle.close()
        path = Path(handle.name)
        with sqlite3.connect(str(path)) as conn:
            conn.execute(
                """
                CREATE TABLE trade_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    rule TEXT,
                    margin_usdt REAL,
                    amount REAL,
                    price REAL,
                    order_id TEXT,
                    status TEXT,
                    fill_confirmed INTEGER DEFAULT 1
                )
                """
            )
            conn.executemany(
                """
                INSERT INTO trade_logs
                (ts, symbol, side, rule, margin_usdt, amount, price, order_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("2026-01-01 00:00:00", "WIN/USDT:USDT", "buy", "initial", 5, 1, 100, "b1", "closed"),
                    ("2026-01-01 00:10:00", "WIN/USDT:USDT", "sell", "tp", 0, 1, 110, "s1", "closed"),
                    ("2026-01-01 00:00:00", "LOSE/USDT:USDT", "buy", "initial", 5, 1, 100, "b2", "closed"),
                    ("2026-01-01 00:10:00", "LOSE/USDT:USDT", "sell", "stop", 0, 1, 90, "s2", "closed"),
                ],
            )
            conn.commit()
        return path

    def test_learns_positive_and_negative_symbol_adjustments(self) -> None:
        path = self._make_db()
        try:
            engine = AutoLearningEngine(path, min_trades=1, refresh_seconds=0)
            self.assertTrue(engine.refresh(force=True))

            win = engine.get_adjustment("WIN/USDT:USDT")
            lose = engine.get_adjustment("LOSE/USDT:USDT")

            self.assertGreater(win.score_multiplier, 1.0)
            self.assertLess(lose.score_multiplier, 1.0)
            self.assertEqual(win.sample_count, 1)
            self.assertEqual(lose.sample_count, 1)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
