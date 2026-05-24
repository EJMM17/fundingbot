from __future__ import annotations

import os
import unittest

os.environ.setdefault("KUCOIN_API_KEY", "dummy")
os.environ.setdefault("KUCOIN_SECRET", "dummy")
os.environ.setdefault("KUCOIN_PASSPHRASE", "dummy")

import config
from trading_engine import TradingEngine


class GlobalMarginTests(unittest.TestCase):
    def setUp(self) -> None:
        self._max_total_margin = config.MAX_TOTAL_MARGIN
        self._max_open_positions = config.MAX_OPEN_POSITIONS
        config.MAX_TOTAL_MARGIN = 1_000.0
        config.MAX_OPEN_POSITIONS = 2

    def tearDown(self) -> None:
        config.MAX_TOTAL_MARGIN = self._max_total_margin
        config.MAX_OPEN_POSITIONS = self._max_open_positions

    def test_allows_dca_when_max_position_count_is_reached(self) -> None:
        engine = TradingEngine()
        engine.states["BTC/USDT:USDT"].total_margin_used = 10.0
        engine.states["ETH/USDT:USDT"].total_margin_used = 10.0

        self.assertTrue(engine._check_global_margin(5.0, "BTC/USDT:USDT"))

    def test_blocks_new_symbol_when_max_position_count_is_reached(self) -> None:
        engine = TradingEngine()
        engine.states["BTC/USDT:USDT"].total_margin_used = 10.0
        engine.states["ETH/USDT:USDT"].total_margin_used = 10.0

        self.assertFalse(engine._check_global_margin(5.0, "SOL/USDT:USDT"))


if __name__ == "__main__":
    unittest.main()
