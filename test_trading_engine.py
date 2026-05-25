from __future__ import annotations

import os
import unittest

os.environ.setdefault("KUCOIN_API_KEY", "dummy")
os.environ.setdefault("KUCOIN_SECRET", "dummy")
os.environ.setdefault("KUCOIN_PASSPHRASE", "dummy")

import config
from trading_engine import TradingEngine


class _FakeExchange:
    """Exchange mínimo para probar _margin_to_amount sin red."""

    def __init__(self, markets: dict) -> None:
        self.markets = markets

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        return f"{float(amount):.8f}"


class MarginToAmountTests(unittest.TestCase):
    def setUp(self) -> None:
        self._leverage = config.LEVERAGE
        self._bump = config.MIN_NOTIONAL_AUTO_BUMP
        config.LEVERAGE = 5

    def tearDown(self) -> None:
        config.LEVERAGE = self._leverage
        config.MIN_NOTIONAL_AUTO_BUMP = self._bump

    def _engine(self, min_amount: float, contract_size: float = 1.0) -> TradingEngine:
        engine = TradingEngine()
        engine.exchange = _FakeExchange({
            "X/USDT:USDT": {
                "contractSize": contract_size,
                "limits": {"amount": {"min": min_amount}},
            }
        })
        return engine

    def test_passthrough_when_above_minimum(self) -> None:
        # notional = 5 * 5 = 25, price 1 → raw 25 contratos, min 10 → sin bump.
        engine = self._engine(min_amount=10.0)
        result = engine._margin_to_amount("X/USDT:USDT", 5.0, 1.0)
        self.assertIsNotNone(result)
        amount, eff_margin = result
        self.assertAlmostEqual(amount, 25.0)
        self.assertAlmostEqual(eff_margin, 5.0)

    def test_bumps_margin_to_meet_minimum(self) -> None:
        # notional = 5 * 5 = 25, price 100 → raw 0.25 < min 10 → bump a 10.
        config.MIN_NOTIONAL_AUTO_BUMP = True
        engine = self._engine(min_amount=10.0)
        result = engine._margin_to_amount("X/USDT:USDT", 5.0, 100.0)
        self.assertIsNotNone(result)
        amount, eff_margin = result
        self.assertAlmostEqual(amount, 10.0)
        # effective_margin = 10 * 100 * 1 / 5 = 200
        self.assertAlmostEqual(eff_margin, 200.0)
        self.assertGreater(eff_margin, 5.0)

    def test_returns_none_when_bump_disabled(self) -> None:
        config.MIN_NOTIONAL_AUTO_BUMP = False
        engine = self._engine(min_amount=10.0)
        result = engine._margin_to_amount("X/USDT:USDT", 5.0, 100.0)
        self.assertIsNone(result)


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
