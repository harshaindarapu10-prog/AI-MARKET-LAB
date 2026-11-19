import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import app
from app import (ApprovalError, DataError, Db, RiskError, add_indicators, generate_signal,
                 generate_synthetic, max_drawdown, risk_check, rsi, run_research, validate_prices)

END = "2026-11-19"


def good():
    return generate_synthetic("NIFTY", 120, end=END)


class TestData(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(len(validate_prices(good())), 120)

    def test_missing_column(self):
        with self.assertRaises(DataError):
            validate_prices(good().drop(columns=["close"]))

    def test_nan(self):
        d = good(); d.loc[5, "close"] = np.nan
        with self.assertRaises(DataError):
            validate_prices(d)

    def test_negative_price(self):
        d = good(); d.loc[5, "low"] = -1
        with self.assertRaises(DataError):
            validate_prices(d)

    def test_bad_ohlc(self):
        d = good(); d.loc[5, "high"] = d.loc[5, "low"] * 0.5
        with self.assertRaises(DataError):
            validate_prices(d)

    def test_duplicate_dates(self):
        d = good(); d.loc[1, "date"] = d.loc[0, "date"]
        with self.assertRaises(DataError):
            validate_prices(d)

    def test_stale(self):
        with self.assertRaises(DataError):
            validate_prices(good(), max_age_days=3, today="2027-03-01")

    def test_empty_and_none(self):
        for bad in (None, pd.DataFrame()):
            with self.assertRaises(DataError):
                validate_prices(bad)

    def test_deterministic(self):
        pd.testing.assert_frame_equal(good(), good())


class TestCalcs(unittest.TestCase):
    def test_sma_exact(self):
        d = add_indicators(good())
        self.assertAlmostEqual(d["sma20"].iloc[-1], d["close"].iloc[-20:].mean(), places=9)

    def test_rsi_bounds_and_extremes(self):
        up = pd.Series(np.arange(1, 60, dtype=float))
        self.assertEqual(rsi(up).iloc[-1], 100.0)
        self.assertEqual(rsi(pd.Series([5.0] * 40)).iloc[-1], 50.0)
        r = rsi(good()["close"]).dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())

    def test_drawdown(self):
        self.assertAlmostEqual(max_drawdown(pd.Series([100, 120, 90, 100])), -0.25)
        with self.assertRaises(DataError):
            max_drawdown(pd.Series([1, 0]))

    def test_signal_fields(self):
        s = generate_signal("NIFTY", good())
        self.assertIn(s["action"], ("BUY", "SELL", "HOLD"))
        self.assertGreater(s["price"], 0)

    def test_too_short(self):
        with self.assertRaises(DataError):
            generate_signal("NIFTY", good().head(30))


class TestRisk(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(risk_check("BUY", 5, 100.0, 100.0, 1e6, 0, 1e6), [])

    def test_failures(self):
        self.assertTrue(risk_check("BUY", 0, 100.0, 100.0, 1e6, 0, 1e6))
        self.assertTrue(risk_check("BUY", 5000, 100.0, 100.0, 1e9, 0, 1e9))
        self.assertTrue(risk_check("BUY", 5, 100.0, 100.0, 10.0, 0, 1e6))
        self.assertTrue(risk_check("BUY", 900, 1000.0, 1000.0, 1e7, 0, 1e6))
        self.assertTrue(risk_check("SELL", 5, 100.0, 100.0, 1e6, 0, 1e6))
        self.assertTrue(risk_check("BUY", 5, 100.0, 120.0, 1e6, 0, 1e6))
        self.assertTrue(risk_check("BUY", 5, float("nan"), 100.0, 1e6, 0, 1e6))


class TestDb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        app.BACKUP_DIR = Path(self.tmp) / "bk"
        self.db = Db(os.path.join(self.tmp, "t.db"), 1_000_000.0)
        self.marks = {"NIFTY": 1000.0}
        self.buy = {"symbol": "NIFTY", "action": "BUY", "price": 1000.0, "data_date": "2026-11-19",
                    "confidence": 0.6, "rationale": "t"}

    def test_duplicate_proposal_blocked(self):
        self.assertIsNotNone(self.db.create_proposal(self.buy, self.marks))
        self.assertIsNone(self.db.create_proposal(self.buy, self.marks))
        self.assertEqual(len(self.db.df("SELECT * FROM proposals")), 1)

    def test_buy_sell_math_and_double_approve(self):
        pid = self.db.create_proposal(self.buy, self.marks)  # 10% of 1M / 1000 = 100 units
        self.db.approve(pid, self.marks)
        cash = 1_000_000 - 100 * 1000 * (1 + app.FEE_RATE)
        self.assertAlmostEqual(self.db.summary(self.marks)["cash"], cash, places=6)
        with self.assertRaises(ApprovalError):
            self.db.approve(pid, self.marks)
        self.assertEqual(len(self.db.df("SELECT * FROM trades")), 1)
        m2 = {"NIFTY": 1010.0}
        sell = dict(self.buy, action="SELL", price=1010.0, data_date="2026-11-20")
        sid = self.db.create_proposal(sell, m2)
        self.db.approve(sid, m2)
        s = self.db.summary(m2)
        self.assertAlmostEqual(s["realized_pnl"], 10 * 100 - 100 * 1010 * app.FEE_RATE, places=6)
        self.assertTrue(self.db.positions(m2).empty)

    def test_ai_cannot_approve(self):
        pid = self.db.create_proposal(self.buy, self.marks)
        for who in ("ai", "system", ""):
            with self.assertRaises(ApprovalError):
                self.db.approve(pid, self.marks, who)
        self.assertEqual(len(self.db.df("SELECT * FROM trades")), 0)

    def test_reject_then_approve_blocked(self):
        pid = self.db.create_proposal(self.buy, self.marks)
        self.db.reject(pid)
        with self.assertRaises(ApprovalError):
            self.db.approve(pid, self.marks)

    def test_sell_without_holdings_creates_nothing(self):
        sell = dict(self.buy, action="SELL")
        self.assertIsNone(self.db.create_proposal(sell, self.marks))

    def test_stale_price_blocks_approval(self):
        pid = self.db.create_proposal(self.buy, self.marks)
        with self.assertRaises(RiskError):
            self.db.approve(pid, {"NIFTY": 1200.0})
        self.assertEqual(len(self.db.df("SELECT * FROM trades")), 0)

    def test_audit_logged(self):
        pid = self.db.create_proposal(self.buy, self.marks)
        self.db.approve(pid, self.marks)
        ev = set(self.db.df("SELECT event FROM audit_log")["event"])
        self.assertTrue({"proposal_created", "trade_executed_simulated"} <= ev)

    def test_backup_and_restore(self):
        pid = self.db.create_proposal(self.buy, self.marks)
        bk = self.db.backup()
        self.assertTrue(bk.exists())
        self.db.approve(pid, self.marks)
        self.db.restore(str(bk))
        self.assertEqual(len(self.db.df("SELECT * FROM trades")), 0)
        with self.assertRaises(DataError):
            self.db.restore("/nonexistent.db")


class TestIntegration(unittest.TestCase):
    def test_full_research_flow(self):
        tmp = tempfile.mkdtemp()
        app.BACKUP_DIR = Path(tmp) / "bk"
        db = Db(os.path.join(tmp, "i.db"))
        data = {s: validate_prices(generate_synthetic(s, 200, end=END)) for s in app.SYMBOLS}
        marks = {s: float(d["close"].iloc[-1]) for s, d in data.items()}
        res = run_research(db, data, marks)
        self.assertEqual(len(res), 3)
        self.assertEqual(len(db.df("SELECT * FROM audit_log WHERE event='ai_decision'")), 3)
        self.assertEqual(len(db.df("SELECT * FROM trades")), 0)  # nothing executes without approval
        run_research(db, data, marks)  # rerun: no duplicate proposals
        n = len(db.df("SELECT * FROM proposals"))
        run_research(db, data, marks)
        self.assertEqual(n, len(db.df("SELECT * FROM proposals")))
        for _, p in db.pending().iterrows():
            db.approve(int(p["id"]), marks)
        self.assertGreaterEqual(db.summary(marks)["equity"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
