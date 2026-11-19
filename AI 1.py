"""AI-MARKET-LAB v0.1 - SIMULATION ONLY.

Synthetic (fictional) data, paper trading, human approval for every trade.
No broker connections, no real orders, no API keys required.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import zlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- config
DB_PATH = os.environ.get("AIML_DB", "ai_market_lab.db")
BACKUP_DIR = Path(os.environ.get("AIML_BACKUP_DIR", "backups"))
SYMBOLS = {"NIFTY": 22000.0, "BANKNIFTY": 47000.0, "SENSEX": 73000.0}
START_CASH = 1_000_000.0
MAX_POSITION_PCT = 0.25   # max single-position value / equity
MAX_QTY = 1000            # max units per simulated order
SIZING_PCT = 0.10         # AI proposes ~10% of equity per BUY
FEE_RATE = 0.0003         # simulated fee
MAX_PRICE_DRIFT = 0.02    # proposal price vs latest mark
KEEP_BACKUPS = 10
REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]


class DataError(ValueError):
    """Market data missing or invalid."""


class RiskError(ValueError):
    """A risk check failed."""


class ApprovalError(ValueError):
    """Approval workflow violation."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------- data
def generate_synthetic(symbol: str, days: int = 250, end=None) -> pd.DataFrame:
    """FICTIONAL deterministic price series. Not real market data."""
    if symbol not in SYMBOLS:
        raise DataError(f"Unknown symbol: {symbol}")
    if not isinstance(days, int) or days < 60:
        raise DataError("days must be an integer >= 60")
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today()
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    dates = pd.bdate_range(end=end.normalize(), periods=days)
    close = SYMBOLS[symbol] * np.cumprod(1 + rng.normal(0.0003, 0.011, days))
    open_ = np.concatenate([[SYMBOLS[symbol]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, days)))
    vol = rng.integers(100_000, 900_000, days)
    return pd.DataFrame({"date": dates, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})


def validate_prices(df, max_age_days: int | None = None, today=None) -> pd.DataFrame:
    """Return a cleaned copy or raise DataError. Never fills/invents values."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise DataError("No data")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataError(f"Missing columns: {missing}")
    out = df[REQUIRED_COLS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in REQUIRED_COLS[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if out.isna().any().any():
        raise DataError("Data contains missing/non-numeric values")
    if not np.isfinite(out[REQUIRED_COLS[1:]].to_numpy(dtype=float)).all():
        raise DataError("Data contains infinite values")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataError("Prices must be positive")
    if (out["volume"] < 0).any():
        raise DataError("Negative volume")
    if (out["high"] < out[["open", "close", "low"]].max(axis=1)).any() or \
       (out["low"] > out[["open", "close", "high"]].min(axis=1)).any():
        raise DataError("Inconsistent OHLC (high/low violated)")
    if out["date"].duplicated().any():
        raise DataError("Duplicate dates")
    out = out.sort_values("date").reset_index(drop=True)
    if max_age_days is not None:
        ref = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
        age = (ref.normalize() - out["date"].iloc[-1]).days
        if age > max_age_days:
            raise DataError(f"Data is stale ({age} days old)")
    return out


# ------------------------------------------------------------- indicators
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out = out.where(loss != 0, 100.0)            # no losses -> 100
    return out.where(~((loss == 0) & (gain == 0)), 50.0)  # flat -> 50


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 60:
        raise DataError("Need at least 60 rows for indicators")
    out = df.copy()
    out["sma20"] = out["close"].rolling(20).mean()
    out["sma50"] = out["close"].rolling(50).mean()
    out["rsi14"] = rsi(out["close"])
    return out


def max_drawdown(series: pd.Series) -> float:
    if series.empty or (series <= 0).any():
        raise DataError("Invalid series for drawdown")
    return float((series / series.cummax() - 1).min())


# ----------------------------------------------------------- AI research
def generate_signal(symbol: str, df: pd.DataFrame) -> dict:
    """Transparent rule-based research (SMA trend + RSI). Not an LLM."""
    ind = add_indicators(validate_prices(df))
    last = ind.iloc[-1]
    vals = [last["close"], last["sma20"], last["sma50"], last["rsi14"]]
    if any(not math.isfinite(float(v)) for v in vals):
        raise DataError("Indicators not computable")
    gap = last["sma20"] / last["sma50"] - 1
    if gap > 0 and last["rsi14"] < 70:
        action, why = "BUY", "Uptrend (SMA20>SMA50) and RSI not overbought"
    elif gap < 0 and last["rsi14"] > 30:
        action, why = "SELL", "Downtrend (SMA20<SMA50) and RSI not oversold"
    else:
        action, why = "HOLD", "No clear edge (trend/RSI conflict or extreme)"
    return {"symbol": symbol, "action": action,
            "confidence": round(min(0.95, 0.5 + abs(float(gap)) * 10), 3),
            "price": float(last["close"]), "data_date": last["date"].strftime("%Y-%m-%d"),
            "rsi14": round(float(last["rsi14"]), 2), "sma_gap_pct": round(float(gap) * 100, 3),
            "rationale": why, "source": "SYNTHETIC rule-based model"}


# ------------------------------------------------------------------- risk
def risk_check(side, qty, price, mark, cash, held_qty, equity) -> list[str]:
    """Return list of failure messages (empty = pass)."""
    f = []
    if side not in ("BUY", "SELL"):
        return ["Invalid side"]
    if not isinstance(qty, (int, np.integer)) or qty < 1:
        return ["Quantity must be an integer >= 1"]
    for name, v in (("price", price), ("mark", mark), ("equity", equity)):
        if not isinstance(v, (int, float, np.floating)) or not math.isfinite(v) or v <= 0:
            return [f"Invalid {name}"]
    if qty > MAX_QTY:
        f.append(f"Quantity exceeds max {MAX_QTY}")
    if abs(mark / price - 1) > MAX_PRICE_DRIFT:
        f.append("Proposal price is stale vs latest price")
    if side == "BUY":
        if qty * price * (1 + FEE_RATE) > cash:
            f.append("Insufficient simulated cash")
        if (held_qty + qty) * price > MAX_POSITION_PCT * equity:
            f.append(f"Position would exceed {MAX_POSITION_PCT:.0%} of equity")
    elif qty > held_qty:
        f.append("Cannot sell more than held (no shorting)")
    return f


# --------------------------------------------------------------- database
class Db:
    def __init__(self, path=DB_PATH, start_cash=START_CASH):
        self.path = str(path)
        with self.tx() as c:
            for stmt in """
            CREATE TABLE IF NOT EXISTS account(id INTEGER PRIMARY KEY CHECK(id=1), cash REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS positions(symbol TEXT PRIMARY KEY, qty INTEGER NOT NULL CHECK(qty>0), avg_cost REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS proposals(id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE NOT NULL,
              ts TEXT, symbol TEXT, side TEXT, qty INTEGER, price REAL, data_date TEXT, confidence REAL,
              rationale TEXT, status TEXT, risk_report TEXT);
            CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT,
              proposal_id INTEGER UNIQUE NOT NULL REFERENCES proposals(id), ts TEXT, symbol TEXT, side TEXT,
              qty INTEGER, price REAL, fee REAL, realized_pnl REAL, approved_by TEXT);
            CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT,
              event TEXT, details TEXT);""".split(";"):
                if stmt.strip():
                    c.execute(stmt)
            c.execute("INSERT OR IGNORE INTO account(id,cash) VALUES(1,?)", (float(start_cash),))

    def _conn(self):
        c = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        return c

    @contextmanager
    def tx(self):
        c = self._conn()
        try:
            c.execute("BEGIN IMMEDIATE")
            yield c
            c.execute("COMMIT")
        except BaseException:
            try:
                c.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            c.close()

    @staticmethod
    def _audit(c, event, details, actor="system"):
        c.execute("INSERT INTO audit_log(ts,actor,event,details) VALUES(?,?,?,?)",
                  (_now(), actor, event, json.dumps(details, default=str)))

    def audit(self, event, details, actor="system"):
        with self.tx() as c:
            self._audit(c, event, details, actor)

    @staticmethod
    def _state(c, marks):
        cash = c.execute("SELECT cash FROM account WHERE id=1").fetchone()["cash"]
        pos = {r["symbol"]: (r["qty"], r["avg_cost"]) for r in c.execute("SELECT * FROM positions")}
        value = 0.0
        for s, (q, _) in pos.items():
            if s not in marks or not math.isfinite(marks[s]) or marks[s] <= 0:
                raise DataError(f"Missing/invalid latest price for held position {s}")
            value += q * marks[s]
        return cash, pos, cash + value

    def create_proposal(self, sig: dict, marks: dict):
        """AI path: can only create PENDING proposals, never trades."""
        side = sig["action"]
        if side == "HOLD":
            return None
        key = hashlib.sha256(f"{sig['symbol']}|{side}|{sig['data_date']}".encode()).hexdigest()
        with self.tx() as c:
            if c.execute("SELECT 1 FROM proposals WHERE key=?", (key,)).fetchone():
                self._audit(c, "duplicate_blocked", {"symbol": sig["symbol"], "side": side}, "ai")
                return None
            cash, pos, equity = self._state(c, marks)
            held = pos.get(sig["symbol"], (0, 0.0))[0]
            price = sig["price"]
            qty = int(SIZING_PCT * equity // price) if side == "BUY" else held
            qty = min(qty, MAX_QTY)
            if qty < 1:
                self._audit(c, "no_proposal", {"symbol": sig["symbol"], "reason": "qty<1"}, "ai")
                return None
            fails = risk_check(side, qty, price, marks.get(sig["symbol"], price), cash, held, equity)
            status = "REJECTED_RISK" if fails else "PENDING"
            cur = c.execute(
                "INSERT INTO proposals(key,ts,symbol,side,qty,price,data_date,confidence,rationale,status,risk_report)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (key, _now(), sig["symbol"], side, qty, price, sig["data_date"],
                 sig["confidence"], sig["rationale"], status, json.dumps(fails)))
            self._audit(c, "proposal_created", {"id": cur.lastrowid, "status": status, "risk": fails}, "ai")
            return cur.lastrowid

    def approve(self, proposal_id: int, marks: dict, approved_by: str = "human"):
        """Executes ONE simulated trade. Called only from the human UI button."""
        if not approved_by or approved_by.strip().lower() in ("ai", "system", "auto", "bot"):
            raise ApprovalError("Approval must come from a human")
        try:
            with self.tx() as c:
                p = c.execute("SELECT * FROM proposals WHERE id=?", (proposal_id,)).fetchone()
                if p is None:
                    raise ApprovalError("Proposal not found")
                if p["status"] != "PENDING":
                    raise ApprovalError(f"Proposal is {p['status']}, not PENDING")
                cash, pos, equity = self._state(c, marks)
                s, side, qty, price = p["symbol"], p["side"], p["qty"], p["price"]
                if s not in marks:
                    raise DataError(f"No latest price for {s}")
                held, avg = pos.get(s, (0, 0.0))
                fails = risk_check(side, qty, price, marks[s], cash, held, equity)
                if fails:
                    raise RiskError("; ".join(fails))
                gross = qty * price
                fee = gross * FEE_RATE
                pnl = 0.0
                if side == "BUY":
                    new_q = held + qty
                    new_avg = (held * avg + gross) / new_q
                    c.execute("UPDATE account SET cash=cash-? WHERE id=1", (gross + fee,))
                    c.execute("INSERT INTO positions(symbol,qty,avg_cost) VALUES(?,?,?) "
                              "ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty, avg_cost=excluded.avg_cost",
                              (s, new_q, new_avg))
                else:
                    pnl = (price - avg) * qty - fee
                    c.execute("UPDATE account SET cash=cash+? WHERE id=1", (gross - fee,))
                    if held == qty:
                        c.execute("DELETE FROM positions WHERE symbol=?", (s,))
                    else:
                        c.execute("UPDATE positions SET qty=? WHERE symbol=?", (held - qty, s))
                c.execute("INSERT INTO trades(proposal_id,ts,symbol,side,qty,price,fee,realized_pnl,approved_by)"
                          " VALUES(?,?,?,?,?,?,?,?,?)", (proposal_id, _now(), s, side, qty, price, fee, pnl, approved_by))
                c.execute("UPDATE proposals SET status='APPROVED' WHERE id=?", (proposal_id,))
                self._audit(c, "trade_executed_simulated", {"proposal_id": proposal_id}, approved_by)
        except (ApprovalError, RiskError, DataError) as e:
            self.audit("approval_blocked", {"proposal_id": proposal_id, "reason": str(e)}, approved_by)
            raise

    def reject(self, proposal_id: int, rejected_by: str = "human"):
        with self.tx() as c:
            n = c.execute("UPDATE proposals SET status='REJECTED_HUMAN' WHERE id=? AND status='PENDING'",
                          (proposal_id,)).rowcount
            if n != 1:
                raise ApprovalError("Proposal not found or not PENDING")
            self._audit(c, "proposal_rejected", {"proposal_id": proposal_id}, rejected_by)

    # ---- reads
    def df(self, sql, params=()):
        c = self._conn()
        try:
            return pd.read_sql_query(sql, c, params=params)
        finally:
            c.close()

    def pending(self):
        return self.df("SELECT * FROM proposals WHERE status='PENDING' ORDER BY id")

    def summary(self, marks: dict) -> dict:
        c = self._conn()
        try:
            cash, pos, equity = self._state(c, marks)
        finally:
            c.close()
        t = self.df("SELECT realized_pnl FROM trades WHERE side='SELL'")
        unreal = sum((marks[s] - a) * q for s, (q, a) in pos.items())
        return {"cash": cash, "equity": equity, "return_pct": (equity / START_CASH - 1) * 100,
                "realized_pnl": float(t["realized_pnl"].sum()), "unrealized_pnl": unreal,
                "n_trades": int(self.df("SELECT COUNT(*) n FROM trades")["n"][0]),
                "win_rate_pct": float((t["realized_pnl"] > 0).mean() * 100) if len(t) else None}

    def positions(self, marks: dict) -> pd.DataFrame:
        d = self.df("SELECT * FROM positions")
        if d.empty:
            return d
        d["last"] = d["symbol"].map(marks)
        d["unrealized_pnl"] = (d["last"] - d["avg_cost"]) * d["qty"]
        return d

    def backup(self) -> Path:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        dest = BACKUP_DIR / f"aiml_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db"
        src, dst = self._conn(), sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            src.close()
            dst.close()
        for old in sorted(BACKUP_DIR.glob("aiml_*.db"))[:-KEEP_BACKUPS]:
            old.unlink()
        return dest

    def restore(self, backup_file: str):
        if not Path(backup_file).is_file():
            raise DataError("Backup file not found")
        chk = sqlite3.connect(backup_file)
        try:
            ok = chk.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            chk.close()
        if ok != "ok":
            raise DataError("Backup failed integrity check")
        self.backup()  # safety copy of current state first
        shutil.copyfile(backup_file, self.path)


# ----------------------------------------------------- research + chat
def run_research(db: Db, data: dict, marks: dict) -> list[dict]:
    """Audits every AI decision (incl. HOLD) and queues proposals for approval."""
    results = []
    for sym, df in data.items():
        try:
            sig = generate_signal(sym, df)
            db.audit("ai_decision", sig, "ai")
            sig["proposal_id"] = db.create_proposal(sig, marks)
        except (DataError, RiskError) as e:
            db.audit("ai_error", {"symbol": sym, "error": str(e)}, "ai")
            sig = {"symbol": sym, "action": "ERROR", "rationale": str(e)}
        results.append(sig)
    return results


def assistant_reply(q: str, signals: list[dict], summary: dict | None) -> str:
    ql = q.lower()
    named = [s for s in SYMBOLS if s.lower() in ql]
    if any(w in ql for w in ("signal", "research", "buy", "sell", "view", *[s.lower() for s in SYMBOLS])):
        rows = [s for s in signals if not named or s["symbol"] in named]
        if not rows:
            return "No research available yet."
        return "\n".join(f"**{s['symbol']}**: {s['action']}"
                         + (f" (conf {s['confidence']}, RSI {s['rsi14']}, as of {s['data_date']}) - " if "rsi14" in s else " - ")
                         + s["rationale"] for s in rows) + "\n\n_Synthetic data, simulation only._"
    if summary and any(w in ql for w in ("portfolio", "cash", "equity", "pnl", "performance", "report")):
        return (f"Equity ₹{summary['equity']:,.0f} | Cash ₹{summary['cash']:,.0f} | "
                f"Return {summary['return_pct']:.2f}% | Realized ₹{summary['realized_pnl']:,.0f} | "
                f"Unrealized ₹{summary['unrealized_pnl']:,.0f} (simulated)")
    if "risk" in ql:
        return (f"Checks before every trade: qty 1-{MAX_QTY}, no shorting, sufficient cash, position ≤ "
                f"{MAX_POSITION_PCT:.0%} of equity, price within {MAX_PRICE_DRIFT:.0%} of latest.")
    return "Ask about: signals (NIFTY/BANKNIFTY/SENSEX), portfolio, performance, or risk rules."


# --------------------------------------------------------------------- UI
def main():  # pragma: no cover
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="AI-MARKET-LAB", page_icon="📈", layout="centered")
    st.title("📈 AI-MARKET-LAB")
    st.warning("SIMULATION ONLY - fictional data, paper trades, no broker, no real money.")

    @st.cache_resource
    def get_db():
        return Db()

    try:
        db = get_db()
    except Exception as e:
        st.error(f"Database error: {e}")
        st.stop()

    data, errors = {}, {}
    for s in SYMBOLS:
        try:
            data[s] = validate_prices(generate_synthetic(s), max_age_days=7)
        except DataError as e:
            errors[s] = str(e)
    for s, e in errors.items():
        st.error(f"{s}: {e}")
    marks = {s: float(d["close"].iloc[-1]) for s, d in data.items()}
    signals = []
    for s, d in data.items():
        try:
            signals.append(generate_signal(s, d))
        except DataError as e:
            st.error(f"{s} signal: {e}")
    try:
        summary = db.summary(marks)
    except DataError as e:
        summary = None
        st.error(str(e))

    t_chat, t_res, t_app, t_pf, t_rep, t_adm = st.tabs(
        ["💬 Assistant", "🔬 Research", "✅ Approvals", "💼 Portfolio", "📊 Reports", "🛠 Admin"])

    with t_chat:
        hist = st.session_state.setdefault("chat", [])
        for role, text in hist:
            st.chat_message(role).markdown(text)
        if q := st.chat_input("Ask about signals, portfolio, risk..."):
            hist += [("user", q), ("assistant", assistant_reply(q, signals, summary))]
            st.rerun()

    with t_res:
        if data:
            sym = st.selectbox("Index", list(data))
            ind = add_indicators(data[sym])
            fig = go.Figure(go.Candlestick(x=ind["date"], open=ind["open"], high=ind["high"],
                                           low=ind["low"], close=ind["close"], name=sym))
            fig.add_scatter(x=ind["date"], y=ind["sma20"], name="SMA20")
            fig.add_scatter(x=ind["date"], y=ind["sma50"], name="SMA50")
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{sym} max drawdown (synthetic): {max_drawdown(data[sym]['close']):.1%}")
        if st.button("Run AI research now", type="primary"):
            for r in run_research(db, data, marks):
                st.write(f"**{r['symbol']}** {r['action']} - {r['rationale']}"
                         + (f" → proposal #{r['proposal_id']}" if r.get("proposal_id") else ""))

    with t_app:
        pend = db.pending()
        if pend.empty:
            st.info("No proposals waiting for your approval.")
        for _, p in pend.iterrows():
            with st.expander(f"#{p['id']} {p['side']} {p['qty']} {p['symbol']} @ ₹{p['price']:,.2f}", expanded=True):
                st.write(f"Confidence {p['confidence']} | Data date {p['data_date']}\n\n{p['rationale']}")
                a, b = st.columns(2)
                if a.button("✅ Approve", key=f"a{p['id']}"):
                    try:
                        db.approve(int(p["id"]), marks, "human")
                        st.success("Simulated trade executed.")
                        st.rerun()
                    except (ApprovalError, RiskError, DataError) as e:
                        st.error(f"Blocked: {e}")
                if b.button("❌ Reject", key=f"r{p['id']}"):
                    try:
                        db.reject(int(p["id"]))
                        st.rerun()
                    except ApprovalError as e:
                        st.error(str(e))

    with t_pf:
        if summary:
            c1, c2 = st.columns(2)
            c1.metric("Equity", f"₹{summary['equity']:,.0f}", f"{summary['return_pct']:.2f}%")
            c2.metric("Cash", f"₹{summary['cash']:,.0f}")
            pos = db.positions(marks)
            st.dataframe(pos, use_container_width=True) if not pos.empty else st.info("No open positions.")

    with t_rep:
        if summary:
            st.json({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary.items()})
        tr = db.df("SELECT * FROM trades ORDER BY id DESC")
        st.subheader("Trades")
        st.dataframe(tr, use_container_width=True)
        if not tr.empty:
            st.download_button("Download trades CSV", tr.to_csv(index=False), "trades.csv")
        st.subheader("Audit log (latest 200)")
        st.dataframe(db.df("SELECT * FROM audit_log ORDER BY id DESC LIMIT 200"), use_container_width=True)

    with t_adm:
        if st.button("Create database backup"):
            try:
                st.success(f"Backup saved: {db.backup()}")
            except Exception as e:
                st.error(f"Backup failed: {e}")
        files = sorted(BACKUP_DIR.glob("aiml_*.db")) if BACKUP_DIR.exists() else []
        st.write(f"{len(files)} backup(s) kept (max {KEEP_BACKUPS}).")
        if files:
            st.download_button("Download latest backup", files[-1].read_bytes(), files[-1].name)
        st.caption("Recovery: stop the app, copy a backup over ai_market_lab.db (integrity-checked "
                   "via Db.restore), restart. Note: Streamlit Cloud storage is ephemeral - download backups regularly.")


if __name__ == "__main__":
    main()
