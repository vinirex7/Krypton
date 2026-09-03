"""Client-facing telemetry store for the Krypton dashboard.

This database is intentionally separate from the trading engine state database.
Dashboard reads/writes must never be required for order execution to succeed.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PBKDF2_ITERATIONS = 310_000
SESSION_HOURS = 24


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations, salt_hex, digest_hex = encoded.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TelemetryStore:
    def __init__(self, path: str | Path = "krypton_dashboard.db"):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._init_db()

    def _init_db(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS clients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                client_id TEXT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                equity REAL NOT NULL,
                drawdown REAL NOT NULL DEFAULT 0,
                tactical_equity REAL,
                alpha_equity REAL,
                halted INTEGER NOT NULL DEFAULT 0,
                mode TEXT,
                UNIQUE(client_id, ts)
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_client_ts ON portfolio_snapshots(client_id, ts DESC);
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                signal_time TEXT,
                symbol TEXT NOT NULL,
                sleeve TEXT NOT NULL,
                decision TEXT NOT NULL,
                public_reason TEXT NOT NULL,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_decisions_client_ts ON decisions(client_id, ts DESC);
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                exchange_order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                sleeve TEXT NOT NULL,
                quantity REAL NOT NULL,
                fill_price REAL,
                status TEXT,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_orders_client_ts ON orders(client_id, ts DESC);
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                sleeve TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                quantity REAL NOT NULL,
                pnl REAL,
                return_pct REAL,
                reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_trades_client_ts ON trades(client_id, ts DESC);
            """
        )
        self.db.commit()

    def ensure_client(self, client_id: str, name: str, email: str, password: str) -> None:
        if len(password) < 10:
            raise ValueError("Password must have at least 10 characters")
        self.db.execute(
            """INSERT INTO clients(id,name,email,password_hash,active,created_at)
               VALUES(?,?,?,?,1,?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name,email=excluded.email,password_hash=excluded.password_hash""",
            (client_id, name, email.lower().strip(), _hash_password(password), utcnow()),
        )
        self.db.commit()

    def authenticate(self, email: str, password: str) -> tuple[str, str] | None:
        row = self.db.execute(
            "SELECT id,password_hash FROM clients WHERE email=? AND active=1", (email.lower().strip(),)
        ).fetchone()
        if not row or not _verify_password(password, row["password_hash"]):
            return None
        token = secrets.token_urlsafe(32)
        expires = datetime.now(timezone.utc) + timedelta(hours=SESSION_HOURS)
        self.db.execute(
            "INSERT INTO sessions(token_hash,client_id,expires_at,created_at) VALUES(?,?,?,?)",
            (_token_hash(token), row["id"], expires.isoformat(), utcnow()),
        )
        self.db.commit()
        return token, row["id"]

    def client_for_token(self, token: str) -> str | None:
        row = self.db.execute(
            "SELECT client_id,expires_at FROM sessions WHERE token_hash=?", (_token_hash(token),)
        ).fetchone()
        if not row:
            return None
        if datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
            self.db.execute("DELETE FROM sessions WHERE token_hash=?", (_token_hash(token),))
            self.db.commit()
            return None
        return str(row["client_id"])

    def record_snapshot(self, client_id: str, *, equity: float, drawdown: float, tactical_equity: float | None,
                        alpha_equity: float | None, halted: bool, mode: str, ts: str | None = None) -> None:
        self.db.execute(
            """INSERT OR REPLACE INTO portfolio_snapshots
               (client_id,ts,equity,drawdown,tactical_equity,alpha_equity,halted,mode)
               VALUES(?,?,?,?,?,?,?,?)""",
            (client_id, ts or utcnow(), equity, drawdown, tactical_equity, alpha_equity, int(halted), mode),
        )
        self.db.commit()

    def record_decision(self, client_id: str, *, symbol: str, sleeve: str, decision: str, public_reason: str,
                        signal_time: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        self.db.execute(
            """INSERT INTO decisions(client_id,ts,signal_time,symbol,sleeve,decision,public_reason,metadata_json)
               VALUES(?,?,?,?,?,?,?,?)""",
            (client_id, utcnow(), signal_time, symbol, sleeve, decision, public_reason,
             json.dumps(metadata or {}, sort_keys=True)),
        )
        self.db.commit()

    def record_order(self, client_id: str, *, symbol: str, side: str, sleeve: str, quantity: float,
                     fill_price: float | None, status: str | None, exchange_order_id: str | None,
                     metadata: dict[str, Any] | None = None) -> None:
        self.db.execute(
            """INSERT INTO orders(client_id,ts,exchange_order_id,symbol,side,sleeve,quantity,fill_price,status,metadata_json)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (client_id, utcnow(), exchange_order_id, symbol, side, sleeve, quantity, fill_price, status,
             json.dumps(metadata or {}, sort_keys=True, default=str)),
        )
        self.db.commit()

    def record_trade(self, client_id: str, *, symbol: str, sleeve: str, quantity: float,
                     entry_price: float | None, exit_price: float | None, pnl: float | None,
                     return_pct: float | None, reason: str | None = None) -> None:
        self.db.execute(
            """INSERT INTO trades(client_id,ts,symbol,sleeve,entry_price,exit_price,quantity,pnl,return_pct,reason)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (client_id, utcnow(), symbol, sleeve, entry_price, exit_price, quantity, pnl, return_pct, reason),
        )
        self.db.commit()

    def dashboard(self, client_id: str, limit: int = 50) -> dict[str, Any]:
        latest = self.db.execute(
            "SELECT * FROM portfolio_snapshots WHERE client_id=? ORDER BY ts DESC LIMIT 1", (client_id,)
        ).fetchone()
        first = self.db.execute(
            "SELECT * FROM portfolio_snapshots WHERE client_id=? ORDER BY ts ASC LIMIT 1", (client_id,)
        ).fetchone()
        snapshots = self.db.execute(
            "SELECT ts,equity,drawdown FROM portfolio_snapshots WHERE client_id=? ORDER BY ts DESC LIMIT 500",
            (client_id,),
        ).fetchall()
        decisions = self.db.execute(
            "SELECT ts,signal_time,symbol,sleeve,decision,public_reason FROM decisions WHERE client_id=? ORDER BY ts DESC LIMIT ?",
            (client_id, limit),
        ).fetchall()
        orders = self.db.execute(
            "SELECT ts,symbol,side,sleeve,quantity,fill_price,status FROM orders WHERE client_id=? ORDER BY ts DESC LIMIT ?",
            (client_id, limit),
        ).fetchall()
        trades = self.db.execute(
            "SELECT ts,symbol,sleeve,entry_price,exit_price,quantity,pnl,return_pct,reason FROM trades WHERE client_id=? ORDER BY ts DESC LIMIT ?",
            (client_id, limit),
        ).fetchall()
        initial_equity = float(first["equity"]) if first else None
        current_equity = float(latest["equity"]) if latest else None
        total_return = (current_equity / initial_equity - 1.0) if initial_equity and current_equity else None
        return {
            "summary": {
                "equity": current_equity,
                "initial_equity": initial_equity,
                "total_return": total_return,
                "drawdown": float(latest["drawdown"]) if latest else None,
                "halted": bool(latest["halted"]) if latest else None,
                "mode": latest["mode"] if latest else None,
                "updated_at": latest["ts"] if latest else None,
            },
            "snapshots": [dict(r) for r in reversed(snapshots)],
            "decisions": [dict(r) for r in decisions],
            "orders": [dict(r) for r in orders],
            "trades": [dict(r) for r in trades],
        }
