"""Live engine for the validated Krypton Aggressive C profile.

Architecture
------------
* Spot long-only, one Binance account, two *virtual* sleeves.
* Tactical sleeve: 55% capital target, 2% risk/trade, existing signal stack,
  BTC SMA200 regime filter, continuity gate, cost gate, OCO protection.
* Cross-asset sleeve: 45% capital target, BTC/ETH/SOL/BNB, own SMA200,
  30/90/180d momentum, inverse-vol/covariance sizing, target vol 30%, top 2,
  minimum breadth 2, 45-day rebalance cadence.
* Sleeve capital is rescaled every 90 days. Tactical OCO protects only tactical
  quantity; alpha quantity remains free even when both sleeves own one symbol.
* State is persisted after each material action. Startup refuses an incompatible
  strategy fingerprint or unmanaged balances.

The repository defaults to TESTNET. Production requires USE_TESTNET=false in
.env; this module never flips that safety switch on its own.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from math import sqrt

import numpy as np
import pandas as pd

from binance_client import BinanceInterface
from config import (
    AGGRESSIVE_C_ALPHA_MIN_SELECTED,
    AGGRESSIVE_C_ALPHA_REBALANCE_DAYS,
    AGGRESSIVE_C_ALPHA_SYMBOLS,
    AGGRESSIVE_C_ALPHA_TARGET_VOL,
    AGGRESSIVE_C_ALPHA_TOP_N,
    AGGRESSIVE_C_ALPHA_WEIGHT,
    AGGRESSIVE_C_CONTINUITY_LONG_WINDOW,
    AGGRESSIVE_C_CONTINUITY_MIN_AGE,
    AGGRESSIVE_C_CONTINUITY_SHORT_WINDOW,
    AGGRESSIVE_C_COV_WINDOW,
    AGGRESSIVE_C_MAX_DRAWDOWN_PCT,
    AGGRESSIVE_C_MOMENTUM_WINDOWS,
    AGGRESSIVE_C_REQUIRE_CLEAN_START,
    AGGRESSIVE_C_SLEEVE_REBALANCE_DAYS,
    AGGRESSIVE_C_STATE_DB_FILE,
    AGGRESSIVE_C_TACTICAL_RISK_PER_TRADE,
    AGGRESSIVE_C_TACTICAL_WEIGHT,
    AGGRESSIVE_C_TRANSFER_COST_ASSUMPTION,
    AGGRESSIVE_C_VOL_WINDOW,
    CIRCUIT_BREAKER_PCT,
    ENTRY_FILL_TIMEOUT_SEC,
    FEE_RATE,
    LIVE_QUOTE_ASSET,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MAX_SIMULTANEOUS_POS,
    REGIME_SMA_PERIOD,
    RSI_HIGH,
    RSI_LOW,
    RSI_PERIOD,
    SLIPPAGE_LIMIT_PCT,
    STOP_LOSS_ATR_MULT,
    SUPERTREND_MULTIPLIER,
    SUPERTREND_PERIOD,
    TAKE_PROFIT_ATR_MULT,
    TIMEFRAME,
    TRADING_PAIRS,
    USE_TESTNET,
)
from indicators import compute_atr, compute_signals
from risk_manager import RiskManager

logger = logging.getLogger("Krypton.AggressiveC")
SCHEMA_VERSION = 1
HISTORY_LIMIT = 400
CASH_EPS = 1e-6


def frozen_profile() -> dict:
    return {
        "name": "AGGRESSIVE_C",
        "tactical_symbols": dict(TRADING_PAIRS),
        "alpha_symbols": list(AGGRESSIVE_C_ALPHA_SYMBOLS),
        "tactical_weight": AGGRESSIVE_C_TACTICAL_WEIGHT,
        "alpha_weight": AGGRESSIVE_C_ALPHA_WEIGHT,
        "tactical_risk_per_trade": AGGRESSIVE_C_TACTICAL_RISK_PER_TRADE,
        "alpha_target_vol": AGGRESSIVE_C_ALPHA_TARGET_VOL,
        "alpha_top_n": AGGRESSIVE_C_ALPHA_TOP_N,
        "alpha_min_selected": AGGRESSIVE_C_ALPHA_MIN_SELECTED,
        "alpha_rebalance_days": AGGRESSIVE_C_ALPHA_REBALANCE_DAYS,
        "sleeve_rebalance_days": AGGRESSIVE_C_SLEEVE_REBALANCE_DAYS,
        "transfer_cost_assumption": AGGRESSIVE_C_TRANSFER_COST_ASSUMPTION,
        "max_drawdown": AGGRESSIVE_C_MAX_DRAWDOWN_PCT,
        "momentum_windows": list(AGGRESSIVE_C_MOMENTUM_WINDOWS),
        "vol_window": AGGRESSIVE_C_VOL_WINDOW,
        "cov_window": AGGRESSIVE_C_COV_WINDOW,
        "continuity_min_age": AGGRESSIVE_C_CONTINUITY_MIN_AGE,
        "continuity_short_window": AGGRESSIVE_C_CONTINUITY_SHORT_WINDOW,
        "continuity_long_window": AGGRESSIVE_C_CONTINUITY_LONG_WINDOW,
        "regime_sma": REGIME_SMA_PERIOD,
        "stop_atr_mult": STOP_LOSS_ATR_MULT,
        "tp_atr_mult": TAKE_PROFIT_ATR_MULT,
        "spot_only": True,
        "directional_leverage": 1.0,
        "execution": "completed_close_to_next_daily_open_window",
    }


def strategy_fingerprint() -> str:
    raw = json.dumps(frozen_profile(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cost_gate(entry_price: float, atr: float) -> bool:
    if entry_price <= 0 or atr <= 0:
        return False
    round_trip = 2.0 * FEE_RATE + 0.0005 + 0.0005
    gross_edge = TAKE_PROFIT_ATR_MULT * atr / entry_price
    return gross_edge > 3.0 * round_trip


def _signal_age(signal: pd.Series, ts) -> int:
    s = signal.loc[:ts].astype(int)
    if s.empty or int(s.iloc[-1]) != 1:
        return 0
    age = 0
    for value in reversed(s.tolist()):
        if int(value) != 1:
            break
        age += 1
    return age


def continuity_allowed(signals: dict[str, pd.Series], closes: pd.DataFrame, symbol: str, ts) -> bool:
    """Exact live analogue of the validated persistent-state entry permission."""
    age = _signal_age(signals[symbol], ts)
    if age < AGGRESSIVE_C_CONTINUITY_MIN_AGE:
        return True
    short_values, long_values = [], []
    for s in TRADING_PAIRS:
        series = closes[s].loc[:ts].dropna()
        if len(series) <= AGGRESSIVE_C_CONTINUITY_LONG_WINDOW:
            return True
        short_values.append(float(series.iloc[-1] / series.iloc[-1 - AGGRESSIVE_C_CONTINUITY_SHORT_WINDOW] - 1.0))
        long_values.append(float(series.iloc[-1] / series.iloc[-1 - AGGRESSIVE_C_CONTINUITY_LONG_WINDOW] - 1.0))
    short_breadth = sum(v > 0 for v in short_values) / len(short_values)
    long_breadth = sum(v > 0 for v in long_values) / len(long_values)
    stale_deterioration = short_breadth == 0.0 and long_breadth >= (2.0 / 3.0)
    return not stale_deterioration


def alpha_target_weights(frames: dict[str, pd.DataFrame], ts=None) -> dict[str, float]:
    """Causal Aggressive-C alpha weights from completed daily candles only."""
    symbols = list(AGGRESSIVE_C_ALPHA_SYMBOLS)
    if any(s not in frames or frames[s].empty for s in symbols):
        return {s: 0.0 for s in symbols}
    if ts is None:
        ts = min(frames[s].index[-1] for s in symbols)
    close = pd.concat({s: frames[s]["close"].astype(float).loc[:ts] for s in symbols}, axis=1).sort_index()
    if len(close) < max(200, max(AGGRESSIVE_C_MOMENTUM_WINDOWS), AGGRESSIVE_C_COV_WINDOW) + 1:
        return {s: 0.0 for s in symbols}

    scores, vols = {}, {}
    for s in symbols:
        series = close[s].dropna()
        if ts not in series.index or len(series) < 201:
            continue
        sma200 = float(series.rolling(200).mean().loc[ts])
        px = float(series.loc[ts])
        if not np.isfinite(sma200) or px <= sma200:
            continue
        moms = []
        for w in AGGRESSIVE_C_MOMENTUM_WINDOWS:
            if len(series) <= w:
                moms = []
                break
            mom = float(px / series.shift(w).loc[ts] - 1.0)
            if not np.isfinite(mom):
                moms = []
                break
            moms.append(mom)
        if not moms:
            continue
        score = float(np.mean(moms))
        if score <= 0:
            continue
        rv = float(series.pct_change().rolling(AGGRESSIVE_C_VOL_WINDOW).std().loc[ts] * sqrt(365.0))
        if not np.isfinite(rv) or rv <= 0:
            continue
        scores[s], vols[s] = score, rv

    selected = sorted(scores, key=scores.get, reverse=True)[:AGGRESSIVE_C_ALPHA_TOP_N]
    if len(selected) < AGGRESSIVE_C_ALPHA_MIN_SELECTED:
        return {s: 0.0 for s in symbols}

    inv = np.array([1.0 / vols[s] for s in selected], dtype=float)
    base = inv / inv.sum()
    returns = close[selected].pct_change().dropna().tail(AGGRESSIVE_C_COV_WINDOW)
    if len(returns) < max(20, AGGRESSIVE_C_COV_WINDOW // 2):
        return {s: 0.0 for s in symbols}
    cov = returns.cov().to_numpy(dtype=float) * 365.0
    port_var = float(base @ cov @ base)
    port_vol = sqrt(max(port_var, 0.0))
    scale = min(1.0, AGGRESSIVE_C_ALPHA_TARGET_VOL / port_vol) if port_vol > 0 else 0.0
    out = {s: 0.0 for s in symbols}
    for s, w in zip(selected, base * scale):
        out[s] = float(w)
    return out


class AggressiveCTradeBot:
    def __init__(self):
        self.binance = BinanceInterface()
        self.symbols = list(dict.fromkeys([*TRADING_PAIRS, *AGGRESSIVE_C_ALPHA_SYMBOLS]))
        self.symbol_infos: dict[str, dict] = {}
        self.state: dict = {}
        self.protection_blocked = False
        self.db = sqlite3.connect(AGGRESSIVE_C_STATE_DB_FILE)
        self._init_db()
        self._initialize()

    def _init_db(self):
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS aggressive_c_state (id INTEGER PRIMARY KEY CHECK(id=1), state_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        self.db.commit()

    def _load_state(self):
        row = self.db.execute("SELECT state_json FROM aggressive_c_state WHERE id=1").fetchone()
        return json.loads(row[0]) if row else None

    def _save_state(self):
        self.state["tactical_risk_state"] = self.tactical_risk.snapshot()
        self.db.execute(
            """INSERT INTO aggressive_c_state(id,state_json,updated_at) VALUES(1,?,?)
               ON CONFLICT(id) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at""",
            (json.dumps(self.state, sort_keys=True, default=str), datetime.now(timezone.utc).isoformat()),
        )
        self.db.commit()

    @staticmethod
    def _base_asset(symbol: str) -> str:
        if not symbol.endswith(LIVE_QUOTE_ASSET):
            raise ValueError(f"Símbolo inválido para quote {LIVE_QUOTE_ASSET}: {symbol}")
        return symbol[: -len(LIVE_QUOTE_ASSET)]

    def _initial_state(self, total_cash: float) -> dict:
        now = datetime.now(timezone.utc)
        return {
            "schema_version": SCHEMA_VERSION,
            "strategy_fingerprint": strategy_fingerprint(),
            "initialized_at": now.isoformat(),
            "tactical_cash": total_cash * AGGRESSIVE_C_TACTICAL_WEIGHT,
            "alpha_cash": total_cash * AGGRESSIVE_C_ALPHA_WEIGHT,
            "tactical_positions": {},
            "alpha_qty": {s: 0.0 for s in AGGRESSIVE_C_ALPHA_SYMBOLS},
            "last_alpha_rebalance": None,
            "last_sleeve_rebalance": now.date().isoformat(),
            "portfolio_peak": total_cash,
            "portfolio_daily_start": total_cash,
            "portfolio_daily_date": now.date().isoformat(),
            "portfolio_halted": False,
        }

    def _initialize(self):
        for symbol in self.symbols:
            info = self.binance.get_symbol_info(symbol)
            if info["status"] != "TRADING" or info["quote_asset"] != LIVE_QUOTE_ASSET or not info["is_spot_trading_allowed"]:
                raise RuntimeError(f"Mercado inválido para Aggressive C: {symbol}")
            self.symbol_infos[symbol] = info

        loaded = self._load_state()
        if loaded is None:
            if AGGRESSIVE_C_REQUIRE_CLEAN_START:
                for symbol in self.symbols:
                    if self.binance.get_open_orders(symbol):
                        raise RuntimeError(f"Clean start recusado: ordens abertas em {symbol}")
                    qty = self.binance.get_asset_total(self._base_asset(symbol))
                    if qty * self.binance.get_current_price(symbol) >= self.symbol_infos[symbol]["min_notional"]:
                        raise RuntimeError(f"Clean start recusado: saldo não gerenciado em {symbol}")
            cash = self.binance.get_account_balance(LIVE_QUOTE_ASSET)
            self.state = self._initial_state(cash)
            self.tactical_risk = RiskManager(
                self.state["tactical_cash"],
                risk_per_trade=AGGRESSIVE_C_TACTICAL_RISK_PER_TRADE,
                max_drawdown_pct=AGGRESSIVE_C_MAX_DRAWDOWN_PCT,
                circuit_breaker_pct=CIRCUIT_BREAKER_PCT,
            )
            self._save_state()
        else:
            self.state = loaded
            if int(self.state.get("schema_version", -1)) != SCHEMA_VERSION:
                raise RuntimeError("Schema Aggressive C incompatível")
            if self.state.get("strategy_fingerprint") != strategy_fingerprint():
                raise RuntimeError("Configuração Aggressive C mudou; migração explícita é obrigatória")
            risk_state = self.state.get("tactical_risk_state", {})
            initial = float(risk_state.get("initial_capital", self.state.get("tactical_cash", 0.0)))
            self.tactical_risk = RiskManager(
                max(initial, CASH_EPS),
                risk_per_trade=AGGRESSIVE_C_TACTICAL_RISK_PER_TRADE,
                max_drawdown_pct=AGGRESSIVE_C_MAX_DRAWDOWN_PCT,
                circuit_breaker_pct=CIRCUIT_BREAKER_PCT,
            )
            self.tactical_risk.restore(risk_state)

        self._reconcile_exchange_state(startup=True)
        logger.info(
            "Aggressive C inicializada | mode=%s | fingerprint=%s | tático=%.0f%% alpha=%.0f%%",
            "TESTNET" if USE_TESTNET else "PRODUÇÃO",
            strategy_fingerprint()[:12],
            AGGRESSIVE_C_TACTICAL_WEIGHT * 100,
            AGGRESSIVE_C_ALPHA_WEIGHT * 100,
        )

    def _tactical_qty(self, symbol: str) -> float:
        return float(self.state["tactical_positions"].get(symbol, {}).get("quantity", 0.0))

    def _alpha_qty(self, symbol: str) -> float:
        return float(self.state["alpha_qty"].get(symbol, 0.0))

    def _managed_qty(self, symbol: str) -> float:
        return self._tactical_qty(symbol) + self._alpha_qty(symbol)

    def _portfolio_equity(self) -> float:
        equity = self.binance.get_account_balance(LIVE_QUOTE_ASSET)
        for symbol in self.symbols:
            qty = self.binance.get_asset_total(self._base_asset(symbol))
            equity += qty * self.binance.get_current_price(symbol)
        return float(equity)

    def _tactical_equity(self) -> float:
        equity = float(self.state["tactical_cash"])
        for symbol, pos in self.state["tactical_positions"].items():
            equity += float(pos["quantity"]) * self.binance.get_current_price(symbol)
        return equity

    def _alpha_equity(self) -> float:
        equity = float(self.state["alpha_cash"])
        for symbol in AGGRESSIVE_C_ALPHA_SYMBOLS:
            equity += self._alpha_qty(symbol) * self.binance.get_current_price(symbol)
        return equity

    def _portfolio_risk_update(self):
        now = datetime.now(timezone.utc)
        equity = self._portfolio_equity()
        current_date = now.date().isoformat()
        if self.state.get("portfolio_daily_date") != current_date:
            self.state["portfolio_daily_date"] = current_date
            self.state["portfolio_daily_start"] = equity
        self.state["portfolio_peak"] = max(float(self.state.get("portfolio_peak", equity)), equity)
        peak = float(self.state["portfolio_peak"])
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd >= AGGRESSIVE_C_MAX_DRAWDOWN_PCT:
            self.state["portfolio_halted"] = True
            logger.critical("AGGRESSIVE C PORTFOLIO HALT | DD %.2f%% >= %.2f%%", dd * 100, AGGRESSIVE_C_MAX_DRAWDOWN_PCT * 100)
        return equity, dd

    def _reconcile_exchange_state(self, startup=False):
        tactical_fill_detected = False
        for symbol in self.symbols:
            info = self.symbol_infos[symbol]
            actual = self.binance.get_asset_total(self._base_asset(symbol))
            expected = self._managed_qty(symbol)
            tolerance = max(info["step_size"] * 1.5, 1e-10)
            delta = actual - expected
            if abs(delta) <= tolerance:
                continue
            if delta > tolerance:
                raise RuntimeError(f"Saldo não gerenciado detectado em {symbol}: +{delta:.12f}")

            missing = -delta
            tpos = self.state["tactical_positions"].get(symbol)
            if tpos and missing <= float(tpos["quantity"]) + tolerance:
                new_qty = max(0.0, float(tpos["quantity"]) - missing)
                tactical_fill_detected = True
                if new_qty * self.binance.get_current_price(symbol) < info["min_notional"]:
                    self.state["tactical_positions"].pop(symbol, None)
                else:
                    tpos["quantity"] = new_qty
                    if not self.binance.has_active_oco(symbol, tpos.get("order_list_id")):
                        tpos["order_list_id"] = None
                logger.warning("Reconciliação atribuiu redução à perna tática | %s | missing=%.10f", symbol, missing)
            else:
                raise RuntimeError(f"Déficit de ativo incompatível com estado gerenciado em {symbol}: {missing:.12f}")

        actual_cash = self.binance.get_account_balance(LIVE_QUOTE_ASSET)
        virtual_cash = float(self.state["tactical_cash"]) + float(self.state["alpha_cash"])
        cash_delta = actual_cash - virtual_cash
        if abs(cash_delta) > 0.01:
            if tactical_fill_detected and cash_delta > 0:
                self.state["tactical_cash"] += cash_delta
                logger.info("Provento de OCO reconciliado no caixa tático | %.2f", cash_delta)
            elif cash_delta > 0:
                self.state["tactical_cash"] += cash_delta * AGGRESSIVE_C_TACTICAL_WEIGHT
                self.state["alpha_cash"] += cash_delta * AGGRESSIVE_C_ALPHA_WEIGHT
                logger.warning("Depósito USDT externo alocado 55/45 | %.2f", cash_delta)
            else:
                raise RuntimeError(f"Retirada/gasto USDT não gerenciado detectado: {cash_delta:.2f}")

        self._refresh_protection_status()
        if not startup:
            self._save_state()

    def _refresh_protection_status(self):
        blocked = False
        for symbol, pos in self.state["tactical_positions"].items():
            if not self.binance.has_active_oco(symbol, pos.get("order_list_id")):
                blocked = True
                break
        self.protection_blocked = blocked

    def _cancel_tactical_oco(self, symbol: str):
        pos = self.state["tactical_positions"].get(symbol)
        if not pos:
            return
        oid = pos.get("order_list_id")
        if oid is not None:
            self.binance.cancel_oco_order(symbol, oid)
            pos["order_list_id"] = None
            self._save_state()

    def _protect_tactical(self, symbol: str) -> bool:
        pos = self.state["tactical_positions"].get(symbol)
        if not pos:
            return True
        qty = float(pos["quantity"])
        if qty * self.binance.get_current_price(symbol) < self.symbol_infos[symbol]["min_notional"]:
            self.state["tactical_positions"].pop(symbol, None)
            self._save_state()
            return True
        order = self.binance.create_oco_order(
            symbol=symbol,
            quantity=qty,
            take_profit_price=float(pos["take_profit"]),
            stop_price=float(pos["stop_loss"]),
            symbol_info=self.symbol_infos[symbol],
        )
        if not order:
            self.protection_blocked = True
            logger.critical("Perna tática sem OCO | %s", symbol)
            return False
        pos["order_list_id"] = order.get("orderListId")
        self._save_state()
        self._refresh_protection_status()
        return True

    def _execute_order(self, symbol: str, side: str, quantity: float, sleeve: str, *, reference_price=None, market=False):
        if quantity <= 0:
            return None
        before_cash = self.binance.get_account_balance(LIVE_QUOTE_ASSET)
        before_base = self.binance.get_asset_total(self._base_asset(symbol))
        info = self.symbol_infos[symbol]
        if market:
            order = self.binance.place_market_order(symbol, side, quantity, info)
        else:
            mid = self.binance.get_current_price(symbol)
            px = mid * (1.0005 if side.upper() == "BUY" else 0.9995)
            order = self.binance.place_limit_order(
                symbol, side, quantity, px, info,
                reference_price=reference_price if reference_price is not None else mid,
            )
        if not order:
            return None
        filled = order if order.get("status") == "FILLED" else self.binance.wait_for_fill(symbol, order["orderId"], ENTRY_FILL_TIMEOUT_SEC)
        if not filled:
            return None
        after_cash = self.binance.get_account_balance(LIVE_QUOTE_ASSET)
        after_base = self.binance.get_asset_total(self._base_asset(symbol))
        cash_delta = after_cash - before_cash
        base_delta = after_base - before_base
        if sleeve == "tactical":
            self.state["tactical_cash"] += cash_delta
        elif sleeve == "alpha":
            self.state["alpha_cash"] += cash_delta
        else:
            raise ValueError("sleeve inválido")
        self._save_state()
        return {"order": filled, "cash_delta": cash_delta, "base_delta": base_delta}

    def _close_tactical(self, symbol: str, reason: str):
        pos = self.state["tactical_positions"].get(symbol)
        if not pos:
            return
        self._cancel_tactical_oco(symbol)
        qty = float(pos["quantity"])
        result = self._execute_order(symbol, "SELL", qty, "tactical", market=True)
        if not result:
            self._protect_tactical(symbol)
            return
        sold = max(0.0, -float(result["base_delta"]))
        remaining = max(0.0, qty - sold)
        if remaining * self.binance.get_current_price(symbol) < self.symbol_infos[symbol]["min_notional"]:
            self.state["tactical_positions"].pop(symbol, None)
        else:
            pos["quantity"] = remaining
            self._protect_tactical(symbol)
        self._save_state()
        logger.info("Tático encerrado/reduzido | %s | reason=%s | sold=%.8f", symbol, reason, sold)

    def _open_tactical(self, symbol: str, atr: float, signal_price: float):
        if symbol in self.state["tactical_positions"] or len(self.state["tactical_positions"]) >= MAX_SIMULTANEOUS_POS:
            return
        if self.protection_blocked or self.state.get("portfolio_halted"):
            return
        t_equity = self._tactical_equity()
        if not self.tactical_risk.can_trade(t_equity):
            return
        available = min(float(self.state["tactical_cash"]), self.binance.get_account_balance(LIVE_QUOTE_ASSET))
        mid = self.binance.get_current_price(symbol)
        sizing = self.tactical_risk.calculate_position_size(
            t_equity, mid, atr,
            allocation_pct=TRADING_PAIRS[symbol],
            available_cash=max(0.0, available / (1.0 + FEE_RATE)),
        )
        if sizing["quantity"] <= 0:
            return
        result = self._execute_order(symbol, "BUY", sizing["quantity"], "tactical", reference_price=signal_price)
        if not result:
            return
        bought = max(0.0, float(result["base_delta"]))
        if bought * mid < self.symbol_infos[symbol]["min_notional"]:
            return
        fill = result["order"]
        executed = float(fill.get("executedQty", bought))
        quote = float(fill.get("cummulativeQuoteQty", 0.0))
        entry = quote / executed if executed > 0 and quote > 0 else mid
        self.state["tactical_positions"][symbol] = {
            "entry_price": entry,
            "quantity": bought,
            "stop_loss": entry - float(sizing["sl_distance"]),
            "take_profit": entry + float(sizing["tp_distance"]),
            "order_list_id": None,
        }
        self._save_state()
        self._protect_tactical(symbol)
        logger.info("Aggressive C tactical BUY | %s | qty=%.8f | entry=%.8f", symbol, bought, entry)

    def _rebalance_alpha(self, frames: dict[str, pd.DataFrame], signal_time):
        weights = alpha_target_weights(frames, signal_time)
        alpha_eq = self._alpha_equity()
        prices = {s: self.binance.get_current_price(s) for s in AGGRESSIVE_C_ALPHA_SYMBOLS}

        # Sell excess first; never sell more than the alpha-owned free quantity.
        for symbol in AGGRESSIVE_C_ALPHA_SYMBOLS:
            current_qty = self._alpha_qty(symbol)
            current_value = current_qty * prices[symbol]
            target_value = alpha_eq * float(weights.get(symbol, 0.0))
            excess = max(0.0, current_value - target_value)
            sell_qty = min(current_qty, excess / prices[symbol] if prices[symbol] > 0 else 0.0)
            if sell_qty * prices[symbol] < self.symbol_infos[symbol]["min_notional"]:
                continue
            result = self._execute_order(symbol, "SELL", sell_qty, "alpha")
            if result:
                sold = max(0.0, -float(result["base_delta"]))
                self.state["alpha_qty"][symbol] = max(0.0, current_qty - sold)
                self._save_state()

        alpha_eq = self._alpha_equity()
        for symbol in AGGRESSIVE_C_ALPHA_SYMBOLS:
            current_qty = self._alpha_qty(symbol)
            current_value = current_qty * prices[symbol]
            target_value = alpha_eq * float(weights.get(symbol, 0.0))
            deficit = max(0.0, target_value - current_value)
            available = min(float(self.state["alpha_cash"]), self.binance.get_account_balance(LIVE_QUOTE_ASSET))
            spend = min(deficit, max(0.0, available / (1.0 + FEE_RATE)))
            if spend < self.symbol_infos[symbol]["min_notional"] or prices[symbol] <= 0:
                continue
            result = self._execute_order(symbol, "BUY", spend / prices[symbol], "alpha")
            if result:
                bought = max(0.0, float(result["base_delta"]))
                self.state["alpha_qty"][symbol] = current_qty + bought
                self._save_state()

        self.state["last_alpha_rebalance"] = pd.Timestamp(signal_time).date().isoformat()
        self._save_state()
        logger.info("Alpha rebalance | target=%s | gross=%.3f", weights, sum(weights.values()))

    def _scale_tactical_positions(self, factor: float):
        for symbol in list(self.state["tactical_positions"]):
            pos = self.state["tactical_positions"].get(symbol)
            if not pos:
                continue
            old_qty = float(pos["quantity"])
            target_qty = max(0.0, old_qty * factor)
            delta = target_qty - old_qty
            if abs(delta) * self.binance.get_current_price(symbol) < self.symbol_infos[symbol]["min_notional"]:
                continue
            self._cancel_tactical_oco(symbol)
            if delta < 0:
                result = self._execute_order(symbol, "SELL", -delta, "tactical", market=True)
                if result:
                    sold = max(0.0, -float(result["base_delta"]))
                    pos["quantity"] = max(0.0, old_qty - sold)
            else:
                available = min(float(self.state["tactical_cash"]), self.binance.get_account_balance(LIVE_QUOTE_ASSET))
                max_qty = available / ((1.0 + FEE_RATE) * self.binance.get_current_price(symbol)) if available > 0 else 0.0
                result = self._execute_order(symbol, "BUY", min(delta, max_qty), "tactical")
                if result:
                    pos["quantity"] = old_qty + max(0.0, float(result["base_delta"]))
            if float(pos.get("quantity", 0.0)) * self.binance.get_current_price(symbol) < self.symbol_infos[symbol]["min_notional"]:
                self.state["tactical_positions"].pop(symbol, None)
            else:
                self._protect_tactical(symbol)
            self._save_state()

    def _scale_alpha_positions(self, factor: float):
        prices = {s: self.binance.get_current_price(s) for s in AGGRESSIVE_C_ALPHA_SYMBOLS}
        # Sells before buys.
        if factor < 1.0:
            for symbol in AGGRESSIVE_C_ALPHA_SYMBOLS:
                old = self._alpha_qty(symbol)
                qty = old * (1.0 - factor)
                if qty * prices[symbol] < self.symbol_infos[symbol]["min_notional"]:
                    continue
                result = self._execute_order(symbol, "SELL", qty, "alpha", market=True)
                if result:
                    self.state["alpha_qty"][symbol] = max(0.0, old - max(0.0, -float(result["base_delta"])))
                    self._save_state()
        elif factor > 1.0:
            for symbol in AGGRESSIVE_C_ALPHA_SYMBOLS:
                old = self._alpha_qty(symbol)
                qty = old * (factor - 1.0)
                available = min(float(self.state["alpha_cash"]), self.binance.get_account_balance(LIVE_QUOTE_ASSET))
                max_qty = available / ((1.0 + FEE_RATE) * prices[symbol]) if available > 0 and prices[symbol] > 0 else 0.0
                qty = min(qty, max_qty)
                if qty * prices[symbol] < self.symbol_infos[symbol]["min_notional"]:
                    continue
                result = self._execute_order(symbol, "BUY", qty, "alpha")
                if result:
                    self.state["alpha_qty"][symbol] = old + max(0.0, float(result["base_delta"]))
                    self._save_state()

    def _rebalance_sleeves(self, now_date):
        t_eq, a_eq = self._tactical_equity(), self._alpha_equity()
        total = t_eq + a_eq
        if total <= 0 or t_eq <= 0 or a_eq <= 0:
            return
        target_t = total * AGGRESSIVE_C_TACTICAL_WEIGHT
        target_a = total * AGGRESSIVE_C_ALPHA_WEIGHT
        factor_t, factor_a = target_t / t_eq, target_a / a_eq

        # Reduce overweight sleeves first so actual USDT is available to scale the other.
        if factor_t < 1.0:
            self._scale_tactical_positions(factor_t)
        if factor_a < 1.0:
            self._scale_alpha_positions(factor_a)

        # Re-earmark pooled USDT toward the validated 55/45 sleeve target.
        t_eq, a_eq = self._tactical_equity(), self._alpha_equity()
        total = t_eq + a_eq
        desired_t_cash = max(0.0, total * AGGRESSIVE_C_TACTICAL_WEIGHT - sum(
            self._tactical_qty(s) * self.binance.get_current_price(s) for s in TRADING_PAIRS
        ))
        actual_pool = float(self.state["tactical_cash"]) + float(self.state["alpha_cash"])
        new_t_cash = min(max(desired_t_cash, 0.0), actual_pool)
        self.state["tactical_cash"] = new_t_cash
        self.state["alpha_cash"] = actual_pool - new_t_cash

        # Increase underweight sleeve holdings proportionally after the cash transfer.
        t_eq, a_eq = self._tactical_equity(), self._alpha_equity()
        total = t_eq + a_eq
        if t_eq > 0:
            factor_t = (total * AGGRESSIVE_C_TACTICAL_WEIGHT) / t_eq
            if factor_t > 1.0:
                self._scale_tactical_positions(factor_t)
        if a_eq > 0:
            factor_a = (total * AGGRESSIVE_C_ALPHA_WEIGHT) / a_eq
            if factor_a > 1.0:
                self._scale_alpha_positions(factor_a)

        self.state["last_sleeve_rebalance"] = now_date.isoformat()
        self._save_state()
        logger.info("Sleeves rebalanced | tactical≈55%% alpha≈45%%")

    def _load_frames(self):
        frames = {}
        for symbol in self.symbols:
            df = self.binance.get_ohlcv(symbol, interval=TIMEFRAME, limit=HISTORY_LIMIT, closed_only=True)
            if df.empty:
                raise RuntimeError(f"Sem candles fechados para {symbol}")
            frames[symbol] = df
        common_signal_time = min(frames[s].index[-1] for s in self.symbols)
        return frames, common_signal_time

    @staticmethod
    def _days_since(raw_date: str | None, current_date) -> int | None:
        if not raw_date:
            return None
        return (current_date - pd.Timestamp(raw_date).date()).days

    def daily_cycle(self):
        now = datetime.now(timezone.utc)
        self._reconcile_exchange_state()
        portfolio_eq, portfolio_dd = self._portfolio_risk_update()
        frames, signal_time = self._load_frames()
        signal_date = pd.Timestamp(signal_time).date()

        t_eq = self._tactical_equity()
        self.tactical_risk.reset_daily(t_eq, now.date())
        self.tactical_risk.can_trade(t_eq)

        signals = {}
        for symbol in TRADING_PAIRS:
            signals[symbol] = compute_signals(
                frames[symbol],
                st_period=SUPERTREND_PERIOD,
                st_mult=SUPERTREND_MULTIPLIER,
                rsi_period=RSI_PERIOD,
                rsi_low=RSI_LOW,
                rsi_high=RSI_HIGH,
                macd_fast=MACD_FAST,
                macd_slow=MACD_SLOW,
                macd_sig=MACD_SIGNAL,
            )
        closes = pd.concat({s: frames[s]["close"].astype(float) for s in TRADING_PAIRS}, axis=1)
        btc = frames["BTCUSDT"]["close"].loc[:signal_time]
        risk_on = len(btc) >= REGIME_SMA_PERIOD and float(btc.iloc[-1]) > float(btc.rolling(REGIME_SMA_PERIOD).mean().iloc[-1])

        # Tactical exits are always allowed, including during a portfolio halt.
        for symbol in list(self.state["tactical_positions"]):
            if signal_time not in signals[symbol].index or int(signals[symbol].loc[signal_time]) != 1:
                self._close_tactical(symbol, "Signal reversal/flat")

        alpha_due_days = self._days_since(self.state.get("last_alpha_rebalance"), signal_date)
        if alpha_due_days is None or alpha_due_days >= AGGRESSIVE_C_ALPHA_REBALANCE_DAYS:
            # During a global halt the rebalance may reduce risk but cannot create new exposure.
            if self.state.get("portfolio_halted"):
                zero_frames_target = {s: 0.0 for s in AGGRESSIVE_C_ALPHA_SYMBOLS}
                logger.warning("Portfolio halt ativo; alpha não aumenta exposição")
                # Sell to cash instead of buying under a new target.
                for symbol in AGGRESSIVE_C_ALPHA_SYMBOLS:
                    qty = self._alpha_qty(symbol)
                    if qty * self.binance.get_current_price(symbol) >= self.symbol_infos[symbol]["min_notional"]:
                        result = self._execute_order(symbol, "SELL", qty, "alpha", market=True)
                        if result:
                            self.state["alpha_qty"][symbol] = max(0.0, qty - max(0.0, -float(result["base_delta"])))
                self.state["last_alpha_rebalance"] = signal_date.isoformat()
                self._save_state()
            else:
                self._rebalance_alpha(frames, signal_time)

        sleeve_days = self._days_since(self.state.get("last_sleeve_rebalance"), now.date())
        if sleeve_days is None or sleeve_days >= AGGRESSIVE_C_SLEEVE_REBALANCE_DAYS:
            self._rebalance_sleeves(now.date())

        if not self.state.get("portfolio_halted") and risk_on and not self.protection_blocked:
            for symbol in TRADING_PAIRS:
                if symbol in self.state["tactical_positions"]:
                    continue
                if signal_time not in signals[symbol].index or int(signals[symbol].loc[signal_time]) != 1:
                    continue
                atr = float(compute_atr(frames[symbol]["high"], frames[symbol]["low"], frames[symbol]["close"]).loc[signal_time])
                close = float(frames[symbol].loc[signal_time, "close"])
                allowed = continuity_allowed(signals, closes, symbol, signal_time)
                if allowed and cost_gate(close, atr):
                    self._open_tactical(symbol, atr, close)

        portfolio_eq, portfolio_dd = self._portfolio_risk_update()
        self._save_state()
        logger.info(
            "Aggressive C cycle | signal=%s | equity=%.2f | DD=%.2f%% | tactical=%.2f | alpha=%.2f | halted=%s",
            signal_time, portfolio_eq, portfolio_dd * 100,
            self._tactical_equity(), self._alpha_equity(), self.state.get("portfolio_halted"),
        )

    def run(self):
        logger.info("Krypton Aggressive C iniciado | quote=%s | UTC | mode=%s", LIVE_QUOTE_ASSET, "TESTNET" if USE_TESTNET else "PRODUÇÃO")
        now = datetime.now(timezone.utc)
        last_daily_run = None
        if now.hour == 0 and 5 <= now.minute < 10:
            self.daily_cycle()
            last_daily_run = now.date()
        else:
            logger.info("Boot fora da janela 00:05 UTC; nenhuma entrada atrasada será enviada.")
        while True:
            now = datetime.now(timezone.utc)
            if now.hour == 0 and now.minute >= 5 and last_daily_run != now.date():
                try:
                    self.daily_cycle()
                except Exception:
                    logger.exception("Falha no ciclo Aggressive C")
                last_daily_run = now.date()
            try:
                self._reconcile_exchange_state()
                self._portfolio_risk_update()
                self._save_state()
            except Exception:
                logger.exception("Falha de reconciliação; novas operações ficam inseguras")
                self.state["portfolio_halted"] = True
                self._save_state()
            time.sleep(30)
