"""Audit real Binance/Testnet executions without changing order behavior.

Measures fill rate, partial fills, cancellations, time-to-final-state, execution
price versus submitted price, and the exact commission amount/asset returned by
Binance fills. No BNB discount is assumed in code.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from binance_client import BinanceInterface
from config import TRADING_PAIRS, USE_TESTNET


def _ms_to_iso(value):
    if not value:
        return None
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).isoformat()


def audit_symbol(api: BinanceInterface, symbol: str, limit: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = api.client.get_all_orders(symbol=symbol, limit=limit)
    fills = api.client.get_my_trades(symbol=symbol, limit=limit)
    by_order = defaultdict(list)
    for fill in fills:
        by_order[int(fill["orderId"])].append(fill)

    order_rows = []
    fee_rows = []
    for order in orders:
        oid = int(order["orderId"])
        order_fills = by_order.get(oid, [])
        executed_qty = float(order.get("executedQty", 0) or 0)
        quote_qty = float(order.get("cummulativeQuoteQty", 0) or 0)
        avg_fill = quote_qty / executed_qty if executed_qty > 0 and quote_qty > 0 else None
        submitted = float(order.get("price", 0) or 0)
        improvement_bps = None
        if avg_fill is not None and submitted > 0:
            raw = (submitted - avg_fill) / submitted * 10_000.0
            improvement_bps = raw if order.get("side") == "BUY" else -raw
        created = int(order.get("time", 0) or 0)
        updated = int(order.get("updateTime", created) or created)
        order_rows.append({
            "symbol": symbol,
            "order_id": oid,
            "side": order.get("side"),
            "type": order.get("type"),
            "status": order.get("status"),
            "created_at": _ms_to_iso(created),
            "updated_at": _ms_to_iso(updated),
            "time_to_final_sec": max(0.0, (updated - created) / 1000.0),
            "orig_qty": float(order.get("origQty", 0) or 0),
            "executed_qty": executed_qty,
            "fill_ratio": executed_qty / float(order.get("origQty", 1) or 1),
            "submitted_price": submitted if submitted > 0 else None,
            "avg_fill_price": avg_fill,
            "price_improvement_bps_vs_order": improvement_bps,
            "fill_count": len(order_fills),
        })
        for fill in order_fills:
            fee_rows.append({
                "symbol": symbol,
                "order_id": oid,
                "trade_id": fill.get("id"),
                "time": _ms_to_iso(fill.get("time")),
                "price": float(fill.get("price", 0) or 0),
                "qty": float(fill.get("qty", 0) or 0),
                "quote_qty": float(fill.get("quoteQty", 0) or 0),
                "commission": float(fill.get("commission", 0) or 0),
                "commission_asset": fill.get("commissionAsset"),
            })
    return pd.DataFrame(order_rows), pd.DataFrame(fee_rows)


def summarize(orders: pd.DataFrame, fees: pd.DataFrame) -> dict:
    if orders.empty:
        return {"orders": 0}
    terminal = orders[orders["status"].isin(["FILLED", "CANCELED", "REJECTED", "EXPIRED"])]
    filled = terminal[terminal["status"] == "FILLED"]
    partial = terminal[(terminal["executed_qty"] > 0) & (terminal["status"] != "FILLED")]
    cancelled = terminal[terminal["status"] == "CANCELED"]
    return {
        "orders": int(len(orders)),
        "terminal_orders": int(len(terminal)),
        "fill_rate": float(len(filled) / len(terminal)) if len(terminal) else 0.0,
        "partial_fill_rate": float(len(partial) / len(terminal)) if len(terminal) else 0.0,
        "cancel_rate": float(len(cancelled) / len(terminal)) if len(terminal) else 0.0,
        "median_time_to_final_sec": float(terminal["time_to_final_sec"].median()) if len(terminal) else 0.0,
        "median_price_improvement_bps_vs_order": float(filled["price_improvement_bps_vs_order"].dropna().median()) if len(filled) else 0.0,
        "commission_by_asset": ({str(k): float(v) for k, v in fees.groupby("commission_asset")["commission"].sum().items()} if not fees.empty else {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Krypton Binance execution audit")
    parser.add_argument("--symbols", nargs="+", default=list(TRADING_PAIRS))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--prefix", default="execution_audit")
    args = parser.parse_args()

    api = BinanceInterface()
    all_orders, all_fees = [], []
    for symbol in [s.upper() for s in args.symbols]:
        orders, fees = audit_symbol(api, symbol, args.limit)
        all_orders.append(orders)
        all_fees.append(fees)
    orders = pd.concat(all_orders, ignore_index=True) if all_orders else pd.DataFrame()
    fees = pd.concat(all_fees, ignore_index=True) if all_fees else pd.DataFrame()
    orders.to_csv(f"{args.prefix}_orders.csv", index=False)
    fees.to_csv(f"{args.prefix}_fees.csv", index=False)
    report = summarize(orders, fees)
    report["mode"] = "TESTNET" if USE_TESTNET else "PRODUCTION"
    with open(f"{args.prefix}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Comissão é registrada no valor e ativo EXATOS retornados pela Binance; nenhum desconto BNB é presumido.")


if __name__ == "__main__":
    main()
