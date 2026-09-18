import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import cross_asset_shadow as shadow


class CrossAssetShadowTests(unittest.TestCase):
    def test_last_completed_daily_close_never_uses_current_utc_bar(self):
        self.assertEqual(
            shadow.last_completed_daily_close("2026-08-28T23:59:59Z"),
            pd.Timestamp("2026-08-27", tz="UTC"),
        )
        self.assertEqual(
            shadow.last_completed_daily_close("2026-08-28T00:00:01Z"),
            pd.Timestamp("2026-08-27", tz="UTC"),
        )

    def test_strategy_fingerprint_is_stable_sha256(self):
        a = shadow.strategy_fingerprint()
        b = shadow.strategy_fingerprint()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)
        int(a, 16)

    def test_candidate_is_locked_to_promoted_parameters(self):
        c = shadow.frozen_candidate()
        self.assertEqual(c["alpha_weight"], 0.10)
        self.assertEqual(c["target_vol"], 0.15)
        self.assertEqual(c["top_n"], 2)
        self.assertEqual(c["min_selected"], 2)
        self.assertEqual(c["allocator_rebalance_bars"], 45)
        self.assertEqual(c["sleeve_rebalance_days"], 90)
        self.assertEqual(c["transfer_cost"], 0.003)
        self.assertEqual(c["execution"], "next_daily_open")
        self.assertTrue(c["spot_only"])
        self.assertEqual(c["leverage"], 1.0)

    def test_rebalance_uses_common_bar_count_not_calendar_gap(self):
        # Missing one date deliberately: the 46th common bar is still index 45.
        dates = pd.date_range("2026-08-27", periods=47, freq="D", tz="UTC")
        dates = dates.delete(10)
        data = {}
        for symbol in shadow.ALPHA_SYMBOLS:
            data[symbol] = {"df": pd.DataFrame({"close": 1.0}, index=dates)}
        calendar = shadow.common_paper_calendar(data, dates[-1], shadow.PAPER_START)
        self.assertEqual(len(calendar), 46)
        due, bars = shadow.rebalance_due_from_calendar(calendar, dates[-1])
        self.assertTrue(due)
        self.assertEqual(bars, 45)

    def test_prestart_run_is_read_only_and_does_not_fetch_market_data(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "shadow.jsonl"
            snap = shadow.run(
                "2026-08-27T00:01:00Z",
                out,
                paper_start=pd.Timestamp("2026-08-27", tz="UTC"),
            )
            self.assertEqual(snap["status"], "waiting_for_paper_start")
            self.assertTrue(snap["read_only"])
            self.assertEqual(snap["orders_submitted"], 0)
            rows = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 1)
            recorded = json.loads(rows[0])
            self.assertEqual(recorded["strategy_fingerprint"], shadow.strategy_fingerprint())


if __name__ == "__main__":
    unittest.main()
