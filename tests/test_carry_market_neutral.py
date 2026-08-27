import io
import unittest
import zipfile

import numpy as np
import pandas as pd

import carry_market_neutral as cmn


class CarryMarketNeutralTests(unittest.TestCase):
    def test_trailing_funding_apr_uses_only_known_history(self):
        idx = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
        funding = pd.Series([0.0003] * 7 + [0.10], index=idx)
        apr = cmn.trailing_funding_apr(funding, idx[6], 7)
        self.assertAlmostEqual(apr, 0.0003 * 365, places=12)

    def test_zip_parser_accepts_header_and_headerless(self):
        def make_zip(text):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("x.csv", text)
            return buf.getvalue()

        header = "calc_time,funding_interval_hours,last_funding_rate\n1,8,0.001\n"
        no_header = "1,8,0.001\n2,8,0.002\n"
        a = cmn._read_zip_csv(make_zip(header), cmn.FUNDING_COLUMNS)
        b = cmn._read_zip_csv(make_zip(no_header), cmn.FUNDING_COLUMNS)
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 2)
        self.assertEqual(list(b.columns), cmn.FUNDING_COLUMNS)

    def test_epoch_parser_handles_ms_and_us(self):
        ms = pd.Series([1_700_000_000_000])
        us = pd.Series([1_700_000_000_000_000])
        self.assertEqual(cmn._epoch_to_utc(ms).iloc[0], cmn._epoch_to_utc(us).iloc[0])

    def test_notional_fraction_is_bounded(self):
        with self.assertRaises(ValueError):
            cmn.simulate_funding_carry({}, {}, [], "2025-01-01", "2025-01-02",
                                       notional_fraction=0.75)


if __name__ == "__main__":
    unittest.main()
