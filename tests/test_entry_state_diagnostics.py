import unittest

import pandas as pd

import entry_state_diagnostics as esd


class EntryStateDiagnosticsTests(unittest.TestCase):
    def test_return_at_uses_only_prior_bars(self):
        close = pd.Series([100.0, 110.0, 121.0, 133.1])
        self.assertAlmostEqual(esd._return_at(close, 3, 2), 0.21)

    def test_crossings_counts_state_changes_in_window(self):
        close = pd.Series([9.0, 11.0, 9.0, 11.0, 12.0])
        sma = pd.Series([10.0] * 5)
        self.assertEqual(esd._crossings(close, sma, 4, 4), 3.0)


if __name__ == "__main__":
    unittest.main()
