import tempfile
import unittest
from pathlib import Path

from telemetry import TelemetryStore


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TelemetryStore(Path(self.tmp.name) / "telemetry.db")
        self.store.ensure_client("a", "Cliente A", "a@example.com", "senha-segura-a")
        self.store.ensure_client("b", "Cliente B", "b@example.com", "senha-segura-b")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def test_auth_and_tenant_isolation(self):
        token, client_id = self.store.authenticate("a@example.com", "senha-segura-a")
        self.assertEqual(client_id, "a")
        self.assertEqual(self.store.client_for_token(token), "a")
        self.assertIsNone(self.store.authenticate("a@example.com", "errada"))

        self.store.record_snapshot(
            "a", equity=1100, drawdown=0.02, tactical_equity=600, alpha_equity=500, halted=False, mode="TESTNET"
        )
        self.store.record_snapshot(
            "b", equity=900, drawdown=0.10, tactical_equity=500, alpha_equity=400, halted=False, mode="TESTNET"
        )
        self.store.record_decision(
            "a", symbol="BTCUSDT", sleeve="tactical", decision="BUY", public_reason="Teste"
        )
        self.store.record_decision(
            "b", symbol="SOLUSDT", sleeve="tactical", decision="WAIT", public_reason="Teste"
        )

        a = self.store.dashboard("a")
        self.assertEqual(a["summary"]["equity"], 1100)
        self.assertEqual(len(a["decisions"]), 1)
        self.assertEqual(a["decisions"][0]["symbol"], "BTCUSDT")
        self.assertNotIn("SOLUSDT", str(a))

    def test_return_uses_first_and_latest_snapshot(self):
        self.store.record_snapshot(
            "a", equity=1000, drawdown=0, tactical_equity=550, alpha_equity=450, halted=False, mode="TESTNET",
            ts="2026-01-01T00:00:00+00:00",
        )
        self.store.record_snapshot(
            "a", equity=1250, drawdown=0, tactical_equity=700, alpha_equity=550, halted=False, mode="TESTNET",
            ts="2026-02-01T00:00:00+00:00",
        )
        self.assertAlmostEqual(self.store.dashboard("a")["summary"]["total_return"], 0.25)


if __name__ == "__main__":
    unittest.main()
