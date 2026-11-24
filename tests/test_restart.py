import unittest
from unittest.mock import Mock, patch

from binance_usdm import BinanceError
from krypton import reconnect


class StartupRecovery(unittest.TestCase):
    @patch("krypton.time.sleep")
    @patch("krypton.startup")
    def test_recovers_after_network_and_exchange_outage(self, startup, sleep):
        runner = object()
        startup.side_effect = [TimeoutError(), BinanceError(503, 0, "offline"), runner]
        self.assertIs(reconnect(Mock(), Mock()), runner)
        self.assertEqual(sleep.call_count, 2)

    @patch("krypton.time.sleep")
    @patch("krypton.startup", side_effect=BinanceError(401, -2015, "invalid key"))
    def test_rejects_invalid_credentials(self, startup, sleep):
        with self.assertRaises(BinanceError):
            reconnect(Mock(), Mock())
        sleep.assert_not_called()
