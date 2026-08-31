import tempfile
import unittest
from pathlib import Path

from live_c import LiveAggressiveCTradeBot
from tradebot import _queue


class LiveControlTests(unittest.TestCase):
    def test_decision_request_is_file_based_and_single_consumer(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / ".decision"
            _queue(request, "TEST")
            self.assertTrue(request.exists())
            self.assertTrue(LiveAggressiveCTradeBot._consume_request(request))
            self.assertFalse(request.exists())
            self.assertFalse(LiveAggressiveCTradeBot._consume_request(request))

    def test_queue_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / ".decision"
            _queue(request, "TEST")
            _queue(request, "TEST")
            self.assertTrue(request.exists())
            self.assertTrue(LiveAggressiveCTradeBot._consume_request(request))
            self.assertFalse(LiveAggressiveCTradeBot._consume_request(request))


if __name__ == "__main__":
    unittest.main()
