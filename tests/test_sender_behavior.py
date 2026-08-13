import pprint
import unittest

from sender_behavior import analyze_sender_behavior
from transaction_lookup_layer import lookup_transaction


class SenderBehaviorTest(unittest.TestCase):
    def test_known_transaction_context_prints_behavior_dictionary(self):
        context = lookup_transaction("TX03699312")

        result = analyze_sender_behavior(context)
        pprint.pp(result)

        self.assertEqual(
            set(result.keys()),
            {"amount_deviation", "recent_frequency", "summary"},
        )
        self.assertIsNotNone(result["amount_deviation"]["historical_median"])
        self.assertIsNotNone(result["amount_deviation"]["ratio"])
        self.assertIsInstance(result["summary"], list)


if __name__ == "__main__":
    unittest.main()
