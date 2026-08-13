import unittest

from transaction_lookup_layer import (
    REQUIRED_TRANSACTION_COLUMNS,
    generate_transaction_id,
    lookup_transaction,
)


class TransactionLookupLayerTest(unittest.TestCase):
    def test_known_transaction_id_returns_complete_raw_context(self):
        transaction_id = generate_transaction_id(1)

        row = lookup_transaction(transaction_id)

        self.assertIsNotNone(row)
        self.assertEqual(row["transaction_id"], "TX00000001")
        for column in REQUIRED_TRANSACTION_COLUMNS:
            self.assertIn(column, row.index)

    def test_invalid_transaction_id_returns_none(self):
        self.assertIsNone(lookup_transaction("TX99999999_NOT_REAL"))


if __name__ == "__main__":
    unittest.main()
