import unittest
from decimal import Decimal

from selfgpt.users.helpers import CurrencyConverter


class TestCurrencyConverter(unittest.TestCase):
    def test_dollars_to_cents(self):
        self.assertEqual(CurrencyConverter.dollars_to_cents(1), 100)

    def test_cents_to_dollars(self):
        self.assertEqual(CurrencyConverter.cents_to_dollars(100), Decimal("1"))

    def test_dollars_to_microdollars(self):
        self.assertEqual(CurrencyConverter.dollars_to_microdollars(1), Decimal("1000000"))

    def test_microdollars_to_dollars(self):
        self.assertEqual(CurrencyConverter.microdollars_to_dollars(1000000), Decimal("1"))

    def test_cents_to_microdollars(self):
        self.assertEqual(CurrencyConverter.cents_to_microdollars(100), Decimal("1000000"))

    def test_microdollars_to_cents(self):
        self.assertEqual(CurrencyConverter.microdollars_to_cents(1000000), 100)
