from decimal import Decimal


class CurrencyConverter:
    @staticmethod
    def dollars_to_cents(dollars):
        """Converts dollars to cents."""
        return int(Decimal(dollars) * 100)

    @staticmethod
    def cents_to_dollars(cents):
        """Converts cents to dollars."""
        return Decimal(cents) / 100

    @staticmethod
    def dollars_to_microdollars(dollars):
        """Converts dollars to microdollars."""
        return Decimal(dollars) * 1_000_000

    @staticmethod
    def microdollars_to_dollars(microdollars):
        """Converts microdollars to dollars."""
        return Decimal(microdollars) / 1_000_000

    @staticmethod
    def cents_to_microdollars(cents):
        """Converts cents to microdollars."""
        return Decimal(cents) * 10_000

    @staticmethod
    def microdollars_to_cents(microdollars):
        """Converts microdollars to cents."""
        return Decimal(microdollars) / 10_000
