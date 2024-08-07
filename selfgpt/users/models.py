from decimal import Decimal
from uuid import uuid4

from django.contrib.auth.models import AbstractUser
from django.core.validators import MinValueValidator
from django.db import transaction
from django.db.models import CharField, DateTimeField, DecimalField, EmailField, IntegerField, UUIDField
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

from selfgpt.users.helpers import CurrencyConverter
from selfgpt.users.managers import UserManager


class User(AbstractUser):
    """
    Default custom user model for selfGPT.
    If adding fields that need to be filled at user signup,
    check forms.SignupForm and forms.SocialSignupForms accordingly.
    """

    # First and last name do not cover name patterns around the globe
    name = CharField(_("Your name"), blank=True, max_length=255)
    first_name = None  # type: ignore
    last_name = None  # type: ignore
    email = EmailField(_("email address"), unique=True)
    username = None  # type: ignore
    # UUID for getting id of users safely
    uuid = UUIDField(
        verbose_name="UUID",
        db_index=True,
        default=uuid4,
        editable=False,
        help_text=_("UUID for getting id of users safely"),
    )

    # Personal or main company website
    website = CharField(_("Personal or company website"), blank=True, max_length=255)

    phone = CharField(_("Your phone"), blank=True, max_length=255)

    # When user account is registered
    registered_at = DateTimeField(
        verbose_name=_("Registered at"), default=timezone.now, help_text=_("When user account is registered")
    )

    balance = DecimalField(
        max_digits=15,
        decimal_places=2,
        default=0,
        validators=[MinValueValidator(Decimal("0.00"))],
        help_text=_("Current credit balance of the user. It is in micro dollars (1 USD = 1_000_000 micro dollars)"),
    )

    total_completion_tokens = IntegerField(
        default=0, help_text="Total number of tokens generated in response to the user's prompts across all indexes."
    )
    total_prompt_tokens = IntegerField(
        default=0, help_text="Total number of tokens used in the user's prompts across all indexes."
    )
    total_embedding_operations = IntegerField(
        default=0, help_text="Total number of vectorization operations performed across all indexes."
    )
    total_image_annotations = IntegerField(
        default=0, help_text="Total number of image annotations operations performed across all indexes."
    )

    def add_to_balance(self, amount: Decimal):
        """Add a specified amount of balance to the user's account."""
        if amount < 0:
            raise ValueError("Amount to add cannot be negative.")
        with transaction.atomic():
            self.balance += amount
            self.save()

    def subtract_from_balance(self, amount: Decimal):
        """Subtract a specified amount of balance from the user's account."""
        if amount < 0:
            raise ValueError("Amount to subtract cannot be negative.")
        if self.balance < amount:
            raise ValueError("Cannot subtract more balance than the account holds.")
        with transaction.atomic():
            self.balance -= amount
            self.save()

    def check_balance(self) -> Decimal:
        """Return the current credit balance of the user."""
        return CurrencyConverter().microdollars_to_dollars(self.balance)

    def update_global_stats(self, prompt_tokens=0, completion_tokens=0, embedding_operations=0, image_annotations=0):
        """Updates the user's global statistics for token usage and vectorization operations."""
        if completion_tokens < 0 or prompt_tokens < 0 or embedding_operations < 0 or image_annotations < 0:
            raise ValueError("Statistics update values cannot be negative.")

        # Costs per operations (microdollars)
        cost_per_prompt_token = Decimal("1.00")
        cost_per_completion_token = Decimal("3.00")
        cost_per_embedding_operation = Decimal("0.26")
        cost_per_image_annotation = Decimal("15000.00")

        total_cost = (
            (prompt_tokens * cost_per_prompt_token)
            + (completion_tokens * cost_per_completion_token)
            + (embedding_operations * cost_per_embedding_operation)
            + (image_annotations * cost_per_image_annotation)
        )

        with transaction.atomic():
            self.total_completion_tokens += completion_tokens
            self.total_prompt_tokens += prompt_tokens
            self.total_embedding_operations += embedding_operations
            self.total_image_annotations += image_annotations

            # Here we allow to go into negative balance,
            # as the customer can refill their account
            self.balance -= Decimal(total_cost)
            self.save()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []

    objects = UserManager()

    def get_absolute_url(self) -> str:
        """Get URL for user's detail view.

        Returns:
            str: URL for user detail.

        """
        return reverse("users:detail", kwargs={"uuid": self.uuid})
