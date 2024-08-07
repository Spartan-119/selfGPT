from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DefaultUserAdmin
from django.utils.translation import gettext_lazy as _

from selfgpt.users.forms import UserAdminChangeForm, UserAdminCreationForm
from selfgpt.users.models import User


@admin.register(User)
class UserAdmin(DefaultUserAdmin):
    form = UserAdminChangeForm
    add_form = UserAdminCreationForm
    fieldsets = (
        (None, {"fields": ("email", "password")}),
        (_("Personal info"), {"fields": ("name", "balance")}),  # Include the balance field here
        (
            _("Permissions"),
            {
                "fields": (
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "groups",
                    "user_permissions",
                ),
            },
        ),
        (_("Important dates"), {"fields": ("last_login", "date_joined")}),
        (
            _("Custom fields"),
            {
                "fields": (
                    "uuid",
                    "website",
                    "phone",
                    "registered_at",
                    "total_completion_tokens",
                    "total_prompt_tokens",
                    "total_embedding_operations",
                    "total_image_annotations",
                )
            },
        ),
    )
    list_display = ["email", "name", "is_superuser", "balance"]
    search_fields = ["name", "email"]
    ordering = ["id"]
    readonly_fields = ("uuid", "registered_at")
