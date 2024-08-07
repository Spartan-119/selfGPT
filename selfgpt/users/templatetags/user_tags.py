from django import template
from django.utils.safestring import mark_safe

register = template.Library()


@register.simple_tag
def get_user_balance(user):
    if user.is_authenticated:
        return mark_safe(f"$ {user.check_balance():,.2f}")
    return "$0.00"
