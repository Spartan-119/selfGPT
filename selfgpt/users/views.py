from django.conf import settings
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.messages.views import SuccessMessageMixin
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from django.views.generic import DetailView, RedirectView, TemplateView, UpdateView

from selfgpt.users.models import User


class UserDetailView(LoginRequiredMixin, DetailView):
    model = User
    slug_field = "uuid"
    slug_url_kwarg = "uuid"

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super().get_context_data(**kwargs)
        # Add in the stripe public key
        context["stripe_public_key"] = settings.STRIPE_PUBLIC_KEY
        return context


user_detail_view = UserDetailView.as_view()


class UserUpdateView(LoginRequiredMixin, SuccessMessageMixin, UpdateView):
    model = User
    fields = ["name", "website", "phone"]
    success_message = _("Information successfully updated")

    def get_success_url(self):
        assert self.request.user.is_authenticated  # for mypy to know that the user is authenticated
        return self.request.user.get_absolute_url()

    def get_object(self):
        return self.request.user


user_update_view = UserUpdateView.as_view()


class UserRedirectView(LoginRequiredMixin, RedirectView):
    permanent = False

    def get_redirect_url(self):
        return reverse("users:detail", kwargs={"uuid": self.request.user.uuid})


user_redirect_view = UserRedirectView.as_view()


class UserRefillView(LoginRequiredMixin, TemplateView):
    template_name = "users/user_refill.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Example: Pass Stripe public key and suggested top-up amount to the template
        context["stripe_public_key"] = settings.STRIPE_PUBLIC_KEY
        context["suggested_top_up_amount"] = 1000
        return context


user_refill_view = UserRefillView.as_view()
