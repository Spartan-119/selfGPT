from django.urls import path

from selfgpt.payments.views import CreateCheckoutSessionView, stripe_webhook

app_name = "payments"  # Namespace for the payments app
urlpatterns = [
    path("create-checkout-session/", CreateCheckoutSessionView.as_view(), name="create-checkout-session"),
    path("webhook/", stripe_webhook, name="stripe-webhook"),
]
