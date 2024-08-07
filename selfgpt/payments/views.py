import json
import logging

import stripe
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpResponse, JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from selfgpt.users.helpers import CurrencyConverter

User = get_user_model()
logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name="dispatch")
class CreateCheckoutSessionView(View):
    @staticmethod
    def post(request, *args, **kwargs):
        stripe.api_key = settings.STRIPE_SECRET_KEY

        # Parse the request body to get the amount
        data = json.loads(request.body)
        amount = data.get("amount", 20)

        if amount is None:
            return JsonResponse({"error": "Please enter a valid amount (integer)"}, status=400)

        elif int(amount) < 20:
            return JsonResponse({"error": "Amount must be at least 20 USD"}, status=400)

        # Convert amount to cents for Stripe
        amount_in_cents = int(float(amount) * 100)

        try:
            checkout_session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[
                    {
                        "price_data": {
                            "currency": "usd",
                            "product_data": {
                                "name": "selfGPT balance",
                                "metadata": {"user_uuid": request.user.uuid, "user_email": request.user.email},
                            },
                            "unit_amount": amount_in_cents,
                        },
                        "quantity": 1,
                    }
                ],
                customer_email=request.user.email,
                mode="payment",
                success_url=request.build_absolute_uri(f"/users/{request.user.uuid}/"),
                cancel_url=request.build_absolute_uri(f"/users/{request.user.uuid}/"),
                metadata={"user_email": request.user.email if request.user.is_authenticated else ""},
            )
            return JsonResponse({"id": checkout_session.id})
        except Exception as e:
            logger.error(f"CreateCheckoutSessionView.post has error {e}")
            return JsonResponse({"error": str(e)})


@csrf_exempt
@require_POST
def stripe_webhook(request):
    stripe.api_key = settings.STRIPE_SECRET_KEY

    payload = request.body
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, settings.STRIPE_WEBHOOK_SECRET)
    except ValueError:
        logger.error(f"stripe_webhook error with {request}")
        return HttpResponse(status=400)
    except stripe.error.SignatureVerificationError:
        logger.error("stripe_webhook error with stripe.error.SignatureVerificationError")
        return HttpResponse(status=400)

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        handle_checkout_session(session)
    else:
        logger.info(f"stripe_webhook unhandled event type {event['type']}")

    return HttpResponse(status=200)


def handle_checkout_session(session):
    email = session.get("metadata", {}).get("user_email")
    amount_paid = CurrencyConverter().cents_to_microdollars(session.get("amount_total"))
    try:
        user = User.objects.get(email=email)
        user.add_to_balance(amount_paid)
        logger.info(f"handle_checkout_session - added {amount_paid} credits to user {email}'s account.")
    except ObjectDoesNotExist:
        logger.error(f"handle_checkout_session - user with email {email} not found. session is {session}")
    except Exception as e:
        logger.error(f"handle_checkout_session - error updating user credits: {e}")
