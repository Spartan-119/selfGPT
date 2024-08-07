from django.urls import path

from selfgpt.users.views import user_detail_view, user_redirect_view, user_refill_view, user_update_view

app_name = "users"
urlpatterns = [
    path("~redirect/", view=user_redirect_view, name="redirect"),
    path("~update/", view=user_update_view, name="update"),
    path("<uuid:uuid>/", view=user_detail_view, name="detail"),
    path("<uuid:uuid>/refill/", view=user_refill_view, name="refill"),
]
