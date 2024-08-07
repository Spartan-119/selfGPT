from django.urls import path

from selfgpt.indexes.views import (
    IndexChatView,
    IndexCreateView,
    IndexDeleteView,
    IndexDetailView,
    IndexFileDeleteView,
    IndexFileUploadView,
    IndexListView,
    IndexUpdateView,
    VideoTranscriptionView,
    process_user_input,
)

app_name = "indexes"  # Namespace for the indexes app
urlpatterns = [
    path("", IndexListView.as_view(), name="list"),
    path("create/", IndexCreateView.as_view(), name="create"),
    path("<uuid:uuid>/update/", IndexUpdateView.as_view(), name="update"),
    path("<uuid:uuid>/delete/", IndexDeleteView.as_view(), name="delete"),
    path("<uuid:uuid>/", IndexDetailView.as_view(), name="detail"),
    path("<uuid:uuid>/files/upload/", IndexFileUploadView.as_view(), name="file-upload"),
    path("<uuid:uuid>/files/delete/", IndexFileDeleteView.as_view(), name="file-delete"),
    path("<uuid:uuid>/chat/", IndexChatView.as_view(), name="chat"),
    path("process-input/", process_user_input, name="process_user_input"),
    path("<uuid:uuid>/transcribe-video/", VideoTranscriptionView.as_view(), name="transcribe-video"),
]
