import json
import os
from decimal import Decimal

from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Count
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse, reverse_lazy
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.views.generic import CreateView, DeleteView, DetailView, ListView, UpdateView, View
from django.views.generic.edit import FormMixin, FormView

from selfgpt.indexes.forms import IndexFileForm, IndexForm, UserInputForm, VideoURLForm
from selfgpt.indexes.models import Index, IndexFile, IndexQuery, TaskStatus
from selfgpt.indexes.tasks import task_transcribe_video


class IndexListView(LoginRequiredMixin, ListView):
    model = Index
    context_object_name = "indexes"
    template_name = "indexes/index_list.html"

    def get_queryset(self):
        return (
            Index.objects.filter(user=self.request.user)
            .prefetch_related("files")
            .annotate(
                file_count=Count("files"),
            )
        )


class IndexCreateView(LoginRequiredMixin, CreateView):
    model = Index
    form_class = IndexForm
    template_name = "indexes/index_form.html"
    success_url = reverse_lazy("indexes:list")

    def get(self, request, *args, **kwargs):
        # Check if user's balance is below a certain threshold
        balance_threshold = Decimal("10.00")  # Adjust the threshold as necessary
        if request.user.balance < balance_threshold:
            # Redirect to balance top-up page if balance is too low
            refill_url = reverse("users:refill", kwargs={"uuid": request.user.uuid})
            return redirect(refill_url)
        else:
            # Proceed as normal if balance is sufficient
            return super().get(request, *args, **kwargs)

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)

    def get_success_url(self):
        return reverse("indexes:detail", kwargs={"uuid": self.object.uuid})


class IndexUpdateView(LoginRequiredMixin, UpdateView):
    model = Index
    form_class = IndexForm
    template_name = "indexes/index_form.html"
    slug_field = "uuid"
    slug_url_kwarg = "uuid"
    success_url = reverse_lazy("indexes:list")

    def get_queryset(self):
        return Index.objects.filter(user=self.request.user)


class IndexDeleteView(LoginRequiredMixin, DeleteView):
    model = Index
    context_object_name = "index"
    template_name = "indexes/index_confirm_delete.html"
    slug_field = "uuid"
    slug_url_kwarg = "uuid"
    success_url = reverse_lazy("indexes:list")

    def get_queryset(self):
        return Index.objects.filter(user=self.request.user)


class IndexDetailView(LoginRequiredMixin, DetailView, FormMixin):
    model = Index
    form_class = IndexFileForm  # Assumes the form allows uploading of files
    template_name = "indexes/index_detail.html"
    slug_field = "uuid"
    slug_url_kwarg = "uuid"

    def get_queryset(self):
        return Index.objects.filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["file_form"] = self.get_form()  # Add the file upload form to the context
        return context


@method_decorator(csrf_exempt, name="dispatch")
class IndexFileUploadView(FormView):
    form_class = IndexFileForm
    template_name = "indexes/indexfile_form.html"  # Adjust if necessary

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Corrected to use 'uuid' based on your URL pattern
        context["index"] = get_object_or_404(Index, uuid=self.kwargs.get("uuid"))
        return context

    def post(self, request, create_or_update_index_from_files_task=None, *args, **kwargs):
        # Assuming the UUID is passed as a URL parameter named 'uuid'
        index_uuid = self.kwargs.get("uuid")
        index = get_object_or_404(Index, uuid=index_uuid, user=request.user)
        files = request.FILES.getlist("file")

        # Process each file in the request
        for file in files:
            IndexFile.objects.create(index=index, file=file)

        # Respond with a JSON object to signify success
        return JsonResponse({"message": "Files uploaded successfully"}, status=200)


class IndexFileDeleteView(LoginRequiredMixin, DeleteView):
    model = IndexFile
    context_object_name = "file"
    template_name = "indexes/indexfile_confirm_delete.html"

    def get_success_url(self):
        return reverse_lazy("indexes:detail", kwargs={"uuid": self.object.index.uuid})


class IndexChatView(LoginRequiredMixin, View):
    template_name = "indexes/index_chat.html"

    def get(self, request, *args, **kwargs):
        index_uuid = self.kwargs.get("uuid")
        index = get_object_or_404(Index, uuid=index_uuid, user=request.user)
        form = UserInputForm()
        context = {"form": form, "index": index, "user_name": request.user.name}
        return render(request, self.template_name, context)


@require_POST
@csrf_exempt  # Remember to handle CSRF token properly in AJAX requests
def process_user_input(request, *args, **kwargs):
    # Parse the JSON data from request.body
    data = json.loads(request.body.decode("utf-8"))

    # Get user input (prompt) and Index UUID from the parsed data
    user_input = data.get("user_input")
    index_uuid = data.get("index_uuid")
    index_query_uuid = data.get("index_query_uuid")
    use_context = data.get("use_context")

    index = get_object_or_404(Index, uuid=index_uuid)

    if index_query_uuid is None or len(index_query_uuid) == 0:
        index_query = IndexQuery.objects.create(
            index=index,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
            ],
        )
    else:
        index_query = get_object_or_404(IndexQuery, uuid=index_query_uuid)

    result = index.send_query_to_openai(query=user_input, query_uuid=index_query.uuid, use_context=use_context)
    return JsonResponse({"response": result, "index_query": index_query.uuid})


class VideoTranscriptionView(LoginRequiredMixin, FormView):
    template_name = "indexes/video_transcription_form.html"
    form_class = VideoURLForm

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Corrected to use 'uuid' based on your URL pattern
        context["index"] = get_object_or_404(Index, uuid=self.kwargs.get("uuid"))
        return context

    def form_valid(self, form):
        index_uuid = self.kwargs.get("uuid")
        user_uuid = self.request.user.uuid
        video_url = form.cleaned_data["video_url"]

        if not os.getenv("USE_DOCKER") or os.getenv("USE_DOCKER") == "False":
            task_transcribe_video(index_uuid, user_uuid, video_url)
        else:
            task = task_transcribe_video.apply_async(args=[index_uuid, user_uuid, video_url])
            TaskStatus.objects.create(
                user=self.request.user, task_id=task.id, task_name="Transcribing video", status="PENDING"
            )

        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy("indexes:detail", kwargs={"uuid": self.kwargs.get("uuid")})
