import logging
import os
import shutil
from uuid import uuid4

import fitz
import magic
from django.conf import settings
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.core.validators import FileExtensionValidator, MaxLengthValidator, MinLengthValidator
from django.db import models
from django.shortcuts import get_object_or_404
from django.utils.crypto import get_random_string
from django.utils.text import get_valid_filename
from openai import OpenAI
from pgvector.django import CosineDistance, HnswIndex, VectorField

from selfgpt.indexes.helpers import (
    bytes_to_string,
    create_offset_sentence_chunks,
    encode_resize_image,
    preprocess_text,
    vectorize_chunks,
)
from selfgpt.indexes.tasks import task_vectorize_file
from selfgpt.users.models import User

logger = logging.getLogger(__name__)


class TimeStampedModel(models.Model):
    """
    An abstract base class model that provides self-updating
    'created' and 'modified' fields and UUID
    """

    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)
    uuid = models.UUIDField(db_index=True, default=uuid4, editable=False)

    class Meta:
        abstract = True


class Index(TimeStampedModel):
    user = models.ForeignKey(
        User, related_name="indexes", on_delete=models.CASCADE, help_text="Which user owns the index"
    )

    name = models.CharField(
        validators=[MinLengthValidator(2), MaxLengthValidator(150)], max_length=150, blank=False, null=False
    )

    completion_tokens = models.IntegerField(
        default=0, help_text="Number of tokens generated in response to the user's prompt."
    )
    prompt_tokens = models.IntegerField(default=0, help_text="Number of tokens used in the user's prompt.")
    embedding_operations = models.IntegerField(default=0, help_text="Number of vectorization operations performed.")
    image_annotations = models.IntegerField(default=0, help_text="Number of image annotations performed.")

    def get_similar_content(self, query, top_k=3, query_uuid=None):
        prompt_vector = vectorize_chunks([query])[0]

        # Similarity search by pgvector fields
        similar_nodes = (
            IndexNode.objects.filter(index_file__index=self).annotate(
                distance=CosineDistance("embedding_mini_lm_l6_v2", prompt_vector)
            )
            # .filter(distance__lte=1)  # Filter some not relevant results
            .order_by("distance")[:top_k]
        )

        # Postgresql full text search
        search_vector = SearchVector("content", weight="A") + SearchVector("type", weight="B")
        search_query = SearchQuery(query)
        exact_results = (
            IndexNode.objects.annotate(rank=SearchRank(search_vector, search_query))
            .filter(rank__gte=0.3)
            .order_by("-rank")
        )[:top_k]

        # Extract content from nodes and concatenate into a single string
        combined_results = similar_nodes.union(exact_results)

        context = "\n".join(list({node.content for node in combined_results}))

        if query_uuid:
            query_index = get_object_or_404(IndexQuery, uuid=query_uuid)
            query_index.set_context_nodes([node.id for node in combined_results])

        self.update_usage(embedding_operations=1)
        return context

    def create_prompt_with_context(self, query, query_uuid=None):
        # Retrieve context as a single string
        context = self.get_similar_content(query, query_uuid=query_uuid)

        # If there is no appropriate context, return the query itself
        if context is None or len(context) == 0:
            return query

        # Construct the prompt with context and query
        prompt = (
            f"The following are key information points:\n{context}\n\n"
            f"Based on this information, {query}\n\nUse the language used in the main question."
        )

        return prompt

    def send_query_to_openai(self, query, query_uuid, use_context=True):
        prompt = query
        if use_context:
            prompt = self.create_prompt_with_context(query, query_uuid)
        index_query = get_object_or_404(IndexQuery, uuid=query_uuid)
        index_query.add_message(role="user", content=prompt)

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=index_query.get_messages(),
            frequency_penalty=0.0,
            presence_penalty=0.0,
            temperature=1,
        )

        text_response = response.choices[0].message.content.strip()
        index_query.add_message(role="assistant", content=text_response)

        # Update the IndexQuery instance with the new token counts
        self.update_usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        return text_response

    def update_usage(self, prompt_tokens=0, completion_tokens=0, embedding_operations=0, image_annotations=0):
        """
        Updates the token usage metrics for the query.
        Assumes that prompt_tokens and completion_tokens are integers.
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.embedding_operations += embedding_operations
        self.image_annotations += image_annotations
        self.user.update_global_stats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            embedding_operations=embedding_operations,
            image_annotations=image_annotations,
        )
        self.save()

    def delete(self, *args, **kwargs):
        original_folder_path = os.path.join(settings.MEDIA_ROOT, f"indexes/{self.uuid}")
        deleted_folder_path = os.path.join(settings.MEDIA_ROOT, f"indexes/deleted_{self.uuid}")

        # Check if original folder exists
        if os.path.exists(original_folder_path):
            # Create the root folder for deleted indexes if it doesn't exist
            os.makedirs(os.path.dirname(deleted_folder_path), exist_ok=True)
            # Move the original folder to the new location with "deleted_" prefix
            shutil.move(original_folder_path, deleted_folder_path)

        super().delete(*args, **kwargs)  # Call the "real" delete() method.

    class Meta:
        verbose_name = "Index"
        verbose_name_plural = "Indexes"

        ordering = ["-created"]
        get_latest_by = "-created"

    def __str__(self):
        return f"name: {self.name} / UUID: {self.uuid}"


def get_index_file_location(instance, filename):
    # Sanitize the filename to ensure it's safe for use in the filesystem
    sanitized_filename = get_valid_filename(filename)
    # Extract the extension and filename without extension
    name, ext = os.path.splitext(sanitized_filename)
    # Generate a random string
    random_str = get_random_string(length=5)
    # Construct a new filename using the original name, a random string, and the file extension
    new_filename = f"{name}_{random_str}{ext}"
    # Construct the path using the Index UUID and the new, unique filename
    return f"indexes/{instance.index.uuid}/files/{new_filename}"


class IndexFile(TimeStampedModel):
    TYPE_CHOICES = (
        ("text", "Text"),
        ("video", "Video"),
    )

    index = models.ForeignKey(
        Index, related_name="files", on_delete=models.CASCADE, help_text="The index this file is associated with"
    )
    file = models.FileField(
        upload_to=get_index_file_location,
        validators=[FileExtensionValidator(allowed_extensions=["pdf", "txt"])],
        help_text="The file uploaded for this index",
    )
    original_filename = models.CharField(
        max_length=255, editable=False, help_text="The original file name uploaded by the user"
    )
    type = models.CharField(
        max_length=5, choices=TYPE_CHOICES, default="text", help_text="The type of content stored in this file."
    )

    def save(self, *args, **kwargs):
        if not self.id:  # File is new
            # Sanitize the filename to ensure it's safe to use
            self.original_filename = get_valid_filename(self.file.name)
        super().save(*args, **kwargs)

        mime_type = magic.from_file(self.file.path, mime=True)
        if not os.getenv("USE_DOCKER") or os.getenv("USE_DOCKER") == "False":
            self.vectorize_file(mime_type)
        else:
            task_vectorize_file.apply_async(args=[self.uuid, mime_type], countdown=1)

    def vectorize_file(self, mime_type):
        if mime_type == "application/pdf":
            self.vectorize_pdf_file()
        elif mime_type.startswith("text/"):
            self.vectorize_text_file()

    def vectorize_text_file(self):
        with self.file.open("r") as file:
            text = file.read()

        if isinstance(text, bytes):  # noqa
            text = bytes_to_string(text)

        # Create chunks of the document
        chunks = create_offset_sentence_chunks(text)
        # Vectorize these chunks
        embeddings = vectorize_chunks(chunks)

        prefix = ""
        if self.type == "video":
            prefix = "video_part: "

        # Prepare a list of IndexNode instances without saving them to the DB
        nodes = [
            IndexNode(index_file=self, content=f"{prefix}{chunk}", embedding_mini_lm_l6_v2=embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Use bulk_create to create all instances in a single query
        IndexNode.objects.bulk_create(nodes)
        self.index.update_usage(embedding_operations=len(nodes))

    def vectorize_pdf_file(self):
        pdf_path = self.file.path
        doc = fitz.open(pdf_path)
        all_text = []
        image_annotations = []

        for page_num, page in enumerate(doc):
            # Extract text from each page and append it to all_text
            page_text = page.get_text("text")
            all_text.append(page_text)

            # Extract images
            for image_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_width, image_height = base_image["width"], base_image["height"]

                if image_width > 100 and image_height > 100:
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_path = os.path.join(
                        settings.MEDIA_ROOT,
                        get_index_file_location(self, f"image_{image_index + 1}_page_{page_num + 1}.{image_ext}"),
                    )

                    with open(image_path, "wb+") as image_file:
                        image_file.write(image_bytes)

                    image_annotations.append(self.describe_image(image_path))

        doc.close()

        # Combine text from all pages
        combined_text = " ".join(all_text)

        # Check if the combined text is in bytes, convert it to string if true
        if isinstance(combined_text, bytes):  # noqa
            combined_text = bytes_to_string(combined_text)

        # Create chunks of the document
        content_per_pages = [f"page {num + 1}: {preprocess_text(page)}" for num, page in enumerate(all_text)]
        final_annotations = [f"image {num + 1}: {info}" for num, info in enumerate(image_annotations)]
        chunks = create_offset_sentence_chunks(combined_text)
        chunks += content_per_pages

        # Vectorize the chunks
        embeddings = vectorize_chunks(chunks)
        embeddings_from_images = vectorize_chunks(final_annotations)

        # Prepare a list of IndexNode instances without saving them to the DB
        nodes = [
            IndexNode(index_file=self, content=chunk, embedding_mini_lm_l6_v2=embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        image_nodes = [
            IndexNode(index_file=self, content=chunk, embedding_mini_lm_l6_v2=embedding, type="image")
            for chunk, embedding in zip(final_annotations, embeddings_from_images)
        ]
        nodes += image_nodes

        # Use bulk_create to create all instances in a single query
        IndexNode.objects.bulk_create(nodes)
        # Update usage assuming a method exists on the index to track embedding operations
        self.index.update_usage(embedding_operations=len(nodes))

    def describe_image(self, image_path):
        base64_image = encode_resize_image(image_path)
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=300,
        )

        text_response = response.choices[0].message.content.strip()

        # Update the IndexQuery instance with the new token counts
        self.index.update_usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            image_annotations=1,
        )
        return text_response

    class Meta:
        verbose_name = "Index File"
        verbose_name_plural = "Index Files"
        ordering = ["-created"]

    def __str__(self):
        return f"File for index: {self.index.name} / Uploaded at: {self.created}"


class IndexNode(TimeStampedModel):
    TYPE_CHOICES = (
        ("text", "Text"),
        ("image", "Image"),
    )

    index_file = models.ForeignKey(
        IndexFile,
        related_name="nodes",
        on_delete=models.CASCADE,
        help_text="The index file this node is associated with",
    )
    content = models.TextField(help_text="Content of the node, consisting of few sentences from the document.")
    embedding_mini_lm_l6_v2 = VectorField(dimensions=384, help_text="384 sized embedding for MiniLM L6 V2.")

    type = models.CharField(
        max_length=5,
        choices=TYPE_CHOICES,
        default="text",
        help_text="The type of content stored in this node (text or image).",
    )

    class Meta:
        verbose_name = "Index Node"
        verbose_name_plural = "Index Nodes"
        ordering = ["-created"]

        indexes = [
            HnswIndex(
                name="mini_lm_vectors_index",
                fields=["embedding_mini_lm_l6_v2"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            )
        ]

    def __str__(self):
        return f"Node for index file: {self.index_file.original_filename} / Created at: {self.created}"


class IndexQuery(TimeStampedModel):
    index = models.ForeignKey(
        Index,
        related_name="queries",
        on_delete=models.CASCADE,
        help_text="Reference to the Index related to this query.",
    )
    messages = models.JSONField(help_text="List of messages exchanged with the assistant.", default=list)
    context_node_ids = models.JSONField(help_text="List of IndexNode IDs used as context for the query.", default=list)

    class Meta:
        verbose_name = "Index Query"
        verbose_name_plural = "Index Queries"
        ordering = ["-created"]

    def __str__(self):
        return f"Query for index: {self.index.name} / Created at: {self.created}"

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self.save()

    def get_messages(self):
        return self.messages

    def get_assistant_responses(self):
        return [msg for msg in self.messages if msg["role"] == "assistant"]

    def get_user_prompts(self):
        return [msg for msg in self.messages if msg["role"] == "user"]

    def set_context_nodes(self, node_ids):
        self.context_node_ids += node_ids
        self.save()

    def get_context_nodes(self):
        return self.context_node_ids


class TaskStatus(TimeStampedModel):
    STATUS_CHOICES = (
        ("PENDING", "Pending"),
        ("STARTED", "Started"),
        ("SUCCESS", "Success"),
        ("FAILURE", "Failure"),
        ("RETRY", "Retry"),
        ("REVOKED", "Revoked"),
    )

    user = models.ForeignKey(
        User, related_name="tasks", on_delete=models.CASCADE, help_text="User who initiated the task"
    )
    task_id = models.CharField(max_length=255, unique=True, help_text="Celery task ID")
    task_name = models.CharField(max_length=255, help_text="Name or type of the task")
    status = models.CharField(
        max_length=50, choices=STATUS_CHOICES, default="PENDING", help_text="Current status of the task"
    )
    result = models.TextField(blank=True, null=True, help_text="Task result or error message")

    class Meta:
        verbose_name = "Task Status"
        verbose_name_plural = "Task Statuses"
        ordering = ["-created"]

    def __str__(self):
        return f"{self.task_name} ({self.status}) - {self.user.username}"
