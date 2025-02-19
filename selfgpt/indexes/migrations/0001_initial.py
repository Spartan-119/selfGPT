# Generated by Django 4.2.10 on 2024-02-14 10:20

from django.conf import settings
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion
import selfgpt.indexes.models
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Index",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("modified", models.DateTimeField(auto_now=True)),
                ("uuid", models.UUIDField(db_index=True, default=uuid.uuid4, editable=False)),
                (
                    "name",
                    models.CharField(
                        help_text="The name of the index",
                        max_length=150,
                        validators=[
                            django.core.validators.MinLengthValidator(2),
                            django.core.validators.MaxLengthValidator(150),
                        ],
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        help_text="Which user owns the index",
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="indexes",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "Index",
                "verbose_name_plural": "Indexes",
                "ordering": ["-created"],
                "get_latest_by": "-created",
            },
        ),
        migrations.CreateModel(
            name="IndexFile",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created", models.DateTimeField(auto_now_add=True)),
                ("modified", models.DateTimeField(auto_now=True)),
                ("uuid", models.UUIDField(db_index=True, default=uuid.uuid4, editable=False)),
                (
                    "file",
                    models.FileField(
                        help_text="The file uploaded for this index",
                        upload_to=selfgpt.indexes.models.get_index_file_location,
                        validators=[
                            django.core.validators.FileExtensionValidator(
                                allowed_extensions=["pdf", "doc", "docx", "txt", "xlsx", "csv"]
                            )
                        ],
                    ),
                ),
                (
                    "original_filename",
                    models.CharField(
                        editable=False, help_text="The original file name uploaded by the user", max_length=255
                    ),
                ),
                (
                    "index",
                    models.ForeignKey(
                        help_text="The index this file is associated with",
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="files",
                        to="indexes.index",
                    ),
                ),
            ],
            options={
                "verbose_name": "Index File",
                "verbose_name_plural": "Index Files",
                "ordering": ["-created"],
            },
        ),
    ]
