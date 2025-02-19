# Generated by Django 4.2.10 on 2024-03-11 08:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("indexes", "0012_indexfile_type"),
    ]

    operations = [
        migrations.AlterField(
            model_name="indexfile",
            name="type",
            field=models.CharField(
                choices=[("text", "Text"), ("video", "Video")],
                default="text",
                help_text="The type of content stored in this file.",
                max_length=5,
            ),
        ),
    ]
