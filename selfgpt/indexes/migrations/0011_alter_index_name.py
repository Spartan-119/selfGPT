# Generated by Django 4.2.10 on 2024-02-22 14:32

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("indexes", "0010_index_image_annotations"),
    ]

    operations = [
        migrations.AlterField(
            model_name="index",
            name="name",
            field=models.CharField(
                max_length=150,
                validators=[
                    django.core.validators.MinLengthValidator(2),
                    django.core.validators.MaxLengthValidator(150),
                ],
            ),
        ),
    ]
