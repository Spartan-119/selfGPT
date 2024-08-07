# Generated by Django 4.2.10 on 2024-02-22 12:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("users", "0007_alter_user_balance"),
    ]

    operations = [
        migrations.AlterField(
            model_name="user",
            name="name",
            field=models.CharField(blank=True, max_length=255, verbose_name="Your name"),
        ),
        migrations.AlterField(
            model_name="user",
            name="phone",
            field=models.CharField(blank=True, max_length=255, verbose_name="Your phone"),
        ),
        migrations.AlterField(
            model_name="user",
            name="website",
            field=models.CharField(blank=True, max_length=255, verbose_name="Personal or company website"),
        ),
    ]
