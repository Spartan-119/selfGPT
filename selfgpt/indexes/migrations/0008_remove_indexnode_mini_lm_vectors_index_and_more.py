# Generated by Django 4.2.10 on 2024-02-18 07:39

from django.db import migrations
import pgvector.django


class Migration(migrations.Migration):

    dependencies = [
        ("indexes", "0007_remove_indexquery_completion_tokens_and_more"),
    ]

    operations = [
        migrations.RemoveIndex(
            model_name="indexnode",
            name="mini_lm_vectors_index",
        ),
        migrations.AddIndex(
            model_name="indexnode",
            index=pgvector.django.HnswIndex(
                ef_construction=64,
                fields=["embedding_mini_lm_l6_v2"],
                m=16,
                name="mini_lm_vectors_index",
                opclasses=["vector_cosine_ops"],
            ),
        ),
    ]
