# Generated by Django 4.1.2 on 2022-11-15 06:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("Student", "0005_rename_instructor_id_teaches_mail"),
    ]

    operations = [
        migrations.AddField(
            model_name="takes",
            name="attendace_percentage",
            field=models.DecimalField(decimal_places=2, default=100, max_digits=5),
        ),
    ]