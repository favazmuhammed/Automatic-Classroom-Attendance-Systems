# Generated by Django 4.1.2 on 2022-11-15 11:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("Student", "0006_takes_attendace_percentage"),
    ]

    operations = [
        migrations.AlterField(
            model_name="teaches",
            name="num_students",
            field=models.IntegerField(null=True),
        ),
    ]