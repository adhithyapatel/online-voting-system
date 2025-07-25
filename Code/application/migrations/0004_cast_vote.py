# Generated by Django 3.2.25 on 2025-03-09 20:34

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('application', '0003_delete_election_date'),
    ]

    operations = [
        migrations.CreateModel(
            name='cast_vote',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('candidatename', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='application.candidate')),
                ('election', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='application.election_name')),
                ('voter', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='application.voter')),
            ],
        ),
    ]
