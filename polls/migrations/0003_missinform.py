# Generated by Django 2.1.8 on 2019-04-28 14:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0002_auto_20190421_1639'),
    ]

    operations = [
        migrations.CreateModel(
            name='MissInform',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('url', models.CharField(max_length=200)),
                ('text', models.CharField(max_length=200)),
            ],
        ),
    ]
