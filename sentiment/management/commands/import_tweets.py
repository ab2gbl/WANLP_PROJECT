# import_tweets.py

import csv
from django.core.management.base import BaseCommand
from sentiment.models import Tweet  # Change 'your_app_name' to the name of your app

class Command(BaseCommand):
    help = 'Load a list of tweets from a CSV file into the database'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='The CSV file to import.')

    def handle(self, *args, **options):
        file_path = options['csv_file']
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip the header row
            count = 0
            for row in reader:
                # Assuming the tweet text is in the first column
                Tweet.objects.create(text=row[1])
                count +=1
                self.stdout.write(self.style.SUCCESS(f'Imported tweet: "{count} "'))
