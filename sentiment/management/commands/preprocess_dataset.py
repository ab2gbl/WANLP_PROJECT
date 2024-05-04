from django.core.management.base import BaseCommand
from sentiment.models import Tweet  # Adjust 'your_app' to the name of your Django app

from pyarabic.araby import strip_tashkeel, normalize_hamza

def preprocess_text(text):
    text = strip_tashkeel(text)
    text = normalize_hamza(text)
    return text


class Command(BaseCommand):
    help = 'Preprocess all tweets in the database'

    def handle(self, *args, **options):
        tweets = Tweet.objects.all()
        count = 0
        for tweet in tweets:
            original_text = tweet.text
            preprocessed_text = preprocess_text(original_text)
            if original_text != preprocessed_text:
                tweet.text = preprocessed_text
                tweet.save()
                count += 1
                print(count, original_text, '->', preprocessed_text)
        self.stdout.write(self.style.SUCCESS(f'Successfully preprocessed {count} tweets.'))
