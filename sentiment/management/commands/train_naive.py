from django.core.management.base import BaseCommand
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from pyarabic.araby import strip_tashkeel, tokenize

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    normalized_text = strip_tashkeel(text)
    tokens = tokenize(normalized_text)
    return ' '.join(tokens)

class Command(BaseCommand):
    help = 'Train a Naive Bayes sentiment analysis model on tweets and save it'

    def create_and_train_naive_bayes_model(self, data_path):
        # Load data
        data = pd.read_csv(data_path)

        # Preprocess text data
        data["preprocessed_text"] = data["text"].apply(preprocess_text)

        # Use CountVectorizer to vectorize text data
        vectorizer = CountVectorizer(max_features=5000)  # Limit features to manage memory
        X = vectorizer.fit_transform(data["preprocessed_text"])
        vocab_dict = vectorizer.vocabulary_

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, data["sentiment"], test_size=0.2, random_state=42, stratify=data["sentiment"])

        # Create and train the Naive Bayes model
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)

        # Calculate accuracy and more detailed performance metrics
        y_pred = nb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Save the model and vocabulary using joblib
        dump(nb_model, 'naive_bayes_model.joblib')
        dump(vocab_dict, 'naive_bayes_vocab_dict.joblib')

        return nb_model, vocab_dict, accuracy, report

    def handle(self, *args, **kwargs):
        model, vocab, accuracy, report = self.create_and_train_naive_bayes_model('./final.csv')
        self.stdout.write(self.style.SUCCESS(f'Naive Bayes model trained and saved successfully! Accuracy: {accuracy:.2f}'))
        self.stdout.write(report)

