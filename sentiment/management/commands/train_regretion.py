from django.core.management.base import BaseCommand
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump
from pyarabic.araby import strip_tashkeel, tokenize

def preprocess_text(text):
    normalized_text = strip_tashkeel(text)
    tokens = tokenize(normalized_text)
    return ' '.join(tokens)  # return a single string of tokens

class Command(BaseCommand):
    help = 'Train sentiment analysis model on tweets and save it'

    def create_and_train_logistic_regression_model(self, data_path):
        # Load data
        data = pd.read_csv(data_path)

        # Preprocess text data
        data["preprocessed_text"] = data["text"].apply(preprocess_text)

        # Use CountVectorizer to vectorize text data
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data["preprocessed_text"])
        vocab_dict = vectorizer.vocabulary_

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, data["sentiment"], test_size=0.2, random_state=42)

        # Create and train the Logistic Regression model
        log_reg_model = LogisticRegression(max_iter=1000)
        log_reg_model.fit(X_train, y_train)

        # Calculate accuracy on test set
        y_pred = log_reg_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save the model and vocabulary using joblib
        dump(log_reg_model, 'logistic_regression_model.joblib')
        dump(vocab_dict, 'vocab_dict.joblib')

        return log_reg_model, vocab_dict, accuracy

    def handle(self, *args, **kwargs):
        model, vocab, accuracy = self.create_and_train_logistic_regression_model('./final.csv')
        self.stdout.write(self.style.SUCCESS(f'Logistic Regression Model trained and saved successfully! Accuracy: {accuracy:.2f}'))
