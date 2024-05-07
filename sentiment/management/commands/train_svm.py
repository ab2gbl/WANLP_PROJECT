# sentiment/management/commands/train_svm.py

from django.core.management.base import BaseCommand
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from joblib import dump
from pyarabic.araby import strip_tashkeel, tokenize

def preprocess_text(text):
    normalized_text = strip_tashkeel(text)
    tokens = tokenize(normalized_text)
    return ' '.join(tokens)  # return a single string of tokens

class Command(BaseCommand):
    help = 'Train a sentiment analysis model on tweets using SVM and save it'

    def create_and_train_svm_model(self, data_path):
        print("start")
        # Load data
        data = pd.read_csv(data_path)
        
        # Preprocess text data
        data["preprocessed_text"] = data["text"].apply(preprocess_text)
        
        # Use TfidfVectorizer to vectorize text data
        vectorizer = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.75)
        X = vectorizer.fit_transform(data["preprocessed_text"])
        vocab_dict = vectorizer.vocabulary_
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, data["sentiment"], test_size=0.2, random_state=42)
        
        # Create and train the LinearSVC model
        svm_model = LinearSVC(random_state=42)
        svm_model.fit(X_train, y_train)
        
        # Calculate accuracy on test set
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save the model and vocabulary using joblib
        dump(svm_model, 'svm_model.joblib')
        dump(vocab_dict, 'svm_vocab_dict.joblib')
        
        return svm_model, vocab_dict, accuracy

    def handle(self, *args, **kwargs):
        model, vocab, accuracy = self.create_and_train_svm_model('./final.csv')
        self.stdout.write(self.style.SUCCESS(f'SVM model trained and saved successfully! Accuracy: {accuracy:.2f}'))
