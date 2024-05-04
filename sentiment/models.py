
from django.db import models 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from pyarabic.araby import strip_tashkeel,tokenize





def preprocess_text(text):
    normalized_text = strip_tashkeel(text)
    tokens = tokenize(normalized_text)
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def create_and_train_svm_model(data_path):
    # Load data
    data = pd.read_excel(data_path)

    # Preprocess text data
    data["preprocessed_text"] = data["Feed"].apply(preprocess_text)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data["preprocessed_text"], data["Sentiment"], test_size=0.2, random_state=42)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)

    # Train the vectorizer on training data
    vectorizer.fit(X_train)

    # Transform data into TF-IDF vectors
    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Create and train the SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_tfidf, y_train)

    return svm_model, vectorizer


class Tweet(models.Model):
    text = models.TextField()
    
    def __str__(self):
        return self.text