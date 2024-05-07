import re
from textblob import TextBlob
from nltk.stem.wordnet import WordNetLemmatizer 
import itertools
import numpy as np
import nltk
from .models import create_and_train_svm_model
from joblib import load

from pyarabic.araby import tokenize
from collections import Counter
from .preprocces import shakl, alif

# Load the model and vocabulary from the saved files
log_reg_model = load('naive_bayes_model.joblib')
vocab_dict = load('naive_bayes_vocab_dict.joblib')


def preprocess_text(text):
    normalized_text = shakl(text)
    normalized_text = alif (text)
    tokens = tokenize(normalized_text)
    return tokens  # return list of tokens


def preprocess_and_vectorize(text, vocab_dict):
    # Assume preprocess_text is defined as before to tokenize and normalize text
    tokens = preprocess_text(text)
    vector = [0] * len(vocab_dict)
    token_counts = Counter(tokens)
    for token, count in token_counts.items():
        if token in vocab_dict:
            vector[vocab_dict[token]] = count
    tweet=' '.join(tokens)
    return vector,tweet

class sentiment_analysis_code():
    
    def get_tweet_sentiment(self, tweet):
       
        vectorized_text,tweet = preprocess_and_vectorize(tweet, vocab_dict)
        print (tweet)
        predicted_sentiment = log_reg_model.predict([vectorized_text])
        if predicted_sentiment== 'positive':
                return 'Positive'
        elif predicted_sentiment== 'negative':
                return 'Negative'
        else:
                return 'Neutral'