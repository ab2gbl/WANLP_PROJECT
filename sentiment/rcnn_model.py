import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .preprocces import shakl, alif
from pyarabic.araby import tokenize

# Load the RCNN model
rcnn_model = load_model('rcnn_model.h5')

# Load the tokenizer
with open('rcnn_tokenizer.json', 'r') as f:
    data = f.read()
    tokenizer = tokenizer_from_json(data)

def preprocess_text(text):
    normalized_text = shakl(text)
    normalized_text = alif(normalized_text)
    tokens = tokenize(normalized_text)
    return ' '.join(tokens)  # return string of tokens for tokenizer

def preprocess_and_vectorize(text):
    # Tokenize and convert to sequence using the loaded tokenizer
    processed_text = preprocess_text(text)  # Preprocess to tokenize and normalize
    sequence = tokenizer.texts_to_sequences([processed_text])  # Convert text to a sequence of integers
    padded_sequence = pad_sequences(sequence, maxlen=100)  # Pad sequences to the same length
    return padded_sequence

class sentiment_analysis_code():
    
    def get_tweet_sentiment(self, tweet):
        print('this one by rcnn')
        vectorized_text = preprocess_and_vectorize(tweet)
        predicted_sentiment = rcnn_model.predict(vectorized_text)
        predicted_class = np.argmax(predicted_sentiment, axis=1)[0]
        
        # Map prediction to label
        sentiment_labels = {1: 'Positive', 0: 'Negative', 2: 'Neutral'}  # Adjust as needed based on your labels
        return sentiment_labels[predicted_class]

# Example usage:
analyzer = sentiment_analysis_code()
tweet = "Your example tweet text here."
sentiment = analyzer.get_tweet_sentiment(tweet)
print(f"Predicted sentiment: {sentiment}")
