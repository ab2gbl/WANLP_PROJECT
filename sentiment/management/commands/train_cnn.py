from django.core.management.base import BaseCommand
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from pyarabic.araby import strip_tashkeel, tokenize, normalize_hamza

def preprocess_text(text):
    normalized_text = normalize_hamza(text)
    normalized_text = strip_tashkeel(normalized_text)
    tokens = tokenize(normalized_text)
    return ' '.join(tokens)
class Command(BaseCommand):
    help = 'Train sentiment analysis model on tweets and save it'

    def create_and_train_cnn_model(self, data_path):
        # Load data
        data = pd.read_csv(data_path)
        
        # Preprocess text data
        data["preprocessed_text"] = data["text"].apply(preprocess_text)

        # Map sentiment strings to integers
        sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}  # Add other sentiments if they exist
        data['sentiment'] = data['sentiment'].map(sentiment_mapping).astype('int')

        # Prepare tokenizer and text sequences
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(data["preprocessed_text"])
        sequences = tokenizer.texts_to_sequences(data["preprocessed_text"])
        X = pad_sequences(sequences, maxlen=100)  # Adjust maxlen based on your text data
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, data['sentiment'], test_size=0.2, random_state=42)

        # CNN model architecture
        model = Sequential([
            Embedding(input_dim=10000, output_dim=50, input_length=100),
            Conv1D(filters=128, kernel_size=5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(10, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid') if len(sentiment_mapping) == 2 else Dense(len(sentiment_mapping), activation='softmax')
        ])
        # Compile the model
        loss_function = 'binary_crossentropy' if len(sentiment_mapping) == 2 else 'sparse_categorical_crossentropy'
        model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

        # Save the model and tokenizer
        model.save('cnn_model.h5')
        tokenizer_json = tokenizer.to_json()
        with open('tokenizer.json', 'w') as file:
            file.write(tokenizer_json)

        return model, tokenizer


    def handle(self, *args, **kwargs):
        model, tokenizer = self.create_and_train_cnn_model('./final.csv')
        self.stdout.write(self.style.SUCCESS('CNN model trained and saved successfully!'))
