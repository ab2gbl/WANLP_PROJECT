# train_arabert.py

from django.core.management.base import BaseCommand
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pyarabic.araby import strip_tashkeel

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([0 if label == 'negative' else 1 for label in self.labels])[idx]
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class Command(BaseCommand):
    help = 'Train a sentiment analysis model using AraBERT'

    def add_arguments(self, parser):
        parser.add_argument('--data_path', type=str, help='CSV file path for the training data')

    def preprocess_text(self, text):
        # Normalize text by removing diacritics
        text = strip_tashkeel(text)
        return text

    def handle(self, *args, **options):
        data_path = options.get('data_path', './final.csv')  # Default path if not provided
        data = pd.read_csv(data_path)
        
        # Apply preprocessing to the text
        data['text'] = data['text'].apply(self.preprocess_text)

        train_texts, val_texts, train_labels, val_labels = train_test_split(data['text'], data['sentiment'], test_size=0.1)
        
        model_name = "aubmindlab/bert-base-arabertv02"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=128)
        val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=128)

        train_dataset = Dataset(train_encodings, train_labels)
        val_dataset = Dataset(val_encodings, val_labels)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        model.save_pretrained('./arabert_sentiment_model')
        tokenizer.save_pretrained('./arabert_sentiment_model')

        self.stdout.write(self.style.SUCCESS('AraBERT model trained and saved successfully!'))
