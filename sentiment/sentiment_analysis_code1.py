from camel_tools.sentiment import SentimentAnalyzer
from collections import Counter
from .preprocces import shakl, alif
from pyarabic.araby import normalize_hamza
sa = SentimentAnalyzer("CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment")

def preprocess_text(text):
    normalized_text = shakl(text)
    normalized_text = normalize_hamza (text)
    return normalized_text  # return list of tokens


class sentiment_analysis_code():
    
    def get_tweet_sentiment(self, tweet):
        sa.predict("أنا أحبك جدا و شكرا" )
        
        tweet1 = preprocess_text(tweet)
        print (tweet)
        predicted_sentiment = sa.predict_sentence(tweet1 )
        print(predicted_sentiment)
        if predicted_sentiment== 'positive':
                return 'Positive'
        elif predicted_sentiment== 'negative':
                return 'Negative'
        elif predicted_sentiment== 'neutral':
                return 'Neutral'