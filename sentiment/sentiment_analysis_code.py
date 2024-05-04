import re
from textblob import TextBlob
from nltk.stem.wordnet import WordNetLemmatizer 
import itertools
import numpy as np
import nltk
from .models import create_and_train_svm_model
from joblib import load

from pyarabic.araby import strip_tashkeel, tokenize,normalize_hamza 
from collections import Counter

# Load the model and vocabulary from the saved files
log_reg_model = load('logistic_regression_model.joblib')
vocab_dict = load('vocab_dict.joblib')


def preprocess_text(text):
    normalized_text = strip_tashkeel(text)
    normalized_text = normalize_hamza (text)
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
    '''
    lem = WordNetLemmatizer()

    def cleaning(self, text):
        txt = str(text)
        txt = re.sub(r"http\S+", "", txt)
        if len(txt) == 0:
            return 'no text'
        else:
            txt = txt.split()
            index = 0
            for j in range(len(txt)):
                if txt[j][0] == '@':
                    index = j
            txt = np.delete(txt, index)
            if len(txt) == 0:
                return 'no text'
            else:
                words = txt[0]
                for k in range(len(txt)-1):
                    words+= " " + txt[k+1]
                txt = words
                txt = re.sub(r'[^\w]', ' ', txt)
                if len(txt) == 0:
                    return 'no text'
                else:
                    txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))
                    txt = txt.replace("'", "")
                    txt = nltk.tokenize.word_tokenize(txt)
                    #data.content[i] = [w for w in data.content[i] if not w in stopset]
                    for j in range(len(txt)):
                        txt[j] = self.lem.lemmatize(txt[j], "v")
                    if len(txt) == 0:
                        return 'no text'
                    else:
                        return txt
    '''
    def get_tweet_sentiment(self, tweet):
        '''
        cleaned_tweet = ' '.join(self.cleaning(tweet))  # Clean the tweet
        # Load the trained SVM model and vectorizer
        svm_model, vectorizer = create_and_train_svm_model("../AJGT.xlsx")
        # Preprocess the tweet text
        preprocessed_tweet = vectorizer.transform([cleaned_tweet])
        # Predict sentiment using the SVM model
        prediction = svm_model.predict(preprocessed_tweet)
        #cleaning of tweet
        tweet = ' '.join(self.cleaning(tweet))
        if prediction== 'Positive':
                return 'Positive'
        elif prediction== 'Negative':
                return 'Negative'
        else:
                return 'Neutral'  
        '''
        #cleaned_tweet = ' '.join(self.cleaning(tweet))
        
        vectorized_text,tweet = preprocess_and_vectorize(tweet, vocab_dict)
        print (tweet)
        predicted_sentiment = log_reg_model.predict([vectorized_text])
        if predicted_sentiment== 'positive':
                return 'Positive'
        elif predicted_sentiment== 'negative':
                return 'Negative'
        else:
                return 'Neutral'