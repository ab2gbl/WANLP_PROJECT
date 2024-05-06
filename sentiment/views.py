
from django.shortcuts import render, redirect, HttpResponse
from .forms import Sentiment_Typed_Tweet_analyse_form
from .sentiment_analysis_code1 import sentiment_analysis_code
from .forms import Sentiment_Imported_Tweet_analyse_form
from .tweepy_sentiment import Import_tweet_sentiment
from collections import Counter

from .models import Tweet
from pyarabic.araby import strip_tashkeel, normalize_hamza
from .preprocces import shakl, alif

def preprocess_text(text):
    text = shakl(text)
    text = alif(text)
    return text

    
def sentiment_analysis(request):
    return render(request, 'home/sentiment.html')

def sentiment_analysis_type(request):
    if request.method == 'POST':
        form = Sentiment_Typed_Tweet_analyse_form(request.POST)
        analyse = sentiment_analysis_code()
        if form.is_valid():
            tweet = form.cleaned_data['sentiment_typed_tweet']
            sentiment = analyse.get_tweet_sentiment(tweet)
            args = {'tweet':tweet, 'sentiment':sentiment}
            return render(request, 'home/sentiment_type_result.html', args)

    else:
        form = Sentiment_Typed_Tweet_analyse_form()
        return render(request, 'home/sentiment_type.html')

def sentiment_analysis_import(request):
    if request.method == 'POST':
        form = Sentiment_Imported_Tweet_analyse_form(request.POST)
        tweet_text = Import_tweet_sentiment()
        analyse = sentiment_analysis_code()

        if form.is_valid():
            handle = form.cleaned_data['sentiment_imported_tweet']

            if handle[0]=='#':
                list_of_tweets = tweet_text.get_hashtag(handle)
                list_of_tweets_and_sentiments = []
                for i in list_of_tweets:
                    list_of_tweets_and_sentiments.append((i,analyse.get_tweet_sentiment(i)))
                args = {'list_of_tweets_and_sentiments':list_of_tweets_and_sentiments, 'handle':handle}
                return render(request, 'home/sentiment_import_result_hashtag.html', args)

            list_of_tweets = tweet_text.get_tweets(handle)
            list_of_tweets_and_sentiments = []
            if handle[0]!='@':
                handle = str('@'+handle)
            for i in list_of_tweets:
                list_of_tweets_and_sentiments.append((i,analyse.get_tweet_sentiment(i)))
            args = {'list_of_tweets_and_sentiments':list_of_tweets_and_sentiments, 'handle':handle}
            return render(request, 'home/sentiment_import_result.html', args)

    else:
        form = Sentiment_Imported_Tweet_analyse_form()
        return render(request, 'home/sentiment_import.html')



def search_tweets(request):
    query = request.GET.get('q', '')
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    total_sentiments = 0 
    
    analyse = sentiment_analysis_code()
    if query:
        processed_query = preprocess_text(query)
        matching_tweets = Tweet.objects.filter(text__icontains=processed_query)
        sentiments = [analyse.get_tweet_sentiment(preprocess_text(tweet.text)) for tweet in matching_tweets]
        sentiment_counts = Counter(sentiments)
        total_sentiments = sum(sentiment_counts.values()) 
    return render(request, 'search_tweets.html', {
        'query': query,
        'sentiment_counts': sentiment_counts,
        'total_sentiments': total_sentiments
    })

