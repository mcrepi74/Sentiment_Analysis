import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pymongo import MongoClient

# Configuration de MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["sentiment_analysis"]
collection = db["tweets"]

# Initialisation de l'analyseur de sentiment de Vader
nltk.download("vader_lexicon")  # Télécharger le lexique de Vader
analyzer = SentimentIntensityAnalyzer()

# Fonction pour étiqueter les tweets avec VaderSentiment
def analyze_sentiment_vader(text):
    sentiment = analyzer.polarity_scores(text)
    if sentiment["compound"] >= 0.05:
        return "positif"
    elif sentiment["compound"] <= -0.05:
        return "négatif"
    else:
        return "neutre"

# Récupération des tweets depuis MongoDB
tweets = collection.find()

# Analyse et étiquetage des tweets avec VaderSentiment
for tweet in tweets:
    tweet_text = tweet["text"]
    sentiment_label = analyze_sentiment_vader(tweet_text)
    collection.update_one({"_id": tweet["_id"]}, {"$set": {"vader_sentiment": sentiment_label}})
