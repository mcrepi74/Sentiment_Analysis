from pymongo import MongoClient

# Configuration de MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["sentiment_analysis"]
collection = db["tweets"]

# Insérer les tweets dans la collection
for tweet in tweets:
    collection.insert_one(tweet._json)
