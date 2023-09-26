from kafka import KafkaConsumer
from pymongo import MongoClient

# Configuration de MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["sentiment_analysis"]
collection = db["tweets"]

# Configuration de Kafka Consumer
consumer = KafkaConsumer('twitter_tweets', bootstrap_servers='localhost:9092')

# Consommation et stockage des tweets dans MongoDB
for message in consumer:
    tweet = message.value.decode('utf-8')
    collection.insert_one({"tweet": tweet})
    print("Tweet enregistré dans MongoDB:", tweet)
