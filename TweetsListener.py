from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream
from kafka import KafkaProducer

# Configuration des clés d'API Twitter
consumer_key = "YOUR_CONSUMER_KEY"
consumer_secret = "YOUR_CONSUMER_SECRET"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

# Configuration de Kafka Producer
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# Classe pour écouter les tweets en continu
class TweetsListener(StreamListener):
    def on_data(self, data):
        producer.send('twitter_tweets', value=data)
        return True

    def on_error(self, status):
        print(status)

# Authentification auprès de l'API Twitter
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Création de l'instance StreamListener
listener = TweetsListener()

# Création de l'instance Stream
twitter_stream = Stream(auth, listener)

# Filtrer les tweets par des mots-clés pertinents
twitter_stream.filter(track=['S&P 500', 'stock market', 'finance'])
