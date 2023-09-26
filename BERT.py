from transformers import BertTokenizer, BertForSequenceClassification
from pymongo import MongoClient

# Configuration de MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["sentiment_analysis"]
collection = db["tweets"]

# Charger le modèle BERT pré-entraîné et le tokenizer
model_name = "bert-base-uncased"  # Exemple de modèle BERT
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Fonction pour étiqueter les tweets avec BERT
def analyze_sentiment_bert(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    
    # Adaptation des classes prédites à vos besoins
    if predicted_class == 0:
        return "négatif"
    elif predicted_class == 1:
        return "neutre"
    else:
        return "positif"

# Récupération des tweets depuis MongoDB
tweets = collection.find()

# Analyse et étiquetage des tweets avec BERT
for tweet in tweets:
    tweet_text = tweet["text"]
    sentiment_label = analyze_sentiment_bert(tweet_text)
    collection.update_one({"_id": tweet["_id"]}, {"$set": {"bert_sentiment": sentiment_label}})
