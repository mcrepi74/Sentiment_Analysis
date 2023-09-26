# Import des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn import preprocessing
import re
from google.colab import files
import io

# Chargement du jeu de données depuis un fichier CSV
uploaded = files.upload()
data = pd.read_csv(io.BytesIO(uploaded['reduced_sentiment140_dataset.csv']))

# Expressions régulières pour le nettoyage du texte
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|rt"
TEXT_CLEANING_RE_2 = "[^A-Za-z0-9.!?:;()]+"

# Fonction de prétraitement du texte
def preprocessing_text(text):
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    text = re.sub(TEXT_CLEANING_RE_2, ' ', str(text).lower()).strip()
    return text

# Transformation des étiquettes de classe en 0 et 1
data['target'] = data['target'].apply(lambda x: 1 if x == 4 else 0)
data['text'] = data['text'].apply(preprocessing_text)

# Séparation des données en ensembles d'entraînement et de test
X = np.array(data['text'])
y = np.array(data['target'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du tokenizer BERT
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Définition des paramètres pour le prétraitement des textes
pad_token = 0
pad_token_segment_id = 0
max_length = 60

# Fonction de conversion des textes en entrées pour BERT
def convert_to_input(reviews):
    input_ids, attention_masks, token_type_ids = [], [], []
    for x in tqdm(reviews, position=0, leave=True):
        inputs = bert_tokenizer.encode_plus(x, add_special_tokens=True, max_length=max_length)
        i, t = inputs["input_ids"], inputs["token_type_ids"]
        m = [1] * len(i)
        padding_length = max_length - len(i)
        i = i + ([pad_token] * padding_length)
        m = m + ([0] * padding_length)
        t = t + ([pad_token_segment_id] * padding_length)
        input_ids.append(i)
        attention_masks.append(m)
        token_type_ids.append(t)
    return [np.asarray(input_ids), np.asarray(attention_masks), np.asarray(token_type_ids)]

# Conversion des données en entrées pour BERT
X_test_input = convert_to_input(X_test)
X_train_input = convert_to_input(X_train)

# Fonction pour créer des exemples à partir des entrées et des étiquettes
def example_to_features(input_ids, attention_masks, token_type_ids, y):
    return {"input_ids": input_ids, "attention_mask": attention_masks, "token_type_ids": token_type_ids}, y

# Création de jeux de données TensorFlow à partir des exemples
train_ds = tf.data.Dataset.from_tensor_slices((X_train_input[0], X_train_input[1], X_train_input[2], y_train)).map(example_to_features).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((X_test_input[0], X_test_input[1], X_test_input[2], y_test)).map(example_to_features).batch(64)

# Chargement du modèle BERT pré-entraîné
bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")

# Définition de l'optimiseur, de la fonction de perte et de la métrique
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# Compilation du modèle BERT
bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Entraînement du modèle
history = bert_model.fit(train_ds, epochs=3, validation_data=test_ds)

# Sauvegarde du modèle pré-entraîné
bert_model.save_pretrained('/content')

# Affichage de l'historique de l'entraînement
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
