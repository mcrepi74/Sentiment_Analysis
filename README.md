# Sentiment_Analysis

Analyse de Sentiment sur les Tweets Financiers
Objectif

L'objectif de ce projet est de collecter des tweets sur un sujet spécifique, en l'occurrence l'indice boursier S&P 500, puis d'analyser le sentiment négatif/positif lié à ces tweets.

Cela permettra d'obtenir un aperçu de l'évolution du sentiment autour du S&P 500 au fil du temps.

Une autre idée sous-jacente est que nous observons parfois une baisse du prix même après de bonnes nouvelles (par exemple, des bénéfices élevés). Cela peut être lié au fait que les investisseurs avaient déjà inclus les bonnes nouvelles dans le prix et s'attendaient à des nouvelles encore meilleures, ce qui entraîne un sentiment négatif. Ainsi, l'analyse du sentiment concernant les moteurs du S&P 500 (croissance du PIB, chômage, etc.) pourrait capturer ce phénomène.

Étapes Générales
Collecte des tweets dans une base de données cloud MongoDB à partir du 01/01/2020 jusqu'au 15/05/2020 à l'aide de MongoStoreHistoricalTweets.py.

Deux méthodes ont été utilisées pour classer les tweets en négatifs ou positifs, VaderSentiment et BERT :

Utilisation de VaderSentimentMongoDB.py pour étiqueter les tweets comme étant positifs ou négatifs. Vader Sentiment est un outil d'analyse de sentiment basé sur un lexique et des règles spécialement conçu pour analyser les sentiments exprimés sur les médias sociaux. L'avantage principal est qu'il ne nécessite pas de jeu de données étiqueté.

Utilisation du modèle BERT pour étiqueter les tweets en tant que positifs ou négatifs. BERT est une technique centrale en traitement du langage naturel (NLP), publiée en 2018 et utilisée dans toutes sortes de tâches de NLP. Il est pré-entraîné sur un corpus de texte, ce qui lui confère d'excellents résultats sur de petits ensembles de données. J'ai utilisé le modèle BERT cased, puis je l'ai affiné en utilisant une version réduite de l'ensemble de données étiquetées Sentiment140. L'entraînement est réalisé avec BERT_training.py et les prédictions avec BERT.py.

Dans le dossier Image, j'ai fourni deux exemples d'une analyse simple des résultats à l'aide de Tableau.

J'ai également utilisé Twitter en tant que producteur pour diffuser des tweets dans Kafka à l'aide de TweetsListener.py, puis les ai consommés/stockés dans une base de données MongoDB via Consumer.py. Kafka est une plateforme de streaming distribuée utilisée pour centraliser les données.