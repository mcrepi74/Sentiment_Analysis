# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:38:47 2020

@author: 33678
"""
from transformers import (TFBertForSequenceClassification,
                          BertTokenizer)
import pandas as pd
import re

from sqlalchemy import create_engine
from postgres_credentials import dbnametwitter, usertwitter, passwordtwitter, hosttwitter, porttwitter

from tqdm import tqdm



#PARAMETERS
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|rt"
TEXT_CLEANING_RE_2 = "[^A-Za-z0-9]+"
DATASET_FILE= r"C:\Users\Bert Sentiment140\data\reduced_sentiment140_dataset.csv"
DATASET_ENCODING = "ISO-8859-1"


#querying the database
def query_database(tabletweets):
    """
    This function returns in pandas dataframe all the rows of the table tabletweets from a database
    """
    engine = create_engine("postgresql+psycopg2://%s:%s@%s:%d/%s" %(usertwitter, passwordtwitter, hosttwitter, porttwitter, dbnametwitter))
    table = pd.read_sql_query("select * from %s" %tabletweets,con=engine, index_col="tweet_id") 
    return table


#Download BERT tokenizer 
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


#Download our trained Bert model
bert_model = TFBertForSequenceClassification.from_pretrained(
    r'C:\Users\Bert Sentiment140\model')

bert_model.summary()


def preprocessing_text(text):
# Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, '', str(text).lower()).strip()
    text = re.sub(TEXT_CLEANING_RE_2, '', str(text).lower()).strip()

    return text

def sentiment_prediction(text):
    """
    This function preprocesses the text and our model computes a prediction of the sentiment
    """
    formatted_input = preprocessing_text(text)
    formatted_input = bert_tokenizer.encode_plus(formatted_input, add_special_tokens=True, \
                                                 pad_to_max_length=True, max_length=120, return_tensors='tf')
    return bert_model.predict(formatted_input).argmax()


#List of topics
names = ["SP500", "Fed", "GDP", "Employment", "Inflation", "Earnings", "Cov19"]


# Create CSVs results
for name in tqdm(names):
    table = query_database("tweets_"+name)
    table = table.drop(columns = ['user_id','favorite_count','retweet_count'])
    table['sentiment'] = table['tweet'].apply(sentiment_prediction)
    table.to_csv("C:/Users/\
                 Bert Sentiment140/Bert_sentiment/{}_sentiment.csv".format(name))
   
    
    
    
    
    
    
    
    
    
