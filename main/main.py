import os
import pandas as pd
import re

import numpy as np
import json
import re

import nltk
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
import glob
from sklearn.model_selection import train_test_split
import datetime
import warnings
warnings.filterwarnings('ignore')




API_KEY = os.environ.get('SPOTIFY_API_KEY')

filepath = r"C:\Users\asint\OneDrive\Desktop\Hackathon_Repo\main\datasets\sentimentdataset.csv"  # forward slashes work fine on Windows too

df = pd.read_csv(filepath)

# Dropping columns which are useless (for now)
df = df.drop(columns=["Likes", "Unnamed: 0", "Unnamed: 0.1", "User", "Timestamp", "Platform", "Hashtags", "Retweets", "Country", "Year", "Month", "Day", "Hour"])

# Next: normalize the data, remove emojis, lowercase, try to encode the sentiments
def remove_emojis(text):
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F"
                               "\U0001F300-\U0001F5FF"
                               "\U0001F680-\U0001F6FF"
                               "\U0001F700-\U0001F77F"
                               "\U0001F780-\U0001F7FF"
                               "\U0001F800-\U0001F8FF"
                               "\U0001F900-\U0001F9FF"
                               "\U0001FA00-\U0001FA6F"
                               "\U0001FA70-\U0001FAFF"
                               "\U00002702-\U000027B0"
                               "\U000024C2-\U0001F251]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

df['Text'] = df['Text'].apply(remove_emojis)
df['Sentiment'] = df['Sentiment'].str.strip()

print(df.head())

"""
    - 191 unique sentiments
    - generalize through tokenization
    - multiclass classification
    - 
"""

inputs = df["Text"].to_list()

print(inputs)




