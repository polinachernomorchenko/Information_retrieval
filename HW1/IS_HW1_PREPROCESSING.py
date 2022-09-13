#!/usr/bin/env python
# coding: utf-8

# In[16]:

import os
import pandas as pd

from pymystem3 import Mystem
mystem = Mystem()

from nltk.corpus import stopwords
stop_words = stopwords.words('russian')

import re
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')




# In[17]:


def folder2df(friends_folder):

    seasons = os.listdir(friends_folder)

    titles = []
    texts = []

    for season in seasons:
        episodes = os.listdir(os.path.join(friends_folder, season))

        for episode in episodes:
            with open(os.path.join(friends_folder, season, episode), 'r', encoding = 'utf-8') as file:
                text = file.read()
                texts.append(text)

        titles.extend(episodes)

    friends_data = pd.DataFrame(columns = ['title', 'text'])
    friends_data['title'], friends_data['text'] = titles, texts
    
    return friends_data


# In[18]:


def preprocesser(text, stopwords=stop_words, mystem=mystem):

    text = text.lower()
    tokenized_text = tokenizer.tokenize(text)
    clean_text = ' '.join([i for i in tokenized_text if i not in stopwords])
    lemmas = mystem.lemmatize(clean_text)
    return ''.join(lemmas)

