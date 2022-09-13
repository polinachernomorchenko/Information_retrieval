#!/usr/bin/env python
# coding: utf-8

# In[3]:

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd


def inverted_index_matrix(friends_data, vectorizer):
    
    text = friends_data['text'].tolist()
    bow_text = vectorizer.fit_transform(text)
    voc = vectorizer.get_feature_names_out()
    
    friends_bow_matrix = pd.DataFrame(bow_text.toarray(), columns = [voc]) 
    friends_bow_matrix.insert(0, 'title', friends_data['title'])
    friends_bow_matrix = friends_bow_matrix.transpose()
    friends_bow_matrix = friends_bow_matrix.rename(columns=friends_bow_matrix.iloc[0])                         
    friends_bow_matrix.drop(index=friends_bow_matrix.index[0], axis=0, inplace=True)
    
    return friends_bow_matrix


# In[4]:


def inverted_index_dictionary(inverted_matrix):
    
    inverted_dictionary = {}
    docs = np.asarray(inverted_matrix.columns)
    
    for i in range(len(inverted_matrix)):
        row = inverted_matrix.iloc[i].values.tolist()
        res = [[docs[j], row[j]] for j in range(len(docs))]
        inverted_dictionary[inverted_matrix.index[i][0]] = res
        
    return inverted_dictionary
        

