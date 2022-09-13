#!/usr/bin/env python
# coding: utf-8

# In[4]:

import pandas as pd
from math import prod


def switch(el):

    if el.startswith('м'):
         return 'моника'
    elif el.startswith('р'):
        return 'рейчел'
    elif el.startswith('ч'):
        return 'чендлер'
    elif el.startswith('д'):
        return 'джоуи'
    elif el.startswith('ф'):
        return 'фиби'


# In[5]:


def matrix_stat(inverted_matrix, characters):
    
    print('СТАТИСТИКА ПО МАТРИЦЕ')
    
    inverted_matrix['total'] = inverted_matrix.sum(axis=1)
    inverted_matrix = inverted_matrix.sort_values(by = 'total')    
    print('Самое частое слово: ', inverted_matrix.index[-1][0])
    print('Самое редкое слово: ', inverted_matrix.index[0][0])
    
    
    inverted_matrix['prod'] = inverted_matrix.prod(axis = 1)
    all_docs_words_df = inverted_matrix[inverted_matrix['prod'] != 0]
    all_docs_words = [i[0] for i in all_docs_words_df.index.tolist()]
    print('Слова, которые есть во всех документах: ', ', '.join(all_docs_words))
    
    
    chars = [[inverted_matrix.index[i][0], inverted_matrix['total'][i]]
             for i in range(len(inverted_matrix)) 
             if inverted_matrix.index[i][0] in characters]
    
    for i in range(len(chars)):
        chars[i][0] = switch(chars[i][0])
        
    d_chars = {}
    for i in chars:
        if i[0] in d_chars:
            d_chars[i[0]] += i[1]
        else:
            d_chars[i[0]] = i[1]
            
    popular_char = max(d_chars, key=d_chars.get) 
    print('Самый частоупоминаемый персонаж: ', popular_char.capitalize())
    print('\n')
    


# In[6]:


def dictionary_stat(inverted_dictionary, characters):
    
    print('СТАТИСТИКА ПО СЛОВАРЮ')
    
    wordlist = []
    for i in inverted_dictionary:
        total = sum(cnt for doc, cnt in inverted_dictionary[i])
        wordlist.append([i, total])
        
    wordlist = sorted(wordlist, key = lambda x: x[1])    
    print('Самое частое слово по словарю: ', wordlist[-1][0])
    print('Самое редкое слово по словарю: ', wordlist[0][0])
    
    all_docs_words = []
    for i in inverted_dictionary:
        mult = prod(cnt for doc, cnt in inverted_dictionary[i])
        if mult != 0:
            all_docs_words.append(i)           
    print('Слова, которые есть во всех документах: ', ', '.join(all_docs_words))
    
    
    chars = [[i, sum(cnt for doc, cnt in inverted_dictionary[i])] for i in characters]
    for i in range(len(chars)):
        chars[i][0] = switch(chars[i][0])
        
    d_chars = {}
    for i in chars:
        if i[0] in d_chars:
            d_chars[i[0]] += i[1]
        else:
            d_chars[i[0]] = i[1]
            
    popular_char = max(d_chars, key=d_chars.get) 
    print('Самый частоупоминаемый персонаж: ', popular_char.capitalize())

