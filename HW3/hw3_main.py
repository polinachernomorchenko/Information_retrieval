import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
from nltk.corpus import stopwords
stop_words = stopwords.words('russian')
import re
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vec = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

tmp_dict = {}
k = 2
b = 0.75
query = 'как перестать страдать по бывшему'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True, help='Path to dir with data')
    arg = parser.parse_args()
    return arg


def json2df(path_to_json = 'email/data.jsonl'):
    
    df = pd.read_json(path_to_json, lines = True)
    df['question'] = df['question'].astype(str) +" "+ df['comment']  # джоин  bc комментарий == развернутый вопрос
    df = df.drop(columns = ['comment', 'sub_category', 'author', 'author_rating', 'poll'])
    df_q = pd.DataFrame(df['question'])
    df_a = df.drop(columns = ['question'])
    
    return df_q, df_a
    

def preprocesser(text, stopwords=stop_words):

    text = text.lower()
    tokenized_text = tokenizer.tokenize(text)
    clean_text = ' '.join([i for i in tokenized_text if i not in stopwords])
    
    return clean_text


def fast_lemmatizer(tokens):
    
    global tmp_dict
    words = []
    for t in tokens:
        if t in tmp_dict:
            words.append(tmp_dict[t])
        else:
            pv = morph.parse(t)
            lemma = pv[0].normal_form
            tmp_dict[t] = lemma
            words.append(lemma)
            
    return ' '.join(words)
            

def okapi(value, k, b, t_len, avg_len):
    
    ch = value * (k + 1)
    zn = value + k*(1 - b + b*(t_len/avg_len))
    res = ch/zn
    
    return res
    

def create_bm_matrix(vec, tfidf_vectorizer, data_dir):
    questions, answers = json2df(data_dir)
    questions['question'] = questions['question'].apply(preprocesser)
    questions['question'] = questions['question'].map(lambda x: fast_lemmatizer(x.split()))
    
    tf = vec.fit_transform(questions['question'])
    tfidf_spr = tfidf_vectorizer.fit_transform(questions['question']) 
    idf = tfidf_vectorizer.idf_ 
    
    q_lens = [np.sum(i) for i in tf]
    avg_len = np.mean(q_lens)
    
    for r, c in zip(*tf.nonzero()): 
        value = tf[r, c]
        t_len = q_lens[r]
        res = okapi(value, k, b, t_len, avg_len)
        mult_res = res * idf[c]
        tf[r, c] = mult_res
    
    return tf, vec, answers
    

def query_prep(query, vec):
    
    clean_query = preprocesser(query)
    lem_query = fast_lemmatizer(clean_query.split())
    vec_query = vec.transform([lem_query])
    
    return vec_query


def search(vec_query, bm_matrix, answers):
    
    res = pairwise_distances(bm_matrix, vec_query, metric = 'cosine')
    idx = np.argmin(res)
    ans = [i['text'] for i in answers['answers'][idx]]
    
    return '\n'.join(ans)
    

def main(vec, tfidf_vectorizer, data_dir):
    bm_matrix, vec, answers = create_bm_matrix(vec, tfidf_vectorizer, data_dir)
    vec_query = query_prep(query, vec)
    ans = search(vec_query, bm_matrix, answers)
    print(ans)


if __name__ == "__main__":
    data_dir = parse_args().data_dir
    main(vec, tfidf_vectorizer, data_dir)




