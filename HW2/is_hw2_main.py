import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from is_hw1_preprocessing import preprocesser
from scipy.spatial.distance import cdist
import argparse
from pathlib import Path

# фрагмент 1 серии в качестве тестового запроса
query = """Сегодня меня высмеяли  
           на двенадцати собеседованиях.

           А настроение у тебя на удивление неплохое

           У тебя было бы такое же, если бы ты
           нашел сапоги от "Джон энд Дэвид"...

           ...на распродаже за полцены! """

vectorizer = TfidfVectorizer()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lemmas_dir', type=Path, required=True, help='Path to dir with lemmatized data')
    arg = parser.parse_args()
    return arg


def query_transformer(text, vectorizer):  # предобработка и индексация запроса
    text = preprocesser(text)
    vec = vectorizer.transform([text])
    return vec


def search(vec, tfidf_data, friends_data):  # поиск
    dist = cdist(tfidf_data.toarray(), vec.toarray(), metric='cosine')
    idx = np.argmin(dist) 
    return friends_data['title'][idx]
    

def main(query, lemmas_dir):
    friends_data = pd.read_csv(lemmas_dir)
    tfidf_data = vectorizer.fit_transform(friends_data['text'])
    vec = query_transformer(query, vectorizer)
    res = search(vec, tfidf_data, friends_data)
    print('Наиболее подходящий документ: ', res)


if __name__ == "__main__":
    lemmas_dir = parse_args().lemmas_dir
    main(query, lemmas_dir)

   

