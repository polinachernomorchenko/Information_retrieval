import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from IS_HW1_PREPROCESSING import preprocesser
from scipy.spatial.distance import cdist


friends_data = pd.read_csv('friends_lemmatized.csv')
vectorizer = TfidfVectorizer()

# фрагмент 1 серии в качестве тестового запроса
query = """Сегодня меня высмеяли  
           на двенадцати собеседованиях.

           А настроение у тебя на удивление неплохое

           У тебя было бы такое же, если бы ты
           нашел сапоги от "Джон энд Дэвид"...

           ...на распродаже за полцены! """ 


def query_transformer(text, vectorizer):  # предобработка и индексация запроса
    text = preprocesser(text)
    vec = vectorizer.transform([text])
    return vec


def search(vec, tfidf_data, friends_data):  # поиск
    dist = cdist(tfidf_data.toarray(), vec.toarray(), metric='cosine')
    idx = np.argmin(dist) 
    return friends_data['title'][idx]
    

def main(query, friends_data):
    tfidf_data = vectorizer.fit_transform(friends_data['text'])  # не вижу смысла делать функцию для одной строчки
    vec = query_transformer(query, vectorizer)
    res = search(vec, tfidf_data, friends_data)
    print('Наиболее подходящий документ: ', res)


if __name__ == "__main__":
    main(query, friends_data)

   

