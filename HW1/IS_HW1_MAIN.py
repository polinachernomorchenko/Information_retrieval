from IS_HW1_PREPROCESSING import folder2df, preprocesser
from IS_HW1_INVERTED_INDEX import inverted_index_matrix, inverted_index_dictionary
from IS_HW1_STATS import matrix_stat, dictionary_stat
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer='word')
friends_folder = 'friends-data/friends-data'
characters = ['моника', 'мон',
              'рейчел', 'рейч', 'рэйчел', 'рэйч',
              'чендлер', 'чэндлер', 'чен',
              'фиби', 'фибс',
              'джоуи', 'джо']


def main():

    friends_data = folder2df(friends_folder)                           # собирает данные из папок в датафрейм
    friends_data['text'] = friends_data['text'].apply(preprocesser)    # препроцессинг
    inverted_matrix = inverted_index_matrix(friends_data, vectorizer)  # создает обратный индекс в виде матрицы
    inverted_dictionary = inverted_index_dictionary(inverted_matrix)   # трансформирует матрицу в словарь
    matrix_stat(inverted_matrix, characters)                           # считает задания по матрице и выводит
    dictionary_stat(inverted_dictionary, characters)                   # считает задания по словарю и выводит


if __name__ == '__main__':
    main()
