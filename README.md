# Information_retrieval
## HW1:

(1) Запуск: файл IS_HW1_MAIN.py -> main()

(2) Модули:	
Препроцессинг:
- folder2df (inp: путь к папке, out: данные)
- - preprocesser (нижний ригистр, удаление стоп-слов, лемматизация)

Обратный индекс:
- inverted_index_matrix (inp: данные, out: обратный индекс в виде матрицы)
- inverted_index_dictionary (inp: обратный индекс в виде матрицы, out: обратный индекс в виде словаря)

Статистика:
- switch (унифицирует имена персонажей)
- matrix_stat (выводит ответы на задания по матрице)
- dictionary_stat (выводит ответы на задания по словарю)

## HW2

(1) Запуск: is_hw2_main.py -> main()

(2) Функции:
- parse_args(): парсит путь до файла с предобработанными документами
- query_transformer(text, vectorizer): препроцессинг и векторизация запроса
- search(vec, tfidf_data, friends_data): поиск наиболее релевантного запросу документа
