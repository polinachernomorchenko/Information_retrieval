# Information_retrieval
## HW1:

(1) Запуск: файл IS_HW1_MAIN.py -> main()

(2) Модули:	
Препроцессинг:
- folder2df (inp: путь к папке, out: данные)
- preprocesser (нижний ригистр, удаление стоп-слов, лемматизация)

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

## HW3

(1) Запуск: is_hw3_main.py -> main()

(2) Функции:
- parse_args(): парсит путь до файла с данными
- json2df(): парсит json, возвращает 2 датафрейма: с вопросами и с ответами
- preprocesser(): токенизация, нижний регистр
- fast_lemmatizer(): лемматизирует через пайморфи, но кэширует уже обработанные слова в словарь, экономя время
- okapi(): вычисляет второй множитель по формуле bm25
- create_bm_matrix(): возвращает посчитанную bm-матрицу, настроенный для документов countвекторайзер и датафрейм с ответами из json2df()
- query_prep(): препроцессинг и векторизация запроса
- search(): поиск

Комментарий: поиск ведётся по вопросам, в качестве выдачи - ответы на них

## HW4
### Ссылка на папку с вспомогательными файлами: https://drive.google.com/drive/folders/1-mGV95y2SpwHljfKOy3VafKTea2d4_gI?usp=sharing
### Задание 1:

(1) Запуск: is_hw4_bert_search.py -> main()

(2) Функции:
- parse_args(): парсит путь до файлов с данными (оригинальный json и векторизованный бертом корпус)
- json2df():парсит json, возвращает датафрейм с ответами
- corpus_creation(): парсит файл с векторизованным корпусом, возвращает его в виде numpy array
- mean_pooling(): копипаст с huggingface для векторизации запроса
- q_vec(): векторизация запроса
- search(): поиск

Комментарий: векторизованный корпус лежит по ссылке в 'corpus.zip'

### Задание 2:

(1) Запуск: is_hw4_quality_comparsion.py -> main()

(2) Функции: 
- parse_args(): парсит путь до файлов с данными (вопросы и ответы для bm/bert, зафиченный под 10k-корпус countvectorizer)
- bert_arr_creation(): возвращает два numpy array для векторизованных бертом вопросов и ответов
- bert_metric(): считает метрику для bert-поиска
- bm_metric(): считает метрику для bm-поиска
- main(): выводит обе метрики

Комментарий: все релевантные файлы лежат по ссылке

## Project

(1) Запуск: is_project_main.py -> main()
(2) Функции:
- query_preprocessor/bert_vec(): возвращает индексированный запрос
- inner_(not)_bert_search(): считает метрику между запросом и корпусом, возвращает индексы топ-n результатов
- (not)_bert_search(): сопоставляет индексы с текстом вопросов/ответов, возвращает время выполнения inner_search и текст выдачи
- res_printer(): форматирует и выводит текст выдачи
- add_bg_from_url(): устанавливает картинку-фон

Комментарий: поиск ведётся по вопросам, в выдаче выводятся и вопросы, и ответы.
