import numpy as np
import pandas as pd
import time
from pymystem3 import Mystem
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

mystem = Mystem()
stop_words = stopwords.words('russian')
tokenizer = RegexpTokenizer(r'\w+')


def query_preprocessor(text, vec, stopwords=stop_words, mystem=mystem, tokenizer=tokenizer):
    text = text.lower()
    tokenized_text = tokenizer.tokenize(text)
    clean_text = ' '.join([i for i in tokenized_text if i not in stopwords])
    lemmas = mystem.lemmatize(clean_text)
    vec_query = vec.transform([''.join(lemmas)])

    return vec_query


def inner_not_bert_search(vec_query, matrix, num_ans):
    res = matrix*vec_query.T
    res = res.toarray().reshape(1, -1)[0]
    idx = np.argpartition(res, -num_ans)[-num_ans:]
    idx = sorted(idx, key=lambda x: res[x], reverse=True)

    return idx


def not_bert_search(query, matrix, questions, answers, vec, num_ans):
    start_time = time.time()
    vec_query = query_preprocessor(query, vec)
    idx = inner_not_bert_search(vec_query, matrix, num_ans)
    t = time.time() - start_time

    end_res_a = [str(answers['answers'][i]).split('|') for i in idx]
    end_res_q = [questions['question'][i] for i in idx]
    end_res = [[end_res_q[j], end_res_a[j]] for j in range(len(end_res_a))]

    return end_res, t