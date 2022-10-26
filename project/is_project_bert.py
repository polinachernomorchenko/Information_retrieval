import pandas as pd
import numpy as np
import time
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def bert_corpus_creation(bert_corpus_path):
    df = pd.read_csv(bert_corpus_path)
    corpus = np.asarray(df.values)
    return corpus


def bert_vec(q, tokenizer=tokenizer):

    encoded_input = tokenizer([q], padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


def inner_bert_search(query, corpus, num_ans):
    q_vec = bert_vec(query)
    res = cosine_similarity(corpus, q_vec).reshape(1, -1)[0]
    idx = np.argpartition(res, -num_ans)[-num_ans:]
    idx = sorted(idx, key=lambda x: res[x])
    return idx


def bert_search(query, bert_matrix, questions, answers, num_ans):
    start_time = time.time()
    idx = inner_bert_search(query, bert_matrix, num_ans)
    t = time.time() - start_time

    end_res_a = [str(answers['answers'][i]).split('|') for i in idx]
    end_res_q = [questions['question'][i] for i in idx]
    end_res = [[end_res_q[j], end_res_a[j]] for j in range(len(end_res_a))]

    return end_res, t

