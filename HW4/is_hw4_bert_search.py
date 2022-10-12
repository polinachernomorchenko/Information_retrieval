import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cdist

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
query = 'Как перестать страдать по бывшему?'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=Path, required=True, help='Path to dir with json')
    parser.add_argument('--corpus_dir', type=Path, required=True, help='Path to dir with vec corpus')
    args = parser.parse_args()
    return args


def json2df(path_to_json):
    
    df = pd.read_json(path_to_json, lines = True)
    df['question'] = df['question'].astype(str) + ' ' + df['comment']  # джоин  bc комментарий == развернутый вопрос
    df = df.drop(columns=['comment', 'sub_category', 'author', 'author_rating', 'poll'])
    df_a = df.drop(columns=['question'])
    
    return df_a


def corpus_creation(path):
    corpus = pd.read_csv(path)
    corpus = corpus.drop(columns = ['Unnamed: 0'])
    corpus_arr = np.asarray(corpus.values)
    return corpus_arr


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def q_vec(q, tokenizer=tokenizer):    
    
    encoded_input = tokenizer([q], padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


def search(q_v, df_a, corpus):
    dist = cdist(corpus, q_v, metric = 'cosine')
    idx = np.argmin(dist)
    ans = [i['text'] for i in df_a['answers'][idx]]
    
    return '\n'.join(ans)


def main(json_dir, corpus_dir):
    df_a = json2df(json_dir)
    corpus = corpus_creation(corpus_dir)
    q_v = q_vec(query)
    print(search(q_v, df_a, corpus))
    

if __name__ == "__main__":
    json_dir = parse_args().json_dir
    corpus_dir = parse_args().corpus_dir
    main(json_dir, corpus_dir)






