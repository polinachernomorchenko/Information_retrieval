import pandas as pd
import numpy as np
import pickle
from scipy import sparse
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_q_dir', type=Path, required=True, help='Path to dir with bert-questions')
    parser.add_argument('--bert_a_dir', type=Path, required=True, help='Path to dir with bert-answers')
    parser.add_argument('--vec_dir', type=Path, required=True, help='Path to dir with counvectorizer')
    parser.add_argument('--bm_q_dir', type=Path, required=True, help='Path to dir lemmatized questions for bm')
    parser.add_argument('--bm_a_dir', type=Path, required=True, help='Path to dir with bm answers matrix')
    arg = parser.parse_args()
    return arg


def bert_arr_creation(bert_q_path, bert_a_path):
    q = pd.read_csv(bert_q_path)
    a = pd.read_csv(bert_a_path)
    q_arr = np.asarray(q.values)
    a_arr = np.asarray(a.values)
    
    return q_arr, a_arr


def bert_metric(bert_q_path, bert_a_path):
    q_arr, a_arr = bert_arr_creation(bert_q_path, bert_a_path)
    t_a_arr = a_arr.T
    res = np.dot(q_arr, t_a_arr)
    metric = [1 for i in range(len(res)) if i in np.argpartition(res[i], -5)[-5:]]
    bert_res = len(metric)/len(q_arr)
    
    return bert_res


def bm_metric(vec_path, bm_q_path, bm_a_path):
    vec = pickle.load(open(vec_path, 'rb'))
    bm_matrix = sparse.load_npz(bm_a_path)
    q_bm = pd.read_csv(bm_q_path)
    q_bm_vec = vec.transform(q_bm['question'])
    t_bm_matrix = bm_matrix.T
    res = q_bm_vec * t_bm_matrix
    res = res.toarray()
    
    metric = [1 for i in range(len(res)) if i in np.argpartition(res[i], -5)[-5:]]
    bm_res = len(metric)/len(res)
    
    return bm_res


def main():
    bert_q_path = parse_args().bert_q_dir
    bert_a_path = parse_args().bert_a_dir
    vec_path = parse_args().vec_dir
    bm_q_path = parse_args().bm_q_dir
    bm_a_path = parse_args().bm_a_dir

    bert_res = bert_metric(bert_q_path, bert_a_path)
    bm_res = bm_metric(vec_path, bm_q_path, bm_a_path)

    print(f'Значение метрики для BERT: {bert_res}')
    print(f'Значение метрики для BM-25: {bm_res}')
    

if __name__ == "__main__":
    main()

