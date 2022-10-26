import streamlit as st
import pandas as pd
from scipy import sparse
import pickle
import base64
from is_project_bert import bert_corpus_creation, bert_search
from is_project_not_bert import not_bert_search
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-corpus_bert", type=Path,
                        help='Path to BERT corpus')
    parser.add_argument("-corpus_bm", type=Path,
                        help='Path to BM25 corpus')
    parser.add_argument("-corpus_tfidf", type=Path,
                        help="Path to TF-IDF corpus")
    parser.add_argument("-answers_text", type=Path,
                        help="Path to readable answers df")
    parser.add_argument("-questions_text", type=Path,
                        help="Path to readable questions df")
    parser.add_argument("-vec_bm", type=Path,
                        help="Path to countvectorizer for bm")
    parser.add_argument("-vec_tfidf", type=Path,
                        help="Path to tfidfvectorizer for tfidf")
    parser.add_argument("-image_bg", type=Path,
                        help="Path to background image")

    args = parser.parse_args()
    return args


def res_printer(res):
    for i in range(len(res)):
        st.write('\n\n')
        if res[i][0][-1] == ' ':
            res[i][0] = res[i][0][:-1]

        s = '**' + res[i][0] + '**'
        st.markdown(s)
        st.write('\n'.join(res[i][1]))


def add_bg_from_url(im_file):

    with open(im_file, "rb") as f:
        encoded_string = base64.b64encode(f.read())

    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


def main():
    args = parse_args()

    tfidf_matrix = sparse.load_npz(args.corpus_tfidf)
    bm_matrix = sparse.load_npz(args.corpus_bm)
    bert_matrix = bert_corpus_creation(args.corpus_bert)

    tfidf_vec = pickle.load(open(args.vec_tfidf, 'rb'))
    bm_vec = pickle.load(open(args.vec_bm, 'rb'))

    answers = pd.read_csv(args.answers_text)
    questions = pd.read_csv(args.questions_text)

    add_bg_from_url(args.image_bg)
    st.title('👀 поиск')
    c_query, c_choice = st.columns([4, 1])
    choice = c_choice.selectbox(label='Тип индекса', options=['TF-IDF', 'BM-25', 'BERT'])
    query = c_query.text_input('Введите запрос:')
    num_ans = st.slider("Количество ответов", min_value=1, max_value=10)

    if query:
        if choice != 'BERT':
            if choice == 'TF-IDF':
                vec = tfidf_vec
                matrix = tfidf_matrix
            else:
                vec = bm_vec
                matrix = bm_matrix

            res, t = not_bert_search(query, matrix, questions, answers, vec, num_ans)
            res_printer(res)
            st.write(f'{round(t, 3)} сек')
        else:
            res, t = bert_search(query, bert_matrix, questions, answers, num_ans)
            res_printer(res)
            st.write(f'{round(t, 3)} сек')


if __name__ == '__main__':
    main()





