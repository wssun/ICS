import argparse
import math
import os
import pickle

import numpy as np
from tqdm import tqdm
from identifier_splitting import split_identifier_into_parts
import nltk
from transformers import (RobertaTokenizer)


def add_args(parser):
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--saved_model_path", type=str, default='')
    parser.add_argument("--result_path", type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    data_path = args.data_path
    saved_model_path = args.saved_model_path
    result_path = args.result_path

    if not os.path.exists(saved_model_path):
        print("The directory path does not exist!")
    else:
        # load the saved data
        U2_save_path = os.path.join(saved_model_path, "U2.csv")
        sigma2_save_path = os.path.join(saved_model_path, "sigma2.csv")
        word_index_save_path = os.path.join(saved_model_path, "word_index.npy")
        dictionary_save_path = os.path.join(saved_model_path, "dictionary.npy")
        n_train_document = 167288
        U2 = np.loadtxt(open(U2_save_path, "rb"), delimiter=",", skiprows=0)
        sigma2 = np.loadtxt(open(sigma2_save_path, "rb"), delimiter=",", skiprows=0)
        inv_sigma2 = np.linalg.inv(sigma2)
        with open(word_index_save_path, "rb") as f:
            word_index = pickle.load(f)
        with open(dictionary_save_path, "rb") as f:
            dictionary = pickle.load(f)

        tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
        tokenizer.do_lower_case = True
        with open(data_path, 'r', encoding="utf-8") as reader:
            lines = reader.readlines()
            clean_reviews = []
            for line in tqdm(lines):
                id, code = line.split('\t')
                code = " ".join(split_identifier_into_parts(code))
                reviews = [code]
                for review in reviews:
                    s = tokenizer.tokenize(review)
                    s = [word.lower() for word in s]
                    s = [word for word in s if not word in set(
                        nltk.corpus.stopwords.words('english'))]
                    s = [word for word in s if word.isalpha()]
                    m = []
                    for word in s:
                        if word[0] == 'ġ':
                            word = word[1:]
                        if word != 'ĉ':
                            m.append(word)
                    clean_reviews.append(m)

    with open(os.path.join(result_path, "test.txt"), 'w', encoding="utf-8") as writer:
        id = 0
        for review in clean_reviews:
            answers = []

            all_word_array = list(dictionary.keys())

            tf_array = []
            idf_array = []
            for word in all_word_array:
                tf = review.count(word) / len(review)
                tf_array.append(tf)
                if word in review:
                    idf = math.log((n_train_document + 1) / (len(set(dictionary[word])) + 2))
                    idf_array.append(idf)
                else:
                    idf = math.log((n_train_document + 1) / (len(set(dictionary[word])) + 1))
                    idf_array.append(idf)
            q = []
            for i in range(0, len(dictionary)):
                q.append(tf_array[i] * idf_array[i])
            v_review = np.dot(q, U2)
            v_review = np.dot(v_review, inv_sigma2)

            unique_word_array = []
            for word in review:
                if word not in unique_word_array:
                    unique_word_array.append(word)

            final_word_array = []
            for j in range(0, len(unique_word_array)):
                if word_index.get(unique_word_array[j]) is not None:
                    final_word_array.append(unique_word_array[j])

            cos_array = []

            for word in final_word_array:
                index = word_index[word]
                word_cos = np.array(U2[index])
                cos_array.append(v_review.dot(word_cos) / (np.linalg.norm(v_review) * np.linalg.norm(word_cos)))
            top_words = np.array(cos_array).argsort()[-10:][::-1]

            for w in top_words:
                if final_word_array[w] not in answers:
                    answers.append(final_word_array[w])
            answers = answers[0:10]
            writer.write(str(id) + "\t" + " ".join(answers) + "\n")
            id += 1