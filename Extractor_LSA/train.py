import argparse
import math
import numpy as np
from identifier_splitting import split_identifier_into_parts
from scipy import linalg
import nltk
from tqdm import tqdm
import os
import pickle
from transformers import (RobertaTokenizer)

stemmer = nltk.stem.snowball.SnowballStemmer("english")
np.random.seed(666)


def lsa(documents, saved_model_path):
    print("calculating lsa...")

    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)

    U2_save_path = os.path.join(saved_model_path, "U2.csv")
    V2_save_path = os.path.join(saved_model_path, "V2.csv")
    sigma2_save_path = os.path.join(saved_model_path, "sigma2.csv")
    word_index_save_path = os.path.join(saved_model_path, "word_index.npy")
    dictionary_save_path = os.path.join(saved_model_path, "dictionary.npy")

    word_index = {}
    unique_words = []
    # key: the word; value: the no of the code snippet in which the word appears
    dictionary = {}
    currentDocId = 0
    index = 0
    for document in documents:
        for word in document:
            if word in dictionary:
                dictionary[word].append(currentDocId)
            else:
                dictionary[word] = [currentDocId]
            if word not in word_index:
                word_index[word] = index
                unique_words.append(word)
                index += 1
        currentDocId += 1

    n_doc = len(documents)
    word_idf_array = []

    # calculate the idf value of each word
    for word in unique_words:
        idf = math.log(n_doc / (len(set(dictionary[word])) + 1))
        word_idf_array.append(idf)
    # generate the word-document matrix
    X = np.zeros([len(unique_words), len(documents)])
    for j in range(0, len(documents)):
        n_dWord = len(documents[j])
        # calculate the tf value of each word
        for i in range(0, len(unique_words)):
            tf = documents[j].count(unique_words[i]) / n_dWord
            X[i][j] = tf * word_idf_array[i]

    # singular value decomposition
    U, sigma, V = linalg.svd(X, full_matrices=False)

    targetDimension = 5
    U2 = U[0:, 0:targetDimension]
    V2 = V[0:targetDimension, 0:]
    sigma2 = np.diag(sigma[0:targetDimension])

    # save the matrix
    np.savetxt(V2_save_path, V2, delimiter=',')
    np.savetxt(U2_save_path, U2, delimiter=',')
    np.savetxt(sigma2_save_path, sigma2, delimiter=',')
    with open(word_index_save_path, "wb") as tf:
        pickle.dump(word_index, tf, pickle.HIGHEST_PROTOCOL)
    with open(dictionary_save_path, "wb") as tf:
        pickle.dump(dictionary, tf, pickle.HIGHEST_PROTOCOL)

    return U2, V2, word_index


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
    with open(data_path, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        clean_reviews = []
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
        tokenizer.do_lower_case = True
        for line in tqdm(lines):
            id, code = line.split('\t')
            code = " ".join(split_identifier_into_parts(code))
            reviews = [code]
            # tokenize, lower, remove stop words, stem, then only keep alphabets in the string
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

        U2, V, word_index = lsa(clean_reviews, saved_model_path)
        # transpose the matrix
        V2 = [[V[j][i] for j in range(len(V))] for i in range(len(V[0]))]

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, "train.txt"), 'w', encoding="utf-8") as writer:
            id = 0

            print("calculating similarity...")

            for i in range(len(clean_reviews)):
                answers = []

                unique_word_array = []
                for word in clean_reviews[i]:
                    if word not in unique_word_array:
                        unique_word_array.append(word)

                doc_cos = np.array(V2[i])
                cos_array = []

                for word in unique_word_array:
                    index = word_index[word]
                    word_cos = np.array(U2[index])
                    cos_array.append(doc_cos.dot(word_cos) / (np.linalg.norm(doc_cos) * np.linalg.norm(word_cos)))

                top_words = np.array(cos_array).argsort()[-10:][::-1]
                for w in top_words:
                    if unique_word_array[w] not in answers:
                        answers.append(unique_word_array[w])
                answers = answers[0:10]

                writer.write(str(id) + "\t" + " ".join(answers) + "\n")
                id += 1