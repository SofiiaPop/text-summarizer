import re
import nltk
import random

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from math import log
import numpy as np


class SVD():
    def __init__(self, matrix, sents):
        self.tf_idf_matrix = np.array(matrix)
        self.sentences = sents
        self.U, self.S, self.V = self.performSVD()


    def find_eigen(self, A, num_iterations=1000, tolerance=1e-6):
        n = A.shape[0]
        b_k = np.random.rand(n)

        for _ in range(num_iterations):
            b_k1 = np.dot(A, b_k)

            b_k1_norm = np.linalg.norm(b_k1)
            b_k1 = b_k1 / b_k1_norm

            if np.linalg.norm(b_k1 - b_k) < tolerance:
                break
            b_k = b_k1

        eigenvalue = np.dot(b_k.T, np.dot(A, b_k)) / np.dot(b_k.T, b_k)

        return eigenvalue, b_k

    def performSVD(self):
        m, n = self.tf_idf_matrix.shape
        a = self.tf_idf_matrix
        aT = self.tf_idf_matrix.T
        aTa = aT @ a

        ev = np.zeros(n)
        evc = np.zeros((n, n))

        for i in range(n):
            ev[i], evc[:, i] = self.find_eigen(aTa)

        sord_idx = np.argsort(ev)[::-1]
        ev = ev[sord_idx]

        v = evc[:, sord_idx]
        singular_values = np.sqrt(np.abs(ev))

        u = np.zeros((m, n))

        for i in range(n):
            sigma = singular_values[i]
            if (sigma > 1e-10):
                u[:, i] = a @ v[:, i] * (1/sigma)
            else:
                u[:, i] = np.zeros(m)

        s = np.diag(singular_values[:n])
        return u, s, v.T

class TF_IDF():
    def __init__(self, sents):
        self.sents = sents
        self.matrix = self.tf_idf(sents)

    def tf_idf(self, sents):
        sents_num = len(sents)
        unique_words = []
        for sent in sents:
            for word in sent:
                if word not in unique_words:
                    unique_words.append(word)
        matrix = [[0.0 for i in range(len(unique_words))] for j in range(sents_num)]
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                sentence = sents[row]
                word = unique_words[col]
                tf = sentence.count(word) / len(sentence)
                num_sent_has_word = sum(1 for sent in sents if word in sent)
                idf = log(sents_num/num_sent_has_word)
                matrix[row][col] = tf * idf
        return matrix

class Preprocess():
    def __init__(self, t):
        self.lemmatizer = WordNetLemmatizer()
        self.text = self.preprocess(t)

    def preprocess(self, t):
        text = t.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\\n', ' ', text)
        text = text.strip()

        sentences = sent_tokenize(text)
        stop_words = set(stopwords.words("english"))

        processed_sent = []
        for sent in sentences:
            words = word_tokenize(sent)
            words = [re.sub(r"[^a-zA-Z0-9]", "", word) for word in words]
            words = [self.lemmatizer.lemmatize(word) for word in words if word not in stop_words and word != ""]
            processed_sent.append(words)
        return processed_sent, sentences

def score_sentences_by_u(U, S, tokenized_sentences, num_of_sent):
    if U.shape[0] == 0 or S.shape[0] == 0:
        print("SVD returned empty matrix. Check the input text or preprocessing.")
        return []

    first_col_u = U[0]
    largest_singular_value = S[0, 0]

    sentence_scores = first_col_u * largest_singular_value

    ranked_sentences = sorted(
        enumerate(sentence_scores),
        key=lambda x: x[1],
        reverse=True
    )

    result = [
        (idx, score, " ".join(tokenized_sentences[idx]))
        for idx, score in ranked_sentences
        if idx < len(tokenized_sentences)
    ]

    return result[:num_of_sent]


def main():
    file_name = input("Enter the filename to read from: ").strip()
    num_sent = int(input("Enter number of top sentences to return: ").strip())
    output_file  = input("Enter the filename to write: ").strip()

    try:
        with open(file_name, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return

    preprocess = Preprocess(text)
    tokenized_sentences, sentences = preprocess.text
    tf_idf = TF_IDF(tokenized_sentences)
    svd = SVD(tf_idf.matrix, tokenized_sentences)

    U, S, _ = svd.performSVD()
    top_sentences = score_sentences_by_u(U, S, tokenized_sentences, num_of_sent=num_sent)

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, _, _ in top_sentences:
            f.write(sentences[idx].strip() + "\n")

if __name__ == "__main__":
    main()
