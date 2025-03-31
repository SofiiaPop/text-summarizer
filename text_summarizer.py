import re
import nltk
import random
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

# pylint: disable=C0200

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
    
    def performSVD(self):
        m, n = self.tf_idf_matrix.shape
        a = self.tf_idf_matrix
        aT = self.tf_idf_matrix.T
        aaT = a @ aT
        aTa = aT @ a

        ev, evc = np.linalg.eig(aTa)
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

    # def matrixU(self):
    #     a = self.tf_idf_matrix
    #     aT = self.tf_idf_matrix.T
    #     aaT = a @ aT
        
    #     EVaaT, EVcaaT = np.linalg.eig(aaT)
    #     ncols = np.argsort(EVaaT)[::-1]
    #     u = EVcaaT[:,ncols]
    #     return u

    # def marixSum(self):
    #     a = self.tf_idf_matrix
    #     aT = self.tf_idf_matrix.T
    #     aaT = a @ aT
    #     aTa = aT @ a
    #     matrix = None
    #     if (np.size(aaT) > np.size(aTa)):
    #         matrix = aTa
    #     else:
    #         matrix = aaT
    #     ev, evc = np.linalg.eig(matrix)
    #     singular_vals = np.sqrt(ev)
    #     singular_vals = np.sort(singular_vals)[::-1]
    #     s = np.zeros_like(a, dtype=float)
    #     np.fill_diagonal(s, singular_vals)
    #     return s

    # def matrixV(self):
    #     a = self.tf_idf_matrix
    #     aT = self.tf_idf_matrix.T
    #     aTa = aT @ a
    #     EVaTa, EVcaTa = np.linalg.eig(aTa)
    #     ncols = np.argsort(EVaTa)[::-1]
    #     vT = EVcaTa[:,ncols].T
    #     return vT


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
        return processed_sent

# with open("text2.txt", "r", encoding="UTF-8") as file:
#     text = " ".join(file.readlines())

def test_svd(num_tests=100, threshold=1e-5):
    failed = 0

    for i in range(num_tests):
        shape=(random.randint(1, 100), random.randint(1, 100))
        matrix = np.random.rand(*shape)

        try:
            # Custom SVD
            svd_custom = SVD(matrix, sents=[])
            A_custom = svd_custom.U @ svd_custom.S @ svd_custom.V

            # Numpy SVD for reference
            U_lib, S_lib, Vt_lib = np.linalg.svd(matrix, full_matrices=False)
            A_lib = U_lib @ np.diag(S_lib) @ Vt_lib

            # Check reconstruction error for both
            err_custom = np.linalg.norm(matrix - A_custom)
            err_lib = np.linalg.norm(matrix - A_lib)

            if err_custom > threshold:
                print(f"Test {i+1}: FAIL | Custom error: {err_custom:.4e} | Library error: {err_lib:.4e}")
                failed += 1
        except Exception as e:
            print(f"Test {i+1}: EXCEPTION - {e}")
            failed += 1
        
        print(shape)
    print(f"\nTotal tests: {num_tests}")
    print(f"Total failures (custom SVD error > {threshold}): {failed}")
    

def score_sentences_by_v(V, S, tokenized_sentences, num_of_sent):

    if V.shape[0] == 0 or S.shape[0] == 0:
        print("SVD returned empty matrix. Check the input text or preprocessing.")
        return []

    first_col_v = V[0]
    largest_singular_value = S[0, 0]

    sentence_scores = first_col_v * largest_singular_value

    ranked_sentences = sorted(
        enumerate(sentence_scores),
        key=lambda x: x[1],
        reverse=True
    )

    result = [
        (idx, score, " ".join(tokenized_sentences[idx]))
        for idx, score in ranked_sentences
        if idx < len(tokenized_sentences)  # ✅ avoid IndexError here
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
    # print(preprocess.text)
    tokenized_sentences = preprocess.text
    tf_idf = TF_IDF(tokenized_sentences)
    # print(tf_idf.matrix)


    svd = SVD(tf_idf.matrix, tokenized_sentences)

    _, S, V = svd.performSVD()
    top_sentences = score_sentences_by_v(V, S, tokenized_sentences, num_of_sent=num_sent)

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, _, _ in top_sentences:
            f.write(text.split(".")[idx].strip() + ".\n")

if __name__ == "__main__":
    main()

# Example run
# test_svd(num_tests=50)


# #print(text, '\n-----')
# preprocess = Preprocess(text)
# #print(preprocess.text)

# # print(preprocess.text)
# tf_idf = TF_IDF(preprocess.text)
# # print(tf_idf.sents)
# # print(tf_idf.matrix)

# test_matrix1 = [[4, 2, 0], [1, 5, 6]]
# test_matrix2 = [[3, 2, 2], [2, 3, -2]]
# test_matrix3 = [[1, 2, 0], [5, 0, 2], [8, 5, 4], [6, 9, 7]]

# svd = SVD(test_matrix1, tf_idf.sents)
# print(svd.U)
# print(svd.S)
# print(svd.V)
# print("--------Mine approach result-------")
# print(svd.U @ svd.S @ svd.V, "\n---------")


# U, S, D = np.linalg.svd(test_matrix1)
# print(U)
# s = np.zeros_like(test_matrix1, dtype=float)
# np.fill_diagonal(s, S)
# print(s)
# print(D)
# print(U@s@D)
