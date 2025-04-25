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
        """
        Find the dominant eigenvalue and eigenvector of a matrix A using the power iteration method.
        :param A: Input matrix
        :param num_iterations: Number of iterations to perform
        :param tolerance: Tolerance for convergence
        :return: Dominant eigenvalue and eigenvector
        """
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
        """
        Perform Singular Value Decomposition (SVD) on the TF-IDF matrix.
        :return: U, S, V matrices
        """
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
            if sigma > 1e-10:
                u[:, i] = a @ v[:, i] * (1/sigma)
            else:
                u[:, i] = np.zeros(m)

        s = np.diag(singular_values[:n])
        return u, s, v.T

class TF_IDF():
    """
    Class to calculate the TF-IDF matrix for a given set of sentences.
    """
    def __init__(self, sents):
        self.sents = sents
        self.matrix = self.tf_idf(sents)

    def tf_idf(self, sents):
        """
        Calculate the TF-IDF matrix for the given sentences.
        :param sents: List of tokenized sentences
        :return: TF-IDF matrix
        """
        sents_num = len(sents)
        unique_words = []
        for sent in sents:
            for word in sent:
                if word not in unique_words:
                    unique_words.append(word)
        matrix = [[0.0 for i in range(len(unique_words))] for j in range(sents_num)]
        for row, _ in enumerate(matrix):
            for col, _ in enumerate(matrix[0]):
                sentence = sents[row]
                word = unique_words[col]
                tf = sentence.count(word) / len(sentence)
                num_sent_has_word = sum(1 for sent in sents if word in sent)
                idf = log(sents_num/num_sent_has_word)
                matrix[row][col] = tf * idf
        return matrix

class Preprocess():
    """
    Class to preprocess the input text.
    """
    def __init__(self, t):
        self.lemmatizer = WordNetLemmatizer()
        self.text = self.preprocess(t)

    def preprocess(self, t):
        """
        Preprocess the input text by removing unwanted characters and tokenizing sentences.
        :param t: Input text
        :return: List of tokenized sentences
        """
        text = re.sub(r'\s+', ' ', t)
        text = re.sub(r'\\n', ' ', text)
        text = text.strip()

        sentences = sent_tokenize(text)
        stop_words = set(stopwords.words("english"))

        sentences = [s for s in sentences if not s.startswith("SECTION")\
                      and not s.startswith("ARTICLE")\
                      and not s.startswith("CHAPTER")]

        processed_sent = []
        for sent in sentences:
            sent = sent.lower()
            words = word_tokenize(sent)
            words = [re.sub(r"[^a-zA-Z0-9]", "", word) for word in words]
            words = [self.lemmatizer.lemmatize(word) \
                    for word in words \
                        if word not in stop_words and word != ""]
            processed_sent.append(words)
        return processed_sent, sentences

def score_sentences_by_u(u, s, original_sentences, num_of_sent):
    """
    Score sentences based on the first column of U and the largest singular value.
    :param U: U matrix from SVD
    :param S: S matrix from SVD
    :param original_sentences: Original sentences
    :param num_of_sent: Number of sentences to return
    :return: List of tuples containing sentence index, score, and original sentence
    """
    if u.shape[0] == 0 or s.shape[0] == 0:
        print("SVD returned empty matrix. Check the input text or preprocessing.")
        return []

    first_col_u = u[:, 0]
    largest_singular_value = s[0, 0]

    sentence_scores = first_col_u * largest_singular_value

    ranked_sentences = sorted(
        enumerate(sentence_scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_ranked = ranked_sentences[:num_of_sent]

    chronological_order = sorted(
        top_ranked,
        key=lambda x: x[0]
    )

    result = [
        (idx, score, original_sentences[idx])
        for idx, score in chronological_order
    ]

    return result

def run_text_summarization(text, sentences_count, output_file=None):
    """
    Run the text summarization process.
    :param text: Input text to summarize
    :param sentences_count: Number of sentences to return
    :param output_file: Optional output file to save the summary
    :return: Summary as a string
    """
    preprocess = Preprocess(text)
    tokenized_sentences, sentences = preprocess.text
    tf_idf = TF_IDF(tokenized_sentences)
    svd = SVD(tf_idf.matrix, tokenized_sentences)

    u, s = svd.U, svd.S
    top_sentences = score_sentences_by_u(u, s, sentences, num_of_sent=sentences_count)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as file:
            for idx, _, _ in top_sentences:
                file.write(sentences[idx].strip() + "\n")

    return "".join(sentence.strip() + "\n" for _, _, sentence in top_sentences)
