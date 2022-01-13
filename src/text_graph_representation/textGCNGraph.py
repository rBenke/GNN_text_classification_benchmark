import itertools
import logging
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from spektral.data import Graph


class TextGCNGraph:

    def __init__(self, text_vectorization_method: str = None, with_connections=True):
        self.with_connections = with_connections

    def transform(self, texts: pd.DataFrame):

        X_tokenized_lst = texts["content"]

        useful_tokens = np.unique([x for token_list in X_tokenized_lst for x in token_list])
        # create word-word submatrix of adjacency matrix
        word_word_adj = self.word_word_matrix(X_tokenized_lst, useful_tokens)
        # Create word-document matrix
        doc_word_adj = self.doc_word_matrix(X_tokenized_lst, useful_tokens)
        word_doc_adj = doc_word_adj.T
        # Create document-document matrix
        doc_doc_adj = np.zeros((len(X_tokenized_lst), len(X_tokenized_lst)))
        # Create Graph Adjacency matrix
        #
        #                 word_word_adj | word_doc_adj
        #  Adj_matrix =   -----------------------------
        #                 doc_word_adj  | doc_doc_adj

        col1 = np.row_stack((word_word_adj, doc_word_adj.toarray()))
        del word_word_adj, doc_word_adj
        col2 = np.row_stack((word_doc_adj.toarray(), doc_doc_adj))
        del word_doc_adj, doc_doc_adj
        graph_adj = np.column_stack((col1, col2))
        graph_adj = sp.csr_matrix(graph_adj)
        graph_adj = graph_adj.astype(np.float32)
        logging.debug(f"Graph adjacency matrix shape: {graph_adj.shape}")
        # create feature matrix
        identity_matrix = np.eye(graph_adj.shape[0])
        identity_matrix = identity_matrix.astype(np.float32)
        # create labels
        label_lst = texts[set(texts.columns).difference({'content', 'index'})]

        docs_label_arr = np.array(label_lst)
        words_label_arr = np.zeros((len(useful_tokens), docs_label_arr.shape[1]))
        label_arr = np.concatenate([words_label_arr, docs_label_arr], axis=0)
        label_arr = label_arr.astype(np.float32)
        logging.debug(f"Label_array shape: {docs_label_arr.shape}")

        textGraph = Graph(x=identity_matrix, a=graph_adj, y=label_arr)
        graph_lst = [textGraph]

        word_index = [-1] * len(useful_tokens)
        index_lst = word_index + texts["index"].tolist()
        return pd.DataFrame({"graphs": graph_lst, "index": [index_lst]})

    def doc_word_matrix(self, X_tokenized_lst, useful_tokens):
        tfidf = TfidfVectorizer(vocabulary=useful_tokens)
        filtered_text = [" ".join(text) for text in X_tokenized_lst]
        doc_word_adj = tfidf.fit_transform(filtered_text)
        return doc_word_adj

    def word_word_matrix(self, tokenized_texts, words_order):
        unigram_freq = pd.value_counts(list(itertools.chain.from_iterable(tokenized_texts)))
        unigram_freq = unigram_freq[words_order]
        unigram_prob = unigram_freq / float(sum(unigram_freq))
        unigram_prob_matrix = np.matmul(np.expand_dims(unigram_prob, 1), np.expand_dims(unigram_prob, 1).T)

        ngrams_lst = [list(ngrams(text, 20)) for text in tokenized_texts]
        ngrams_pairs = [itertools.product(ngram,ngram) for ngram in itertools.chain.from_iterable(ngrams_lst)]
        n_ngrams = len(ngrams_pairs)
        ngrams_pairs_freq_set = Counter()
        for bigram_lst in ngrams_pairs:
            ngrams_pairs_freq_set.update(bigram_lst)
            del bigram_lst
        # bigram_freq_set = {}
        # for bigram, freq in bigram_freq.iteritems():
        #     bigram_freq_set[bigram] = freq

        words_order_dict = dict(zip(words_order,range(len(words_order))))
        bigram_matrix = np.zeros((len(words_order), len(words_order)), dtype=np.float32)
        for bigram, count in ngrams_pairs_freq_set.items():
            word1, word2 = bigram
            if word1 == word2:
                continue
            word1_idx, word2_idx = words_order_dict[word1], words_order_dict[word2]
            bigram_matrix[word1_idx, word2_idx] += count
            bigram_matrix[word2_idx, word1_idx] += count

        bigram__prob_matrix = bigram_matrix / (n_ngrams*n_ngrams)

        adj_matrix = np.divide(bigram__prob_matrix, unigram_prob_matrix)
        adj_matrix[adj_matrix>0] = np.log(adj_matrix[adj_matrix>0])
        adj_matrix[adj_matrix < 0] = 0
        np.fill_diagonal(adj_matrix, 0)
        return (adj_matrix)