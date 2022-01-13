from typing import List
import pandas as pd
import numpy as np
from config import GB_W2V_VERSION
def tf_idf():
    raise NotImplementedError

class Word2vec:
    def __init__(self):
        self._load_word2vec()

    def transform(self, tokens: List[str]):
        vectorized_tokens_lst = []
        for token in tokens:
            try:
                vectorized_token = self.model[token]
            except KeyError:
                # print(token)
                vectorized_token = self._find_representation_for(token)
            vectorized_tokens_lst.append(vectorized_token[np.newaxis,:])
        vectorized_tokens_array = np.concatenate(vectorized_tokens_lst, axis=0)
        vectorized_tokens_df = pd.DataFrame(vectorized_tokens_array, index = tokens)
        return vectorized_tokens_df

    def _find_representation_for(self, token: str):
        return np.zeros(self.model.vector_size)

    def _load_word2vec(self):
        # -- feature matrix
        import gensim.downloader
        # Load vectors directly from the file
        self.model = gensim.downloader.load(GB_W2V_VERSION)
        # >>> print(list(gensim.downloader.info()['models'].keys()))
        # >>> ['fasttext-wiki-news-subwords-300',
        #  'conceptnet-numberbatch-17-06-300',
        #  'word2vec-ruscorpora-300',
        #  'word2vec-google-news-300',
        #  'glove-twitter-25'
        #  'glove-wiki-gigaword-50',
        #  'glove-wiki-gigaword-100',
        #  'glove-wiki-gigaword-200',
        #  'glove-wiki-gigaword-300',
        #  'glove-twitter-25',
        #  'glove-twitter-50',
        #  'glove-twitter-100',
        #  'glove-twitter-200',
        #  '__testing_word2vec-matrix-synopsis']
        return None

