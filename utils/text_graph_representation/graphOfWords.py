from typing import List
from nltk.util import ngrams
from stellargraph import StellarGraph
import pandas as pd
from tqdm import tqdm

from utils.text_vectorization import Word2vec


class GraphOfWords:

    def __init__(self, text_vectorization_method: str = "word2vec"):
        if text_vectorization_method == "word2vec":
            self.text_vectorizer = Word2vec()
        else:
            raise Exception("".join(["Selected method (", text_vectorization_method, ") is not implemented"]))

    def transform(self, texts: List[List[str]]):
        graph_lst = []
        for text in tqdm(texts):
            # edges
            bigrams = set(ngrams(text, 2))
            edge_df = pd.DataFrame(bigrams, columns=["source", "target"])
            # nodes
            all_unique_tokens = list(set(text))
            tokens_df = self.text_vectorizer.transform(all_unique_tokens)
            # create graph
            textGraph = StellarGraph({"tokens": tokens_df}, {"edges": edge_df})
            graph_lst.append(textGraph)
        return graph_lst

    # def text2WoG(self, text: List[str]):
    #     # extract bigrams
    #     bigrams = set(ngrams(text, 2))
    #     edge_df = pd.DataFrame(bigrams)
    #     # create graph
    #     G = nx.from_pandas_edgelist(edge_df, 0, 1)
    #     return G
