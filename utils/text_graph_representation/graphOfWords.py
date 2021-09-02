import numpy as np
from nltk.util import ngrams
import pandas as pd
from tqdm import tqdm
import networkx as nx

from spektral.data import Graph
from utils.text_vectorization import Word2vec


class GraphOfWords:

    def __init__(self, text_vectorization_method: str = "word2vec"):
        if text_vectorization_method == "word2vec":
            self.text_vectorizer = Word2vec()
        else:
            raise Exception("".join(["Selected method (", text_vectorization_method, ") is not implemented"]))

    def transform(self, texts: pd.DataFrame):
        index_lst = []
        graph_lst = []
        for _, row in tqdm(texts.iterrows(), total=texts.shape[0]):
            text = row["content"]
            idx = row["index"]
            label = row[set(row.index).difference({'content', 'index'})].to_numpy().astype(np.float32)
            # edges
            bigrams = set(ngrams(text, 2))
            edge_df = pd.DataFrame(bigrams, columns=["source", "target"])
            if edge_df.shape[0]==0:
                continue
            # nodes
            all_unique_tokens = list(set(text))
            tokens_df = self.text_vectorizer.transform(all_unique_tokens)
            # create graph
            feature_matrix = tokens_df.to_numpy().astype(np.float32)
            G = nx.from_pandas_edgelist(edge_df)
            adjacency_matrix = nx.adjacency_matrix(G, nodelist=all_unique_tokens).astype(np.float32)
            textGraph = Graph(x=feature_matrix, a=adjacency_matrix, y=label)
            graph_lst.append(textGraph)
            index_lst.append(idx)
        return pd.DataFrame({"graphs":graph_lst, "index": index_lst})

    # def text2WoG(self, text: List[str]):
    #     # extract bigrams
    #     bigrams = set(ngrams(text, 2))
    #     edge_df = pd.DataFrame(bigrams)
    #     # create graph
    #     G = nx.from_pandas_edgelist(edge_df, 0, 1)
    #     return G
