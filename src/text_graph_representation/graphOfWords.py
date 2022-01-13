import numpy as np
from nltk.util import ngrams
import pandas as pd
from tqdm import tqdm
import networkx as nx

from spektral.data import Graph
from src.utils.text_vectorization import Word2vec


class GraphOfWords:

    def __init__(self, text_vectorization_method: str = "word2vec", with_connections=True):
        self.with_connections = with_connections
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

            # nodes
            all_unique_tokens = list(set(text))
            tokens_df = self.text_vectorizer.transform(all_unique_tokens)
            feature_matrix = tokens_df.to_numpy().astype(np.float32)
            # edges
            if self.with_connections is True:
                bigrams = set(ngrams(text, 2))
                edge_df = pd.DataFrame(bigrams, columns=["source", "target"])
                G = nx.from_pandas_edgelist(edge_df)
                adjacency_matrix = nx.adjacency_matrix(G, nodelist=all_unique_tokens).astype(np.float32)
            elif self.with_connections == "random":
                adjacency_matrix = np.random.choice([0, 1],
                                                    size=(len(np.unique(text)), len(np.unique(text))),
                                                    p=[4./5, 1./5])
                np.fill_diagonal(adjacency_matrix,0)
            else:
                adjacency_matrix = np.zeros((len(np.unique(text)), len(np.unique(text)))).astype(np.float32)

            # create graph
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
