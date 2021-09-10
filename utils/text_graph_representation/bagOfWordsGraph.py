import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import ngrams
from config import FEATURE_SIZE
from collections import Counter
import networkx as nx

from spektral.data import Graph

class BagOfWordsGraph:

    def __init__(self, with_connections: bool = True):
        self.with_connections = with_connections

    def fit_transform(self, texts: pd.DataFrame):
        index_lst = []
        graph_lst = []
        selected_tokens = self._select_tokens(texts.content)
        for _, row in tqdm(texts.iterrows(), total=texts.shape[0]):
            text = row["content"]
            text = [token for token in text if token in selected_tokens]
            idx = row["index"]
            label = row[set(row.index).difference({'content', 'index'})].to_numpy().astype(np.float32)
            # edges
            if self.with_connections:
                bigrams = set(ngrams(text, 2))
                edge_df = pd.DataFrame(bigrams, columns=["source", "target"])
                G = nx.from_pandas_edgelist(edge_df)
                node_list = [token for token in selected_tokens if token in text]
                adjacency_matrix = nx.adjacency_matrix(G, nodelist=node_list).astype(np.float32)
            else:
                adjacency_matrix = np.zeros((len(np.unique(text)), len(np.unique(text)))).astype(np.float32)
            # nodes
            text_vector = [0] * len(selected_tokens)
            for token in text:
                text_vector[selected_tokens.index(token)] += 1
            text_vector = np.array(text_vector)
            text_vector = text_vector / np.sqrt(np.sum(text_vector ** 2))
            feature_matrix = np.diag(text_vector).astype(np.float32)
            feature_matrix = feature_matrix[feature_matrix.sum(axis=1) > 0, :]
            assert adjacency_matrix.shape[0]==feature_matrix.shape[0]
            # assert adjacency_matrix.shape[0]>0
            if adjacency_matrix.shape[0]==0:
                continue
            # create graph
            textGraph = Graph(x=feature_matrix, a=adjacency_matrix, y=label)
            graph_lst.append(textGraph)
            index_lst.append(idx)
        return pd.DataFrame({"graphs":graph_lst, "index": index_lst})

    def _select_tokens(self, texts):
        all_tokens = [token for text in texts for token in text]
        counter = Counter(all_tokens)
        selected_tokens = counter.most_common(FEATURE_SIZE)
        selected_tokens_lst = [token for token, _ in selected_tokens]
        return selected_tokens_lst
