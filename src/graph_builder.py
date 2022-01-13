import os
import logging
from hashlib import md5

import pickle
import numpy as np

from collections import namedtuple

from config import P_PREPROCESSING_STEPS, GB_GRAPH_TYPE, GB_W2V_VERSION, GB_FEATURE_SIZE # everything here should
# be added to the graph_description attribute
from src.preprocess import DatasetPreprocessing
from src.text_graph_representation.bagOfWordsGraph import BagOfWordsGraph
from src.text_graph_representation.graphOfWords import GraphOfWords
from src.text_graph_representation.textGCNGraph import TextGCNGraph

Graph_description = namedtuple('Graph_description', ['preprocessing_steps', 'graph_type',
                                                     "vectorization", "feature_size"])

class GraphBuilder:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.graph_description = Graph_description(
            P_PREPROCESSING_STEPS, GB_GRAPH_TYPE, GB_W2V_VERSION, GB_FEATURE_SIZE)

    def is_computed(self):
        graphs_version_hash = self.get_graphs_version()
        output_dir = "".join(["data/graphs/", self.dataset_name, "/", graphs_version_hash])
        metadata_file_exists = os.path.isfile(output_dir + "/usage_log.txt")
        if not metadata_file_exists:
            return False

        with open(output_dir + "/usage_log.txt", "r") as file:
            metadata_file = file.readlines()
        is_the_same_processing = metadata_file[1][:-1] == str(self.graph_description)
        return is_the_same_processing

    def build(self):
        data_df, test_indexes = DatasetPreprocessing(self.dataset_name).load_preprocessed_data()
        logging.info('Loading token vectorization model')
        if GB_GRAPH_TYPE == "GraphOfWords":
            g_builder = GraphOfWords(with_connections=True)
        elif GB_GRAPH_TYPE == "BagOfWordsGraph":
            g_builder = BagOfWordsGraph(with_connections=True)
        elif GB_GRAPH_TYPE == "TextGCNGraph":
            g_builder = TextGCNGraph(with_connections=True)
        else:
            raise ValueError("Selected GB_GRAPH_TYPE: " + self.graph_description.graph_type + " is not implemented yet.")

        logging.info('Text to ' + GB_GRAPH_TYPE + ' transormation')
        graphs_df = g_builder.transform(data_df)

        logging.info('Saving graphs.')
        graphs_version_hash = self.get_graphs_version()
        output_dir = "".join(["data/graphs/", self.dataset_name, "/", graphs_version_hash])
        os.makedirs(output_dir, exist_ok=True)
        # Save graphs
        pickle.dump(graphs_df, open(output_dir + "/data.pickle", "wb"))
        # Save labels
        pickle.dump(test_indexes, open(output_dir + "/labels.pickle", "wb"))
        # Save metadata
        with open(output_dir + "/usage_log.txt", "w") as file:
            file.write("PREPROCESSING STEPS AND TEXT TO GRAPH METHOD: \n")
            file.write(str(self.graph_description))
            file.write("\nUSAGE:\n")

        return graphs_df, test_indexes

    def load_graphs(self):
        if self.is_computed():
            output_dir = "".join(["data/graphs/", self.dataset_name, "/", self.get_graphs_version()])
            graph_data = pickle.load(open(output_dir + "/data.pickle", "rb"))
            test_indexes = pickle.load(open(output_dir + "/labels.pickle", "rb"))
            return graph_data, test_indexes
        else:
            return self.build()

    def get_graphs_version(self):
        hash_function = md5()
        hash_function.update(str(self.graph_description).encode("ascii"))
        graphs_version_hash = hash_function.hexdigest()
        graphs_version_hash_str = np.base_repr(int(graphs_version_hash, 16), base=36)

        return graphs_version_hash_str
