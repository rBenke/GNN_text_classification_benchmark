import datetime
import os
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from helpers import parse_args
from utils.text_processing import Preprocess
from utils.text_graph_representation.graphOfWords import GraphOfWords
from utils.text_graph_representation.bagOfWordsGraph import BagOfWordsGraph
from utils.GNN.benchmark import Benchmark
from helpers import load_dataset, prepare_test_idx_lst, print_results
from config import *
if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL)
    multiprocessing.set_start_method('spawn')
    args = parse_args()

    logging.info('Loading raw data')
    all_data, test_indexes = load_dataset(args.dataset_name)
    # TODO check if graph are prepared and create them only if necessary

    logging.info('Text preprocessing')
    preprocess = Preprocess()
    preprocess_steps = ["remove_unicode","lower", "clean", "tokenize", "stopwordsNltk", "alpha_words_only", "lemmatize"]
    all_data["content"] = preprocess.transform(all_data.content, preprocess_steps)

    logging.info('Transform labels to onehot encoding')
    onehot_encoding = OneHotEncoder()
    categories_arr = all_data.category.to_numpy()[:, np.newaxis]
    onehot_categories = onehot_encoding.fit_transform(categories_arr).toarray()
    all_data = pd.concat([all_data.content.reset_index(drop=False), pd.DataFrame(onehot_categories)], axis=1)

    logging.info('Loading token vectorization model')
    # graphOfWords = GraphOfWords()
    bagOfWordsGraph = BagOfWordsGraph(with_connections = True)
    logging.info('Text to graph transormation')
    # graphs_df = graphOfWords.transform(all_data)
    graphs_df = bagOfWordsGraph.fit_transform(all_data)

    logging.info('Prepare train-test split or cross-validation')
    test_indexes = prepare_test_idx_lst(test_indexes, graphs_df)
    graphs_df = graphs_df.drop("index", axis=1)

    logging.info('GoW benchmark')
    datetime_now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    results_folder = "".join(["results/", EXP_VERSION, "/", args.dataset_name, "_", datetime_now, "/"])
    os.makedirs(results_folder, exist_ok=True)
    GoW_benchmark = Benchmark(graphs_df, test_indexes, results_folder)
    GoW_benchmark.benchmark()
    logging.info('Printing results')
    print_results(results_folder)
    ## create textGCN
    ### preprocess textual data

    # learn&validate GNN

# TODO:
#  - implement all GNN models
#  - implement a few architectures for every GNN

