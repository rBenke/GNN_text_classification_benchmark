import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from helpers import parse_args
from utils.text_processing import Preprocess
from utils.text_graph_representation.graphOfWords import GraphOfWords
from utils.GNN.benchmark import Benchmark
from helpers import load_dataset, cross_validation

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    logging.info('Loading raw data')
    all_data, test_indexes = load_dataset(args.dataset_name)
    # TODO check if graph are prepared and create them only if necessary

    logging.info('Text preprocessing')
    preprocess = Preprocess()
    preprocess_steps = ["lower", "tokenize"]  # clean
    all_data["content"] = preprocess.transform(all_data.content, preprocess_steps)

    logging.info('Loading token vectorization model')
    graphOfWords = GraphOfWords()
    logging.info('Text to graph transormation')
    all_data["graphs"] = graphOfWords.transform(all_data.content)

    logging.info('Transform labels to onehot encoding')
    onehot_encoding = OneHotEncoder()
    categories_arr = all_data.category.to_numpy()[:, np.newaxis]
    onehot_categories = onehot_encoding.fit_transform(categories_arr).toarray()
    all_data = pd.concat([all_data.graphs.reset_index(drop=False), pd.DataFrame(onehot_categories)], axis=1)

    logging.info('Prepare train-test split or cross-validation')
    if test_indexes is None:
        test_indexes = cross_validation(list(all_data.index), nFolds=2)
    else:
        print(len(test_indexes))
        test_indexes = all_data["index"].str.contains("test", regex=False)
        test_indexes.fillna(False, inplace=True)
        test_indexes = [all_data.loc[test_indexes, :].index]
        print(len(test_indexes))
    all_data.drop("index", axis=1, inplace=True)

    logging.info('GoW Benchmark')
    GoW_benchmark = Benchmark(all_data, test_indexes)
    GoW_benchmark.benchmark()
    ## create textGCN
    ### preprocess textual data

    # learn&validate GNN

# TODO:
#  - implement all GNN models
#  - write prints and result dumps with all details
#  - implement a few architectures for every GNN
# TODO, after 15.08
#  - implement corpusGraph
