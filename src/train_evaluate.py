import os
import logging
import datetime
import random
from typing import List

import pickle
import numpy as np
import pandas as pd

from src.graph_builder import GraphBuilder
from src.benchmark import Benchmark
from config import G_EXP_VERSION, T_CV_FOLDS


class TrainEvaluate:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def run(self):
        graphs_df, test_indexes = GraphBuilder(self.dataset_name).load_graphs()
        logging.info('Prepare train-test split or cross-validation')
        test_indexes = self.prepare_test_idx_lst(test_indexes, graphs_df)
        # graphs_df = graphs_df.drop("index", axis=1)

        logging.info('GoW benchmark')
        datetime_now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        results_folder = "".join(["results/", G_EXP_VERSION, "/", self.dataset_name, "_", datetime_now, "/"])
        os.makedirs(results_folder, exist_ok=True)
        GoW_benchmark = Benchmark(graphs_df, test_indexes, results_folder)
        GoW_benchmark.benchmark()
        logging.info('Printing results')
        self.print_results(results_folder)

    def prepare_test_idx_lst(self, test_indexes, all_data):
        if all_data.shape[0]==1:
            if test_indexes is None:
                index_lst = all_data["index"][0]
                index_lst = [idx for idx in index_lst if idx != -1]
                test_indexes = self._cross_validation(index_lst, nFolds=T_CV_FOLDS)
            else:
                return test_indexes
        elif test_indexes is None:
            test_indexes = self._cross_validation(list(all_data.index), nFolds=T_CV_FOLDS)
        else:
            test_indexes = all_data["index"].str.contains("test", regex=False)
            test_indexes.fillna(False, inplace=True)
            test_indexes = [all_data.loc[test_indexes, :].index]
        return test_indexes

    def _cross_validation(self, indexes: List[int], nFolds):
        random.seed()
        random.shuffle(indexes)
        indexes_splitted = [x.tolist() for x in np.array_split(indexes, nFolds)]
        return indexes_splitted

    @staticmethod
    def print_results(folder):
        files = os.listdir(folder)
        results_lst = []
        # history_lst = []
        for file_name in files:
            with open(folder + file_name, "rb") as file:
                result_df, description_str, history_obj = pickle.load(file)
                logging.info(description_str)
                results_lst.append(result_df)
                # history_lst.append(history_obj)
        results_df = pd.concat(results_lst, axis=0)
        results_grouped_df = results_df.reset_index().groupby("index").agg(
            train_acc=("train_acc", np.mean),
            test_acc=("test_acc", np.mean),
            training_time=("training_time", np.mean),
            evaluation_time=("evaluation_time", np.mean),
            device=("device", np.unique)
        )
        print(results_grouped_df.to_markdown())
        return results_grouped_df

    @staticmethod
    def print_results_all_datasets(folder):
        folders = os.listdir(folder)
        for dataset_folder in folders:
            logging.critical("")
            logging.critical(dataset_folder.split('_')[0])
            res = TrainEvaluate.print_results(folder + dataset_folder + "/")
            logging.critical("".join(["Best: ", str(max(res.test_acc))]))
