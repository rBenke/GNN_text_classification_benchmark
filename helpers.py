import os
import pickle
import argparse
import logging
import random
import numpy as np
import pandas as pd
from typing import List
from utils.load_data import load_files, load_splitted_files
from config import CV_FOLDS


def parse_args():
    parser = argparse.ArgumentParser(
        description="")

    # parser.add_argument(
    #     "--dataset_path",
    #     dest="dataset_path",
    #     required=True,
    #     help="Path to dataset")
    #
    # parser.add_argument(
    #     "--models_path",
    #     dest="models_path",
    #     required=True,
    #     help="Path to models")
    #
    # parser.add_argument(
    #     "--pretrained_path",
    #     dest="pretrained_path",
    #     required=True,
    #     help="Path to pretrained embedding model")

    parser.add_argument(
        "--dataset_name",
        dest="dataset_name",
        choices=['bbc', 'bbcsport', '20newsgroups', 'reuters', 'ohsumed'],
        required=False,
        help="Name of dataset")

    args = parser.parse_args()
    return args


def load_dataset(dataset_name):
    if dataset_name == "bbc":
        logging.info('Loading BBC dataset')
        all_data = load_files("data/textual/bbc/")
        return all_data, None
    elif dataset_name == "bbcsport":
        logging.info('Loading BBC_sport dataset')
        all_data = load_files("data/textual/bbcsport/")
        return all_data, None
    elif dataset_name == "20newsgroups":
        logging.info('Loading 20newsgroups dataset')
        all_data = load_files("data/textual/20_newsgroups/")
        return all_data, None
    elif dataset_name == "reuters":
        logging.info('Loading Reuters dataset')
        all_data, test_idx = load_splitted_files("data/textual/Reuters21578-Apte-90Cat/")
        return all_data, test_idx
    elif dataset_name == "ohsumed":
        logging.info('Loading Ohsumed dataset')
        all_data, test_idx = load_splitted_files("data/textual/ohsumed-first-20000-docs/")
        return all_data, test_idx
    else:
        raise AttributeError("Selected dataset is not available.")


def cross_validation(indexes: List[int], nFolds):
    random.seed()
    random.shuffle(indexes)
    indexes_splitted = [x.tolist() for x in np.array_split(indexes, nFolds)]
    return indexes_splitted


def prepare_test_idx_lst(test_indexes, all_data):
    if test_indexes is None:
        test_indexes = cross_validation(list(all_data.index), nFolds=CV_FOLDS)
    else:
        test_indexes = all_data["index"].str.contains("test", regex=False)
        test_indexes.fillna(False, inplace=True)
        test_indexes = [all_data.loc[test_indexes, :].index]
    return test_indexes

def print_results(folder):
    files = os.listdir(folder)
    results_lst = []
    # history_lst = []
    for file_name in files:
        with open(folder+file_name, "rb") as file:
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

# 20newsgroups : 0.78
# bbc : 0.976
# bbcsport: 0.992
# ohsumed: 0.45
# reuters: 0.737