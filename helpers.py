import argparse
import logging
import random
import numpy as np
from typing import List
from utils.load_data import load_files, load_splitted_files

def parse_args():
    parser =  argparse.ArgumentParser(
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
        choices=['bbc','bbcsport','20newsgroups', 'reuters', 'ohsumed'],
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

def cross_validation(indexes: List[int], nFolds = 5):
    random.shuffle(indexes)
    indexes_splitted = [x.tolist() for x in np.array_split(indexes, nFolds)]
    return indexes_splitted
