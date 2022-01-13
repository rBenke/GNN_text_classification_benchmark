import argparse
from typing import List
from sklearn.model_selection import train_test_split
from config import T_VALIDATION_SIZE


def parse_args():
    parser = argparse.ArgumentParser(
        description="")

    parser.add_argument(
        "--dataset_name",
        dest="dataset_name",
        choices=['bbc', 'bbcsport', '20newsgroups', 'reuters', 'ohsumed', "ohsumed_small"],
        required=True,
        help="Name of dataset")

    args = parser.parse_args()
    return args

def train_valid_split(indexes: List[int]):
    return train_test_split(indexes, test_size=T_VALIDATION_SIZE)
