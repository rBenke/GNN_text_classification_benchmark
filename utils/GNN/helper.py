from typing import List
from sklearn.model_selection import train_test_split


def train_valid_split(indexes: List[int]):
    return train_test_split(indexes, test_size=0.2, random_state=42)
