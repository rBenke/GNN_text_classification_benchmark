from typing import List
from sklearn.model_selection import train_test_split
from config import VALIDATION_SIZE

def train_valid_split(indexes: List[int]):
    return train_test_split(indexes, test_size=VALIDATION_SIZE)
