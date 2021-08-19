import random
import numpy as np
import pandas as pd
from stellargraph.mapper import PaddedGraphGenerator
from typing import List
from utils.GNN.gcn import GCN
from utils.GNN.helper import train_valid_split

class Benchmark:
    def __init__(self, all_data: pd.DataFrame, test_idx_lst: List[List[int]]):
        self.all_data_gen = PaddedGraphGenerator(graphs=all_data["graphs"])
        self.all_data = all_data
        self.test_idx_lst = test_idx_lst
        print(all_data.columns)
        self.models = [
            GCN(self.all_data_gen, len(all_data.columns)-1)
        ]

    def benchmark(self):
        results = []
        for model in self.models:
            for test_idx in self.test_idx_lst:
                train_seq_gen, validation_seq_gen, test_seq_gen = self._create_generators(test_idx)
                # TODO implement cross-validation
                model.train(train_seq_gen, validation_seq_gen)
                model.validate(test_seq_gen)
                # results.append(model.results())
                # model.result.print_result()
                model.create_clear_model()

    def _cv_test_index(self, nObs, nFolds=5):
        all_indexes = list(range(nObs))
        random.shuffle(all_indexes)
        np.array_split(all_indexes, nFolds)

    def _create_generators(self, test_idx):
        batch_size = 30

        train_validation_idx = set(self.all_data.index).difference(test_idx)
        train_idx, validation_idx = train_valid_split(list(train_validation_idx))
        # all_indexes = np.concatenate([test_idx, train_idx, validation_idx])
        # assert all(np.unique(all_indexes, return_counts=True)[1]==1) # check for duplicates

        train_seq_gen = self.all_data_gen.flow(
            graphs=train_idx, targets=self.all_data.loc[train_idx, self.all_data.columns != "graphs"], batch_size=batch_size
        )
        validation_seq_gen = self.all_data_gen.flow(
            graphs=validation_idx, targets=self.all_data.loc[validation_idx, self.all_data.columns != "graphs"], batch_size=batch_size
        )
        test_seq_gen = self.all_data_gen.flow(
            graphs=test_idx, targets=self.all_data.loc[test_idx, self.all_data.columns != "graphs"], batch_size=batch_size
        )

        return train_seq_gen, validation_seq_gen, test_seq_gen