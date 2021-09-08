import os
import logging
import pandas as pd
import multiprocessing
from tqdm import tqdm
from typing import List
import numpy as np
from config import USE_GPU, LOGGING_LEVEL, BATCH_SIZE
from utils.GNN.gcn import GCN
from utils.GNN.chebGnn import ChebGNN
from utils.GNN.gat import GAT
from utils.GNN.appnpConv import APPNP
from utils.GNN.armaConv import ARMA
from utils.GNN.diffusionConv import DiffusionGNN
from utils.GNN.helper import train_valid_split
from spektral.data import Graph, Dataset, DisjointLoader, BatchLoader
from spektral.transforms import LayerPreprocess

class Benchmark:
    def __init__(self, all_data: pd.DataFrame, test_idx_lst: List[List[int]], results_folder: str):
        self.all_data = all_data
        self.test_idx_lst = test_idx_lst
        self.results_folder = results_folder
        self.n_labels = self.all_data.graphs[0].n_labels
        models_with_versions = [
            # (DiffusionGNN, 0), #BatchLoader
            # (DiffusionGNN, 1), #BatchLoader
            # (DiffusionGNN, 2), #BatchLoader
            (ARMA, 1),
            (ARMA, 3),
            # (ARMA, 4),
            # (ARMA, 5),
            # (ARMA, 6),
            # (APPNP, 3),
            (APPNP, 4),
            # (APPNP, 5),
            (GAT, 0),
            (GAT, 3),
            # (GAT, 4),
            # (GAT, 5),
            # (ChebGNN, 0),
            # (ChebGNN, 1),
            # (ChebGNN, 2),
            (GCN, 0),
            (GCN, 1),
            (GCN, 2),
            (GCN, 3),
            (GCN, 4),
            # (GCN, 5),
            # (GCN, 6),
            (GCN, 7),
            # (GCN, 8)
        ]
        self.models = ["".join([model.__name__, "(", str(version), ",", str(self.n_labels), ")"]) for model, version in models_with_versions]

    def benchmark(self):
        for model_str in tqdm(self.models):
            logging.info(''.join(["Model: ", str(model_str)]))
            for test_idx in self.test_idx_lst:
                if len(self.models)>1:
                    training_process = multiprocessing.Process(target=Benchmark.train_evaluate_model,
                                                               args=(model_str, self.all_data, test_idx, self.results_folder))
                    training_process.start()
                    training_process.join()
                    training_process.close()
                else: # for debugging purpose
                    Benchmark.train_evaluate_model(model_str, self.all_data, test_idx, self.results_folder)
        return None

    @staticmethod
    def train_evaluate_model(model_str, all_data, test_idx, results_folder):
        import tensorflow as tf
        if USE_GPU:
            assert len(tf.config.list_physical_devices('GPU')) > 0
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            assert len(tf.config.list_physical_devices('GPU')) == 0

        logging.basicConfig(level=LOGGING_LEVEL)
        model = eval(model_str)
        train_seq_gen, validation_seq_gen, test_seq_gen = Benchmark.create_generators(all_data, test_idx, model.req_preprocess())
        try:
            model.train(train_seq_gen, validation_seq_gen)
            model.validate(test_seq_gen)
            model.result.save_result(results_folder)
        except Exception as e:
            print(e)
        return None

    @staticmethod
    def create_generators(all_data, test_idx, preprocess_fn = None):
        train_validation_idx = set(all_data.index).difference(test_idx)
        train_idx, validation_idx = train_valid_split(list(train_validation_idx))

        data = np.array([Graph(x=G.x, a=G.a, e=G.e, y=G.y) for G in all_data.graphs])
        class Train_dataset(Dataset):
            def read(self):
                return data[train_idx]
        class Validation_dataset(Dataset):
            def read(self):
                return data[validation_idx]
        class Test_dataset(Dataset):
            def read(self):
                return data[test_idx]

        train_dataset = Train_dataset()
        valid_dataset = Validation_dataset()
        test_dataset = Test_dataset()

        logging.debug("".join(["train_dataset: ", str(len(train_dataset))]))
        logging.debug("".join(["valid_dataset: ", str(len(valid_dataset))]))
        logging.debug("".join(["test_dataset: ", str(len(test_dataset))]))

        if preprocess_fn is not None:
            train_dataset.apply(LayerPreprocess(preprocess_fn))
            valid_dataset.apply(LayerPreprocess(preprocess_fn))
            test_dataset.apply(LayerPreprocess(preprocess_fn))

        train_batch_gen = DisjointLoader(train_dataset, batch_size=BATCH_SIZE)
        validation_batch_gen = DisjointLoader(valid_dataset, batch_size=BATCH_SIZE)
        test_batch_gen = DisjointLoader(test_dataset, batch_size=BATCH_SIZE)

        return train_batch_gen, validation_batch_gen, test_batch_gen
