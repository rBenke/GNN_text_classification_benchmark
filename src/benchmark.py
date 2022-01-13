import os
import logging
import pandas as pd
import multiprocessing
from typing import List
from collections import namedtuple

import numpy as np
from tqdm import tqdm
from spektral.data import Graph, Dataset, BatchLoader, SingleLoader
from spektral.transforms import LayerPreprocess

from config import T_USE_GPU, G_LOGGING_LEVEL, T_BATCH_SIZE, T_MODELS
from src.helpers import train_valid_split

Model_description = namedtuple("Model_description", ["type", "version", "n_categories"])


class Benchmark:
    def __init__(self, all_data: pd.DataFrame, test_idx_lst: List[List[int]], results_folder: str):
        self.all_data = all_data
        self.test_idx_lst = test_idx_lst
        self.results_folder = results_folder
        self.n_labels = self.all_data.graphs[0].n_labels
        self.models = [Model_description(model_type, model_version, self.n_labels) for model_type, model_version in
                       T_MODELS]

    def benchmark(self):
        for model_description in tqdm(self.models):
            logging.info(''.join(["Model: ", str(model_description)]))
            for test_idx in self.test_idx_lst:
                if len(self.models) > 1:
                    training_process = multiprocessing.Process(target=Benchmark.train_evaluate_model,
                                                               args=(model_description, self.all_data, test_idx,
                                                                     self.results_folder))
                    training_process.start()
                    training_process.join()
                    training_process.close()
                else:  # for debugging purpose
                    Benchmark.train_evaluate_model(model_description, self.all_data, test_idx, self.results_folder)
        return None

    @staticmethod
    def train_evaluate_model(model_description, all_data, test_idx, results_folder):
        import importlib
        import tensorflow as tf
        if T_USE_GPU:
            assert len(tf.config.list_physical_devices('GPU')) > 0
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            assert len(tf.config.list_physical_devices('GPU')) == 0

        logging.basicConfig(level=G_LOGGING_LEVEL)

        module = importlib.import_module(f"src.GNN.{model_description.type.lower()}")
        model_class = getattr(module, f"{model_description.type}")
        model = model_class(model_description.version, model_description.n_categories)

        train_seq_gen, validation_seq_gen, test_seq_gen = Benchmark.create_generators(all_data, test_idx,
                                                                                      model.req_preprocess())
        try:
            model.train(train_seq_gen, validation_seq_gen)
            model.validate(test_seq_gen)
            model.result.save_result(results_folder)
        except Exception as e:
            print(e)
        return None

    @staticmethod
    def create_generators(all_data, test_idx, preprocess_fn=None):
        if all_data.shape[0]==1:
            train_validation_idx = set(all_data["index"][0]).difference([-1]+test_idx)
            train_idx, validation_idx = train_valid_split(list(train_validation_idx))
            data = np.array([Graph(x=G.x, a=G.a, e=G.e, y=G.y) for G in all_data.graphs])
            class GraphDataset(Dataset):
                def read(self):
                    return data

            graph_dataset = GraphDataset()

            if preprocess_fn is not None:
                graph_dataset.apply(LayerPreprocess(preprocess_fn))

            train_idx = set(train_idx)
            validation_idx = set(validation_idx)
            test_idx = set(test_idx)

            def mask_to_weights(mask):
                return mask.astype(np.float32) / np.count_nonzero(mask)

            mask_train=[float(idx in train_idx) for idx in all_data["index"][0]]
            weights_train = mask_to_weights(np.array(mask_train))
            mask_valid=np.array([float(idx in validation_idx) for idx in all_data["index"][0]])
            weights_valid=mask_to_weights(np.array(mask_valid))
            mask_test =np.array([float(idx in test_idx) for idx in all_data["index"][0]])
            weights_test=mask_to_weights(np.array(mask_test))

            # checks
            assert np.max((weights_train>0).astype(np.int)+(weights_valid>0).astype(np.int)+(weights_test>0).astype(np.int))==1 # they dont overlap
            assert all(weights_train+weights_valid+weights_test == [idx != -1 for idx in all_data["index"][0]]) # they cover all document nodes

            train_batch_gen = SingleLoader(graph_dataset, sample_weights=weights_train)
            validation_batch_gen = SingleLoader(graph_dataset, sample_weights=weights_valid)
            test_batch_gen = SingleLoader(graph_dataset, sample_weights=weights_test)
        else:
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

            train_batch_gen = BatchLoader(train_dataset, batch_size=T_BATCH_SIZE)
            validation_batch_gen = BatchLoader(valid_dataset, batch_size=T_BATCH_SIZE)
            test_batch_gen = BatchLoader(test_dataset, batch_size=T_BATCH_SIZE)

        return train_batch_gen, validation_batch_gen, test_batch_gen
