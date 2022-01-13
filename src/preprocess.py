import os
import logging
from typing import Tuple, List
from hashlib import md5
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

from src.utils.text_processing import Preprocess
from config import P_PREPROCESSING_STEPS, GB_FEATURE_SIZE


class DatasetPreprocessing:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        pass

    def is_computed(self):
        preprocessing_steps_hash = self.get_preprocessing_version()
        output_dir = "".join(["data/tokenized/", self.dataset_name, "/", preprocessing_steps_hash])
        metadata_file_exists = os.path.isfile(output_dir + "/usage_log.txt")
        if not metadata_file_exists:
            return False

        with open(output_dir + "/usage_log.txt", "r") as file:
            metadata_file = file.readlines()
        is_the_same_processing = metadata_file[1][:-1] == str((P_PREPROCESSING_STEPS, GB_FEATURE_SIZE))
        return is_the_same_processing

    def process(self):

        logging.info('Loading raw ' + self.dataset_name + ' dataset.')
        all_data, test_indexes = self.load_raw_data(self.dataset_name)

        logging.info('Text preprocessing.')
        preprocess = Preprocess()
        preprocess_steps = P_PREPROCESSING_STEPS
        all_data["content"] = preprocess.transform(all_data.content, preprocess_steps)

        logging.info('Transforming labels into onehot encodings.')
        onehot_encoding = OneHotEncoder()
        categories_arr = all_data.category.to_numpy()[:, np.newaxis]
        onehot_categories = onehot_encoding.fit_transform(categories_arr).toarray()
        all_data = pd.concat([all_data.content.reset_index(drop=False), pd.DataFrame(onehot_categories)], axis=1)

        logging.info('Saving preprocessed and tokenized data.')
        preprocessing_steps_hash = self.get_preprocessing_version()
        output_dir = "".join(["data/tokenized/", self.dataset_name, "/", preprocessing_steps_hash])
        os.makedirs(output_dir, exist_ok=True)
        # Save data
        all_data.columns = [str(col_name) for col_name in all_data.columns]
        all_data = all_data.astype({'index': 'str'})
        all_data.to_parquet(output_dir + "/data.parquet")
        # Save labels
        pickle.dump(test_indexes, open(output_dir + "/labels.pickle", "wb"))
        # Save metadata
        with open(output_dir + "/usage_log.txt", "w") as file:
            file.writelines("PREPROCESSING STEPS: \n")
            file.write(str((P_PREPROCESSING_STEPS, GB_FEATURE_SIZE)))
            file.write("\nUSAGE:\n")

        logging.info('Preprocessing finished.')
        return all_data, test_indexes

    def load_files(self, path: str) -> pd.DataFrame:
        """
        Loading files having the following structure: path/categoryName/document
        :param path: string with absolute or relative path to data folder e.g. "data/bbc/"
        :return: pd.DataFrame with two string columns: content and category
        """
        categories = os.listdir(path)
        document_content_lst = []
        document_category_lst = []

        for category in categories:
            documents = os.listdir(path + category)
            for document in documents:
                with open(path + category + "/" + document, encoding="ISO-8859-1") as file:
                    document_content = file.readlines()
                    document_content = " ".join(document_content)
                    if len(document_content):
                        document_content_lst.append(document_content)
                        document_category_lst.append((category))

        return pd.DataFrame({"content": document_content_lst, "category": document_category_lst})

    def load_splitted_files(self, path: str) -> Tuple[pd.DataFrame, List[List[str]]]:
        """
        Wrapper of load_data for folders with train-test split.
        :param path: string with absolute or relative path to data folder e.g. "data/ohsumed-first-20000-docs/"
        :return: Tuple[pd.DataFrame, pd.DataFrame] First DataFrame contains train data, wheras the second one contains test data.
        The DataFrames have two string columns: content and category.
        """
        train_df = self.load_files(path + "training/")
        test_df = self.load_files(path + "test/")
        test_df = test_df.rename('test_{}'.format)
        all_data = pd.concat([train_df, test_df])
        return (all_data, [test_df.index.tolist()])

    def load_raw_data(self, dataset_name):
        if dataset_name == "bbc":
            logging.info('Loading BBC dataset')
            all_data = self.load_files("data/textual/bbc/")
            return all_data, None
        elif dataset_name == "bbcsport":
            logging.info('Loading BBC_sport dataset')
            all_data = self.load_files("data/textual/bbcsport/")
            return all_data, None
        elif dataset_name == "20newsgroups":
            logging.info('Loading 20newsgroups dataset')
            all_data = self.load_files("data/textual/20_newsgroups/")
            return all_data, None
        elif dataset_name == "reuters":
            logging.info('Loading Reuters dataset')
            all_data, test_idx = self.load_splitted_files("data/textual/Reuters21578-Apte-90Cat/")
            return all_data, test_idx
        elif dataset_name == "ohsumed":
            logging.info('Loading Ohsumed dataset')
            all_data, test_idx = self.load_splitted_files("data/textual/ohsumed-first-20000-docs/")
            return all_data, test_idx
        elif dataset_name == "ohsumed_small":
            logging.info('Loading Ohsumed dataset')
            all_data, test_idx = self.load_splitted_files("data/textual/ohsumed_small/")
            return all_data, test_idx
        else:
            raise AttributeError("Selected dataset is not available.")

    @staticmethod
    def get_preprocessing_version():
        preprocessing_definition = (P_PREPROCESSING_STEPS, GB_FEATURE_SIZE)
        hash_function = md5()
        hash_function.update(str(preprocessing_definition).encode("ascii"))
        preprocessing_hash = hash_function.hexdigest()
        preprocessing_version_str = np.base_repr(int(preprocessing_hash,16), base=36)
        return preprocessing_version_str


    def load_preprocessed_data(self):
        logging.info('Loading processed ' + self.dataset_name + ' dataset, version: ' + self.get_preprocessing_version())
        if self.is_computed():
            output_dir = "".join(
                ["data/tokenized/", self.dataset_name, "/", self.get_preprocessing_version()])
            preprocessed_data = pd.read_parquet(output_dir + "/data.parquet")
            test_indexes = pickle.load(open(output_dir + "/labels.pickle", "rb"))
            return preprocessed_data, test_indexes
        else:
            return self.process()

