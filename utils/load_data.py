from typing import Tuple
import pandas as pd
import os


def load_files(path: str) -> pd.DataFrame:
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


def load_splitted_files(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wrapper of load_data for folders with train-test split.
    :param path: string with absolute or relative path to data folder e.g. "data/ohsumed-first-20000-docs/"
    :return: Tuple[pd.DataFrame, pd.DataFrame] First DataFrame contains train data, wheras the second one contains test data.
    The DataFrames have two string columns: content and category.
    """
    train_df = load_files(path + "training/")
    test_df = load_files(path + "test/")
    test_df = test_df.rename('test_{}'.format)
    all_data = pd.concat([train_df, test_df])
    return (all_data, [test_df.index])



if __name__ == "__main__":
    # ohsumed-first-20000-docs
    ohsumed_train, ohsumed_test = load_splitted_data("../data/textual/ohsumed-first-20000-docs/")
    print([len(text) for text in ohsumed_train.content])
    print(ohsumed_train)
    print(ohsumed_test)

    # BBC
    bbc_df = load_data("../data/textual/bbc/")
    print([len(text) for text in bbc_df.content])
    print(bbc_df)


