from typing import List
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import logging
def is_NOT_tokenized(text) -> bool:
    return not is_tokenized(text)

def is_tokenized(text) -> bool:
    # it should be a list of strings
    is_list = type(text) is list
    is_list_of_strings = all([type(elem) is str for elem in text])
    if is_list and is_list_of_strings:
        return True
    else:
        return False

class Preprocess:
    def __init__(self):
        pass

    def fit_transform(self, texts: str, steps: List[str]) -> str:
        for step in tqdm(steps):
            if step == "tokenize":
                texts = [self.tokenize(text) for text in texts]
            elif step == "clean":
                texts = [self.clean_text(text) for text in texts]
            elif step == "lower":
                texts = [text.lower() for text in texts]
            elif step == "stopwords":
                texts = [self.remove_stopwords_tokens(text) for text in texts]
            elif step == "lemmatize":
                texts = [self.lemmatize_tokens(text) for text in texts]
        return texts


    def transform(self, texts: str, steps: List[str]) -> str:
        for step in steps:
            if step == "tokenize":
                texts = [self.tokenize(text) for text in tqdm(texts)]
            elif step == "clean":
                texts = [self.clean_text(text) for text in tqdm(texts)]
            elif step == "lower":
                texts = [text.lower() for text in tqdm(texts)]
            elif step == "stopwords":
                texts = [self.remove_stopwords_tokens(text) for text in tqdm(texts)]
            elif step == "lemmatize":
                texts = [self.lemmatize_tokens(text) for text in tqdm(texts)]
        return texts

    # single document processing
    @staticmethod
    def clean_text(text: str) -> str:
        assert is_NOT_tokenized(text)
        raise NotImplementedError

    @staticmethod
    def tokenize(text: str) -> str:
        assert is_NOT_tokenized(text)
        text = word_tokenize(text)
        return text

    @staticmethod
    def remove_stopwords_tokens(text: List[str]) -> List[str]:
        assert is_tokenized(text)
        raise NotImplementedError

    @staticmethod
    def remove_numeric_tokens(text: List[str]) -> List[str]:
        assert is_tokenized(text)
        raise NotImplementedError

    @staticmethod
    def lemmatize_tokens(text: List[str]) -> List[str]:
        assert is_tokenized(text)
        raise NotImplementedError
