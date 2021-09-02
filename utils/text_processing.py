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

    def transform(self, texts: str, steps: List[str]) -> str:
        for step in steps:
            if step == "tokenize":
                texts = [self.tokenize(text) for text in tqdm(texts)]
            elif step == "alpha_words_only":
                texts = [self.alpha_only_tokens(text) for text in tqdm(texts)]
            elif step == "lower":
                texts = [text.lower() for text in tqdm(texts)]
            elif step == "stopwordsNltk":
                texts = [self.remove_stopwordsNltk_tokens(text) for text in tqdm(texts)]
            elif step == "clean":
                texts = [self.clean(text) for text in tqdm(texts)]
            elif step == "lemmatize":
                texts = [self.lemmatize_tokens(text) for text in tqdm(texts)]
            elif step == "remove_unicode":
                texts = [self.remove_unicode(text) for text in tqdm(texts)]
            else:
                raise ValueError
        return texts


    # single document processing
    @staticmethod
    def alpha_only_tokens(text: List[str]) -> List[str]:
        assert is_tokenized(text)
        text = [word for word in text if word.isalpha()]
        return text

    @staticmethod
    def remove_unicode(text: str) -> str:
        assert is_NOT_tokenized(text)
        text = text.encode(encoding="ascii", errors="ignore")
        text = text.decode()
        return text

    @staticmethod
    def clean(text: str) -> str:
        assert is_NOT_tokenized(text)
        import re
        import string
        # removing mentions
        text = re.sub("@\S+", "", text)
        # remove urls
        text = re.sub("https?:\/\/.*[\r\n]*", "", text)
        # remove punctuation
        punct = set(string.punctuation)
        text = "".join([sign for sign in text if sign not in punct])
        return text

    @staticmethod
    def tokenize(text: str) -> str:
        assert is_NOT_tokenized(text)
        text = word_tokenize(text)
        return text

    @staticmethod
    def remove_stopwordsNltk_tokens(text: List[str]) -> List[str]:
        assert is_tokenized(text)
        from nltk.corpus import stopwords
        # import nltk
        # nltk.download("stopwords")

        stop_words = set(stopwords.words("english"))
        text = [token for token in text if token not in stop_words]
        text = [token for token in text if len(token) > 1]
        return text

    @staticmethod
    def remove_numeric_tokens(text: List[str]) -> List[str]:
        assert is_tokenized(text)
        raise NotImplementedError

    @staticmethod
    def lemmatize_tokens(text: List[str]) -> List[str]:
        assert is_tokenized(text)
        raise NotImplementedError
