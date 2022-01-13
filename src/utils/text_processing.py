from typing import List
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import logging

from collections import Counter
from config import GB_FEATURE_SIZE, P_MIN_WORD_FREQ

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
            elif step == "most_freq":
                texts = self.most_freq(texts)
            else:
                logging.error("".join(["Selected preprocessing step: ", step, " is not implemented."]))
                raise ValueError
        return texts

    @staticmethod
    def most_freq(texts):
        all_tokens = [token for text in texts for token in text]
        tokens_counts = Counter(all_tokens)
        freq_tokens = [token for token, freq in tokens_counts.most_common(GB_FEATURE_SIZE) if freq>P_MIN_WORD_FREQ]
        logging.info(f"Vocabulary size: {str(len(freq_tokens))}")
        texts_filtered = []
        for text in tqdm(texts):
            texts_filtered.append([token for token in text if token in freq_tokens])
        return texts_filtered

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
        text = re.sub("https?://.*[\r\n]*", "", text)
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
    def lemmatize_tokens(text: List[str]) -> List[str]:
        # import nltk
        # nltk.download('wordnet')

        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in text]
        return lemmatized_tokens
