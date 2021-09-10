import logging

EXP_VERSION = "W2V_without_connections"
W2V_VERSION = 'word2vec-google-news-300'
FEATURE_SIZE = None # node feature size (BoW, tf-idf)
USE_GPU = True
TF_FIT_VERBOSE = 0
LOGGING_LEVEL = logging.INFO
CV_FOLDS = 5
BATCH_SIZE = 64
VALIDATION_SIZE=0.2
EARLY_STOPPING = 20
