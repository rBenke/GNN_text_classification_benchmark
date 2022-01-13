import logging

# GENERAL CONFIG
G_EXP_VERSION = "TEST"
G_LOGGING_LEVEL = logging.INFO

# PREPROCESSING CONFIG
P_PREPROCESSING_STEPS = ("lower", "tokenize",  "stopwordsNltk", "most_freq")
P_MIN_WORD_FREQ = 0

# GRAPH BUILDER CONFIG
GB_GRAPH_TYPE = "GraphOfWords" # GraphOfWords, BagOfWordsGraph, TextGCNGraph
GB_W2V_VERSION = "glove-wiki-gigaword-300"
GB_FEATURE_SIZE = 30_000  # node feature size (BoW, tf-idf)

# TRAIN EVALUATE CONFIG
T_USE_GPU = True
T_TF_FIT_VERBOSE = 2
T_CV_FOLDS = 10
T_BATCH_SIZE = 64
T_VALIDATION_SIZE = 0.1
T_EARLY_STOPPING = 10
T_MODELS = [
            ("GCN", 2),
            ("GCN", 3),
            # ("GCN", "textGCN"),
            # (DiffusionGNN, 0), #BatchLoader
            # (ARMA, 1),
            # (APPNP, 1),
            # (APPNP, 2),
            # (GAT, 2),
            # (ChebGNN, 0),
]
