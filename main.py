import multiprocessing
from src.helpers import parse_args
from src.train_evaluate import TrainEvaluate
from config import *

def main():
    logging.basicConfig(level=G_LOGGING_LEVEL)
    multiprocessing.set_start_method('spawn')
    args = parse_args()
    TrainEvaluate(args.dataset_name).run()

if __name__ == "__main__":
    main()
