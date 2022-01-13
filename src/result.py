import numpy as np
import pandas as pd
import datetime
import pickle
class Result:
    def __init__(self):
        self.model_name = "" 
        self.test_acc = 0
        self.train_acc = 0
        self.training_time = np.Inf
        self.evaluation_time = np.Inf
        self.__GPU = None
        self.history = 0

    def __str__(self):
        return " ".join([
            self.model_name,
            "| train:", str(round(self.train_acc, 3)),
            "| test:", str(round(self.test_acc, 3)),
            "|", str(self.__GPU),
            "| training time:", str(round(self.training_time, 2)),
            "| evaluation time:", str(round(self.evaluation_time, 2))
        ])
    
    def get_result_df(self):
        return pd.DataFrame({
            "train_acc": self.train_acc,
            "test_acc": self.test_acc,
            "training_time": self.training_time,
            "evaluation_time": self.evaluation_time,
            "device": self.__GPU
        }, index = [self.model_name])
    
    def save_result(self, folder):
        model_name = self.model_name.replace(" ", "_")
        time_now = datetime.datetime.now().strftime("%H%M%S_%f")
        filename = "".join([folder, model_name, "_", time_now, ".pickle"])
        with open(filename, "wb") as file:
            pickle.dump([self.get_result_df(), str(self), self.history], file)
        return 0

    def set_gpu(self, GPU: int):
        if GPU:
            self.__GPU = "GPU"
        else:
            self.__GPU = "CPU"