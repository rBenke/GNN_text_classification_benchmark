import logging
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from config import T_EARLY_STOPPING, T_TF_FIT_VERBOSE
from src.result import Result


class GNN:
    def __init__(self):
        self.result = Result()
        self.model = None

    def train(self, train_gen, validation_gen):
        epochs = 10_000  # maximum number of training epochs

        es = EarlyStopping(
            monitor="val_loss", min_delta=0, patience=T_EARLY_STOPPING, restore_best_weights=True
        )
        begin_time = datetime.datetime.now()
        history = self.model.fit(
            train_gen.load(), steps_per_epoch = train_gen.steps_per_epoch,
            validation_data=validation_gen.load(), validation_steps = validation_gen.steps_per_epoch,
            epochs=epochs, verbose=T_TF_FIT_VERBOSE, callbacks=[es]
        )
        end_time = datetime.datetime.now()
        self.result.set_gpu(len(tf.config.list_physical_devices('GPU')))
        self.result.history = history.history
        self.result.train_acc = history.history["acc"][-1]
        self.result.training_time = (end_time - begin_time).total_seconds()
        return self.result.train_acc

    def validate(self, test_gen):
        # calculate performance on the test data and return along with history
        begin_time = datetime.datetime.now()
        test_metrics = self.model.evaluate(test_gen.load(), steps=test_gen.steps_per_epoch, verbose=T_TF_FIT_VERBOSE)
        end_time = datetime.datetime.now()
        test_acc = test_metrics[self.model.metrics_names.index("acc")]
        self.result.test_acc = test_acc
        self.result.evaluation_time = (end_time - begin_time).total_seconds()
        logging.info(str(self.result))
        return test_acc
