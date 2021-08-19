from stellargraph.mapper import PaddedGraphGenerator

from tensorflow.keras.callbacks import EarlyStopping

from utils.GNN.result import Result


class GNN:
    def __init__(self, generator: PaddedGraphGenerator, nCategories: int):
        self.result = Result()
        self.nCategories = nCategories
        self.generator = generator
        self.model = self.create_graph_classification_model()

    def train(self, train_gen, validation_gen):
        epochs = 99999  # maximum number of training epochs

        es = EarlyStopping(
            monitor="val_loss", min_delta=0, patience=5, restore_best_weights=True
        )
        self.model.fit(
            train_gen, validation_data = validation_gen, epochs=epochs, verbose=0, callbacks=[es]
        )

    def validate(self, test_gen):
        # calculate performance on the test data and return along with history
        test_metrics = self.model.evaluate(test_gen, verbose=0)
        test_acc = test_metrics[self.model.metrics_names.index("acc")]
        print("test_acc")
        print(test_acc)
        print(test_metrics)
        return None
