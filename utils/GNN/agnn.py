from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph.mapper import PaddedGraphGenerator

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
from utils.GNN.gnn import GNN


class GCN(GNN):
    def __init__(self, generator: PaddedGraphGenerator, nCategories: int):
        super().__init__(generator, nCategories)
        self.model = self.create_graph_classification_model()

    def create_graph_classification_model(self):
        gc_model = GCNSupervisedGraphClassification(
            layer_sizes=[16, 16],
            activations=["relu", "relu"],
            generator=self.generator,
            dropout=0.4,
        )
        x_inp, x_out = gc_model.in_out_tensors()
        predictions = Dense(units=32, activation="relu")(x_out)
        predictions = Dense(units=self.nCategories, activation="softmax")(predictions)

        # Let's create the Keras model and prepare it for training
        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(optimizer=Adam(0.005), loss=categorical_crossentropy, metrics=["acc"])

        return model

    def create_clear_model(self):
        self.model = self.create_graph_classification_model()