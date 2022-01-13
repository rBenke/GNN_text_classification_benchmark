from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool

from src.GNN.gnn import GNN


class GCN(GNN):
    def __init__(self, nCategories: int):
        super().__init__(nCategories)
        self.model = self.create_graph_classification_model()

    def create_graph_classification_model(self):
        class GCN_model(Model):
            def __init__(self, n_hidden, n_labels):
                super().__init__()
                self.graph_conv = GCNConv(n_hidden)
                self.pool = GlobalSumPool()
                self.dropout = Dropout(0.5)
                self.dense = Dense(n_labels, 'softmax')

            def call(self, inputs):
                out = self.graph_conv(inputs)
                out = self.dropout(out)
                out = self.pool(out)
                out = self.dense(out)

                return out

        # Let's create the Keras model and prepare it for training
        model = GCN_model(256, self.nCategories)
        model.compile('adam', 'categorical_crossentropy', metrics=["acc"])

        return model

    def create_clear_model(self):
        self.model = self.create_graph_classification_model()
