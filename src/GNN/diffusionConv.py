from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import DiffusionConv, GlobalSumPool
from src.GNN.gnn import GNN


class Diffusion_model(Model):
    def __init__(self, type, n_labels):
        super().__init__()
        if type == 0:
            self.all_layers = [
                ("with_adj", DiffusionConv(64, K=3, activation="tanh")),
                ("with_graphId", GlobalSumPool()),
                ("x_only", Dense(n_labels, 'softmax'))
            ]
        elif type == 1:
            self.all_layers = [
                ("with_adj", DiffusionConv(64, K=36, activation="tanh")),
                ("with_graphId", GlobalSumPool()),
                ("x_only", Dense(n_labels, 'softmax'))
            ]
        elif type == 2:
            self.all_layers = [
                ("with_adj", DiffusionConv(256, K=2, activation="tanh")),
                ("with_adj", DiffusionConv(128, K=1, activation="tanh")),
                ("with_graphId", GlobalSumPool()),
                ("x_only", Dropout(0.4)),
                ("x_only", Dense(64, 'tanh')),
                ("x_only", Dropout(0.2)),
                ("x_only", Dense(n_labels, 'softmax'))
            ]
        else:
            raise ValueError("Type has to be one of [0, 1, 2].")

    def call(self, inputs):
        try:
            (x_batch, adj), graph_ids = inputs, None  # BatchLoader
        except ValueError:
            x_batch, adj, graph_ids = inputs  # DisjointLoader

        for type, layer in self.all_layers:
            if type == "x_only":
                x_batch = layer(x_batch)
            elif type == "with_adj":
                x_batch = layer([x_batch, adj])
            elif type == "with_graphId":
                if graph_ids is not None:  # DisjointLoader
                    x_batch = layer([x_batch, graph_ids])
                else:  # BatchLoader
                    x_batch = layer(x_batch)
            else:
                raise ValueError("".join([str(type),
                                          " is not allowed as a layer type. Only 'x_only', 'with_adj' and 'with_graphId' can be used"]))
        return x_batch


class DiffusionGNN(GNN):
    def __init__(self, type: int, nCategories: int):
        super().__init__()
        self.type = type
        self.nCategories = nCategories
        self.result.model_name = "DiffusionGNN"
        self.model = self._create_graph_classification_model()

    def _create_graph_classification_model(self):
        model = Diffusion_model(self.type, self.nCategories)
        model.compile('adam', 'categorical_crossentropy', metrics=["acc"])
        return model

    def __str__(self):
        return " ".join(["DiffusionGNN", str(self.type)])

    def req_preprocess(self):
        """
        This function return self if the preprocess function should be
         used before training or None if the preprocess function is not needed
        """
        return self

    @staticmethod
    def preprocess(a):
        return DiffusionConv.preprocess(a)
