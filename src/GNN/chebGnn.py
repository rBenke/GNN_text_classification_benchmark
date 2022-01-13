import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import ChebConv, GlobalSumPool
from spektral.utils.convolution import chebyshev_filter
from src.GNN.gnn import GNN


class ChebGNN_model(Model):
    def __init__(self, type, n_labels):
        super().__init__()
        if type == 0:
            self.all_layers = [
                ("with_adj", ChebConv(64, K=1, activation = "relu")),
                ("x_only", GlobalSumPool()),
                ("x_only", Dense(n_labels, 'softmax'))
            ]
        elif type == 1:
            self.all_layers = [
                ("with_adj", ChebConv(512, K=2, activation = "relu")),
                ("x_only", GlobalSumPool()),
                ("x_only", Dense(n_labels, 'softmax'))
            ]
        elif type == 2:
            self.all_layers = [
                ("with_adj", ChebConv(256, K=2, activation =  "relu")),
                ("with_adj", ChebConv(128, K=2, activation =  "relu")),
                ("x_only", GlobalSumPool()),
                ("x_only", Dropout(0.4)),
                ("x_only", Dense(64, 'relu')),
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
        print(adj)
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


class ChebGNN(GNN):
    def __init__(self, type: int, nCategories: int):
        super().__init__()
        self.type = type
        self.nCategories = nCategories
        self.result.model_name = "ChebGNN"
        self.model = self._create_graph_classification_model()

    def _create_graph_classification_model(self):
        model = ChebGNN_model(self.type, self.nCategories)
        model.compile('adam', 'categorical_crossentropy', metrics=["acc"])
        return model

    def __str__(self):
        return " ".join(["ChebGNN", str(self.type)])

    def req_preprocess(self):
        """
        This function return self if the preprocess function should be
         used before training or None if the preprocess function is not needed
        """
        return self

    @staticmethod
    def preprocess(a):
        cheb_filter = chebyshev_filter(a, 1)
        # print(type(cheb_filter))
        # print(len(cheb_filter))
        # print(cheb_filter[0])
        return np.array(cheb_filter)