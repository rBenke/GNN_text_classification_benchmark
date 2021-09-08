import time

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import APPNPConv, GlobalSumPool, GlobalAttnSumPool, GlobalAttentionPool, SortPool
from spektral.utils import gcn_filter
from utils.GNN.gnn import GNN


class APPNP_model(Model):
    def __init__(self, type, n_labels):
        super().__init__()
        if type == 0:
            self.all_layers = [
                ("with_adj", APPNPConv(64, activation= "relu")),
                ("with_graphId", GlobalSumPool()),
                ("x_only", Dense(n_labels, 'softmax'))
            ]
        elif type == 1:
            self.all_layers = [
                ("with_adj", APPNPConv(128, activation= "relu", mlp_hidden = [512, 256], dropout_rate=0.4)),
                ("with_graphId", GlobalSumPool()),
                ("x_only", Dense(n_labels, 'softmax'))
            ]
        elif type == 2:
            self.all_layers = [
                ("with_adj", APPNPConv(256, activation= "relu", mlp_hidden = [768, 512], dropout_rate=0.4)),
                ("x_only", Dropout(0.4)),
                ("with_adj", APPNPConv(128, activation= "relu", mlp_hidden = [256, 128], dropout_rate=0.4)),
                ("with_graphId", GlobalSumPool()),
                ("x_only", Dropout(0.2)),
                ("x_only", Dense(128, activation= 'relu')),
                ("x_only", Dropout(0.2)),
                ("x_only", Dense(n_labels, activation= 'softmax'))
            ]
        elif type == 3:
            self.all_layers = [
                ("with_adj", APPNPConv(256, activation= "relu", mlp_hidden = [512, 256], dropout_rate=0.6)),
                ("with_graphId", GlobalAttnSumPool()),
                ("x_only", Dropout(0.4)),
                ("x_only", Dense(n_labels, 'softmax'))
            ]
        elif type == 4:
            self.all_layers = [
                ("with_adj", APPNPConv(256, activation= "relu", mlp_hidden = [512, 256], dropout_rate=0.6)),
                ("with_graphId", GlobalAttentionPool(200)),
                ("x_only", Dropout(0.4)),
                ("x_only", Dense(n_labels, 'softmax'))
            ]
        elif type == 5:
            self.all_layers = [
                ("with_adj", APPNPConv(256, activation= "relu", mlp_hidden = [512, 256], dropout_rate=0.6)),
                ("with_graphId", SortPool(5)),
                ("x_only", Dropout(0.4)),
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


class APPNP(GNN):
    def __init__(self, type: int, nCategories: int):
        super().__init__()
        self.type = type
        self.nCategories = nCategories
        self.result.model_name = "APPNP" + str(self.type)
        self.model = self._create_graph_classification_model()

    def _create_graph_classification_model(self):
        model = APPNP_model(self.type, self.nCategories)
        model.compile('adam', 'categorical_crossentropy', metrics=["acc"])
        return model

    def __str__(self):
        return " ".join(["APPNP", str(self.type)])

    def req_preprocess(self):
        """
        This function return self if the preprocess function should be
         used before training or None if the preprocess function is not needed
        """
        return self

    @staticmethod
    def preprocess(a):
        return gcn_filter(a)