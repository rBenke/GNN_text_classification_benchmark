from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GATConv, GlobalSumPool
from src.GNN.gnn import GNN


class GAT_model(Model):
    def __init__(self, type, n_labels):
        super().__init__()
        if type == 0:
            self.all_layers = [
                ("with_adj", GATConv(64, attn_heads=1, activation = "relu")),
                ("with_graphId", GlobalSumPool()),
                ("x_only", Dense(n_labels, 'softmax'))
            ]
        elif type == 1:
            self.all_layers = [
                ("with_adj", GATConv(64, attn_heads=3, dropout_rate=0.2, activation = "relu")),
                ("with_graphId", GlobalSumPool()),
                ("x_only", Dense(n_labels, 'softmax'))
            ]
        elif type == 2:
            self.all_layers = [
                ("with_adj", GATConv(256, attn_heads=2, activation =  "relu")),
                ("x_only", Dropout(0.4)),
                ("with_adj", GATConv(128, attn_heads=2, activation =  "relu")),
                ("with_graphId", GlobalSumPool()),
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


class GAT(GNN):
    def __init__(self, type: int, nCategories: int):
        super().__init__()
        self.type = type
        self.nCategories = nCategories
        self.result.model_name = "GAT" + str(self.type)
        self.model = self._create_graph_classification_model()

    def _create_graph_classification_model(self):
        model = GAT_model(self.type, self.nCategories)
        model.compile('adam', 'categorical_crossentropy', metrics=["acc"])
        return model

    def __str__(self):
        return " ".join(["GAT", str(self.type)])

    def req_preprocess(self):
        """
        This function return self if the preprocess function should be
         used before training or None if the preprocess function is not needed
        """
        return None

    @staticmethod
    def preprocess(a):
        raise Exception("GAT does not require any adjacency matrix preprocessing.")