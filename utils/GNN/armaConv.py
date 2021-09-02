from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import ARMAConv, GlobalSumPool
from spektral.utils import normalized_laplacian, rescale_laplacian
from utils.GNN.gnn import GNN

class ARMA_model(Model):
    def __init__(self, type, n_labels):
        super().__init__()
        if type == 0:
            self.all_layers = [
                ("with_adj", ARMAConv(512, activation= "relu", order=1, iterations=1)),
                ("with_graphId", GlobalSumPool()),
                ("x_only", Dense(n_labels, 'softmax'))
            ]
        elif type == 1:
            self.all_layers = [
                ("with_adj", ARMAConv(512, activation= "relu", order=3, iterations=3)),
                ("with_graphId", GlobalSumPool()),
                ("x_only", Dense(n_labels, 'softmax'))
            ]
        elif type == 2:
            self.all_layers = [
                ("with_adj", ARMAConv(256, activation= "relu", order=2, iterations=2, dropout_rate=0.2)),
                ("x_only", Dropout(0.2)),
                ("with_adj", ARMAConv(128, activation= "relu",  order=2, iterations=2, dropout_rate=0.2)),
                ("with_graphId", GlobalSumPool()),
                ("x_only", Dropout(0.2)),
                ("x_only", Dense(128, activation= 'relu')),
                ("x_only", Dropout(0.2)),
                ("x_only", Dense(n_labels, activation= 'softmax'))
            ]
        else:
            raise ValueError("Type has to be one of [0, 1, 2].")

    def call(self, inputs):
        try:
            (x_batch, adj), graph_ids = inputs, None # BatchLoader
        except ValueError:
            x_batch, adj, graph_ids = inputs # DisjointLoader

        for type, layer in self.all_layers:
            if type=="x_only":
                x_batch = layer(x_batch)
            elif type=="with_adj":
                x_batch = layer([x_batch, adj])
            elif type=="with_graphId":
                if graph_ids is not None: # DisjointLoader
                    x_batch = layer([x_batch, graph_ids])
                else: # BatchLoader
                    x_batch = layer(x_batch)
            else:
                raise ValueError("".join([str(type), " is not allowed as a layer type. Only 'x_only', 'with_adj' and 'with_graphId' can be used"]))
        return x_batch


class ARMA(GNN):
    def __init__(self, type: int, nCategories: int):
        super().__init__()
        self.type = type
        self.nCategories = nCategories
        self.result.model_name = "ARMA" + str(self.type)
        self.model = self.create_graph_classification_model()

    def create_graph_classification_model(self):
        model = ARMA_model(self.type, self.nCategories)
        model.compile('adam', 'categorical_crossentropy', metrics=["acc"])
        return model

    def __str__(self):
        return " ".join(["ARMA", str(self.type)])

    def req_preprocess(self):
        """
        This function return self if the preprocess function should be
         used before training or None if the preprocess function is not needed
        """
        return self

    @staticmethod
    def preprocess(a):
        l = normalized_laplacian(a)
        l = rescale_laplacian(l)
        return l