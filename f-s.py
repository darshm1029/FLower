import flwr as fl
import tensorflow as tf

# Define the FedAvg aggregation function
def aggregate(weights_list):
    # Compute weighted average of model weights
    weights_sum = [w * n for w, n in weights_list[0]]
    for weights, num in weights_list[1:]:
        for i in range(len(weights)):
            weights_sum[i] += weights[i] * num
    weights_avg = [w / sum(n for _, n in weights_list) for w in weights_sum]
    return weights_avg

# Define the Flower server
class SaveModelServer(fl.server.Server):
    def __init__(self, client_manager: fl.server.client_manager.ClientManager, strategy: fl.server.strategy.Strategy):
        super().__init__(client_manager, strategy)
        self.global_model = None

    def evaluate(self, parameters: fl.common.Weights) -> fl.common.EvaluateRes:
        # Evaluate the global model
        res = super().evaluate(parameters)
        if res.num_examples > 0 and res.loss is not None:
            print(f"Eval - Loss: {res.loss}, Accuracy: {res.accuracy}")
        return res

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        # Train the model
        res = super().fit(ins)
        if self.global_model is None:
            self.global_model = res.parameters
        else:
            self.global_model = aggregate([(self.global_model, 1), (res.parameters, 1)])
        return res

    def on_stop(self):
        # Save the global model at the end of training
        if self.global_model is not None:
            tf.keras.models.save_model(self.global_model, "global_model")

server = SaveModelServer(client_manager=fl.server.SimpleClientManager(), strategy=fl.server.strategy.FedAvg())

# Start the Flower server
fl.server.start_server(server_address="0.0.0.0:8080", server=server, config=fl.server.ServerConfig(num_rounds=3))
