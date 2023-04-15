import flwr as fl
import logging
logging.basicConfig(level=logging.DEBUG)

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
server = fl.server.Server(client_manager=fl.server.SimpleClientManager(), strategy=fl.server.strategy.FedAvg())

# Start the Flower server
logging.debug("Starting server")
fl.server.start_server(server_address="127.0.0.1:8080", server=server, config=fl.server.ServerConfig(num_rounds=1,round_timeout=100))
logging.debug("Flower server closed.")