import numpy as np
import flwr as fl
import tensorflow as tf

# Load data
data = np.load('office_data.npz')
X = data['data']
y = data['data_norm']

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(9,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Define the loss function and optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

# Define a function to train the model on a client
def train_on_client(model, X, y, loss_fn, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        for batch_x, batch_y in zip(X, y):
            with tf.GradientTape() as tape:
                preds = model(batch_x)
                loss = loss_fn(batch_y, preds)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return model

# Define a function to evaluate the model on a client
def evaluate_on_client(model, X, y):
    preds = model(X).numpy().reshape(-1)
    return ((preds - y) ** 2).mean()

# Define the Flower client
class MyClient(fl.client.NumPyClient):
    def __init__(self, model, X, y, loss_fn, optimizer):
        self.model = model
        self.X = X
        self.y = y
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    def get_parameters(self):
        return [v.numpy() for v in self.model.trainable_variables]
        
    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model = train_on_client(self.model, self.X, self.y, self.loss_fn, self.optimizer)
        return self.get_parameters(), len(self.X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = evaluate_on_client(self.model, self.X, self.y)
        return len(self.X), {"loss": loss}

# Create a list of Flower clients
clients = []
for i in range(X.shape[0]):
    X_client = X[i]
    y_client = y[i]
    clients.append(MyClient(model, X_client, y_client, loss_fn, optimizer))

#print(type(clients))
# Connect the clients to the Flower server
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=clients)