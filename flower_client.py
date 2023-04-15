import numpy as np
import flwr as fl
import tensorflow as tf
import logging
logging.basicConfig(level=logging.DEBUG)

# Load data
data = np.load('data/example_data.npz')
X = data['data']
y = data['labels']

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Define the loss function and optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

# Define a function to train the model on a client
# def train_on_client(model, X, y, loss_fn, optimizer, num_epochs=5):
#     for epoch in range(num_epochs):
#         for batch_x, batch_y in zip(X, y):
#             with tf.GradientTape() as tape:
#                 preds = model(batch_x)
#                 loss = loss_fn(batch_y, preds)
#             gradients = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(gradients, model.trainable_variables))
def train_on_client(model, X, y, loss_fn, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = tf.keras.metrics.BinaryAccuracy()
        for batch_x, batch_y in zip(X, y):
            with tf.GradientTape() as tape:
                preds = model(batch_x)
                loss = loss_fn(batch_y, preds)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()
            epoch_accuracy.update_state(batch_y, preds)
        epoch_loss /= len(X)
        epoch_acc = epoch_accuracy.result().numpy()
        print("Epoch {}, loss {:.3f}, accuracy {:.3f}".format(epoch+1, epoch_loss, epoch_acc))
    return model

# Save the trained model weights
    model.save_weights("trained_model_weights.h5")
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

    def get_parameters(self,config):
        logging.debug("Sending parameters to server")
        return [v.numpy() for v in self.model.trainable_variables]

    def set_parameters(self, parameters,config):
        logging.debug("Receiving parameters from server")
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        print("Setting parameters on client")
        self.set_parameters(parameters)
        print("Parameters set on client")
        self.model = train_on_client(self.model, self.X, self.y, self.loss_fn, self.optimizer)
        num_samples = len(self.X)
        print("Evaluating on client")
        loss = evaluate_on_client(self.model, self.X, self.y)
        print("Evaluation done")
        return self.get_parameters(config), num_samples, {"loss": loss}

    def evaluate(self, parameters, config):
        logging.debug("Sending evaluation data to server")
        self.set_parameters(parameters)
        loss = evaluate_on_client(self.model, self.X, self.y)
        num_samples = len(self.X)
        logging.debug(f"Received evaluation results from server: num_samples={num_samples}, loss={loss}")
        return num_samples, {"loss": loss}

# Create a Flower client
client = MyClient(model, X, y, loss_fn, optimizer)

# Connect the client to the Flower server
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
