import jax
import jax.numpy as jnp
import jax.random as random

import numpy as np
import matplotlib.pyplot as plt

import optax
from flax.training.train_state import TrainState

from models import AutoEncoder
from data_generator import SimplePendulum, get_batched_data

# 0. Initialize the hyperparameters.
num_epochs = 1000
batch_size = 256
encoder_widths = [64, 32, 16, 1]
decoder_widths = [1, 16, 32, 64]
learning_rate = 1e-4
key = random.PRNGKey(1234)

# 1. Get the training data.
key, subkey = random.split(key)
pend = SimplePendulum(subkey)
train_dataset = pend.get_dataset(20).T

key, subkey = random.split(key)
train_data = get_batched_data(subkey, train_dataset, batch_size)

# Get validation data.
val_dataset = pend.get_dataset(10).T
key, subkey = random.split(key)
val_data = get_batched_data(subkey, val_dataset, batch_size)

# 2. Define and initialize the model.
input_shape = (2,)  # (x, y) is the input to the encoder.
model = AutoEncoder(encoder_widths,
                    decoder_widths,
                    input_shape)

init_data = jnp.ones((batch_size, *input_shape))

# Create the train state
key, subkey = random.split(key)
state = TrainState.create(
    apply_fn=model.apply,
    params=model.init(subkey, init_data)['params'],
    tx=optax.adam(learning_rate),
)


# 3. Define the loss function
def recon_loss(model, params, batched_x):
    """
    Returns the reconstruction loss for batched data.
    It is a mean squared error implementation.
    """
    def recon_loss_single(x):
        """
        The reconstruction loss for a single datapoint x.
        """
        # Separate the position and momentum coordinates.
        q, p = x[2:4], x[4:]

        # Define separate functions for the encoder and the decoder.
        a = lambda input_q: model.apply({'params': params}, input_q, method=model.encode)
        b = lambda input_z: model.apply({'params': params}, input_z, method=model.decode)

        # Reconstruct the position coordinates by passing through the autoencoder.
        z = a(q)
        q_hat = b(z)

        # Reconstruct the momentum coordinates using the equations.
        grad_a = jax.jacfwd(a)(q).squeeze()
        dzdt = jnp.dot(grad_a, p)

        jac_b = jax.jacfwd(b)(z).squeeze()
        p_hat = dzdt * jac_b
        x_hat = jnp.concatenate([q_hat, p_hat])
        loss = jnp.sum(jnp.square(x[2:] - x_hat))
        return loss

    return jnp.mean(jax.vmap(recon_loss_single)(batched_x))


# 4. Write the training loop.
@jax.jit
def train_step(state, batch):
    """
    Train for a single step/batch.
    """
    grad_fn = jax.value_and_grad(recon_loss, argnums=1)
    loss, grads = grad_fn(model, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def compute_loss(state, data):
    """
    Compute the loss of a given dataset.
    """
    loss = 0.
    num_batches = data.shape[0]
    for i in range(num_batches):
        batch = data[i]
        loss += recon_loss(model, state.params, batch)
    loss /= num_batches
    return loss


# 5. Train the model.
for epoch in range(num_epochs):
    # Print the validation loss before the model is trained.
    if epoch == 0:
        val_loss = compute_loss(state, val_data)
        print(f"Untrained validation loss = {val_loss:.3E}")

    # Loop over batches.
    epoch_loss = 0.
    num_batches = train_data.shape[0]
    for i in range(num_batches):
        batch = train_data[i]
        state, loss = train_step(state, batch)
        epoch_loss += loss
    epoch_loss /= num_batches

    # Print the training and validation loss every 50 epochs.
    if epoch % 50 == 0:
        val_loss = compute_loss(state, val_data)
        print(f"Epoch {epoch:04d} - Train loss = {epoch_loss:.3E} - Val loss = {val_loss:.3E}")


def predict_latent_batched(model, params, batched_x):
    def predict_latent_single(x):
        """
        Predict the latent space of a single data point.
        """
        # Separate the position and momentum coordinates.
        q, p = x[2:4], x[4:]

        # Define separate functions for the encoder and the decoder.
        a = lambda input_q: model.apply({'params': params}, input_q, method=model.encode)

        # Reconstruct the position coordinates by passing through the autoencoder.
        z = a(q)

        # Reconstruct the momentum coordinates using the equations.
        grad_a = jax.jacfwd(a)(q).squeeze()
        dzdt = jnp.dot(grad_a, p)

        return z, dzdt

    return jax.vmap(predict_latent_single)(batched_x)


def get_latent_variables(model, state, batched_input):
    """
    Function to get the latent variables predicted by the model.
    :param state: The train state of the trained model.
    :param batched_input: Batched input data (unshuffled).
    :return: (z, dz/dt) (2, time_steps)
    """
    z_list = []
    dzdt_list = []
    for i in range(batched_input.shape[0]):
        batch = batched_input[i]
        z, dzdt = predict_latent_batched(model, state.params, batch)
        z_list.append(z)
        dzdt_list.append(dzdt)

    z = np.concatenate(z_list, axis=0).squeeze()
    dzdt = np.concatenate(dzdt_list, axis=0)
    latent_variables = np.stack((z, dzdt), axis=0)
    return latent_variables


# # Get test_data
test_trajectory = pend.get_trajectory().T
key, subkey = random.split(key)
test_data = get_batched_data(subkey, test_trajectory, batch_size, permute=False)
latent_variables = get_latent_variables(model, state, test_data)

# Plot the results.
plt.figure()
plt.plot(latent_variables[0], latent_variables[1])
plt.xlabel(r"$z$")
plt.ylabel(r"$dz/dt$")
plt.title("Phase space diagram")
plt.show()

num_time_steps = latent_variables.shape[1]
time_steps = jnp.linspace(0., num_time_steps * 0.01, num_time_steps)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
ax1.plot(time_steps, latent_variables[0])
ax1.set_ylabel(r"$z$")
ax1.set_xlabel("Time")
ax1.set_title(r"How $z$ varies with time.")

ax2.plot(time_steps, latent_variables[1])
ax2.set_ylabel(r"$dz/dt$")
ax2.set_xlabel("Time")
ax2.set_title(r"How $dz/dt$ varies with time.")
plt.show()
