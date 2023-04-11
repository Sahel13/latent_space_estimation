import jax
import jax.numpy as jnp
import jax.random as random

import numpy as np
import matplotlib.pyplot as plt

import optax
from flax.training.train_state import TrainState

from models import AutoEncoder
from data_generator import DoublePendulum, get_batched_data

# 0. Initialize the hyperparameters.
num_epochs = 500
batch_size = 256
encoder_widths = [128, 64, 32, 4]
decoder_widths = [4, 32, 64, 128]
learning_rate = 1e-4

# 1. Get the data.
key = random.PRNGKey(12345)
pend = DoublePendulum()
key, subkey = random.split(key)
train_data = pend.get_dataset(subkey, 20).T
key, subkey = random.split(key)
batched_data = get_batched_data(subkey, train_data, batch_size)

# 2. Define and initialize the model.
input_shape = (4,)  # (x1, y1, x2, y2) is the input to the encoder.
model = AutoEncoder(encoder_widths,
                    decoder_widths,
                    input_shape)

init_data = jnp.ones((batch_size, *input_shape))
key, subkey = random.split(key)

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
        q, p = x[4:8], x[8:12]

        # Define separate functions for the encoder and the decoder.
        a = lambda input_q: model.apply({'params': params}, input_q, method=model.encode)
        b = lambda input_z: model.apply({'params': params}, input_z, method=model.decode)

        # Reconstruct the position coordinates by passing through the autoencoder.
        z = a(q)
        q_hat = b(z)

        # Reconstruct the momentum coordinates using the equations.
        jac_a = jax.jacfwd(a)(q)
        dzdt = jnp.dot(jac_a, p)

        jac_b = jax.jacfwd(b)(z)
        p_hat = jnp.dot(jac_b, dzdt)
        x_hat = jnp.concatenate([q_hat, p_hat])
        loss = jnp.sum(jnp.square(x[4:] - x_hat))
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


# 5. Train the model.
for epoch in range(num_epochs):
    epoch_loss = 0.
    num_batches = batched_data.shape[0]
    for i in range(num_batches):
        batch = batched_data[i]
        state, loss = train_step(state, batch)
        epoch_loss += loss
    epoch_loss /= num_batches

    if epoch % 50 == 0:
        print(f"Epoch {epoch} - Loss = {epoch_loss}")


# def predict_latent_batched(model, params, batched_x):
#     def predict_latent_single(x):
#         """
#         Predict the latent space of a single data point.
#         """
#         # Separate the position and momentum coordinates.
#         q, p = x[4:8], x[8:]
#
#         # Define separate functions for the encoder and the decoder.
#         a = lambda input_q: model.apply({'params': params}, input_q, method=model.encode)
#
#         # Reconstruct the position coordinates by passing through the autoencoder.
#         z = a(q)
#
#         # Reconstruct the momentum coordinates using the equations.
#         jac_a = jax.jacfwd(a)(q)
#         dzdt = jnp.dot(jac_a, p)
#
#         return z, dzdt
#
#     return jax.vmap(predict_latent_single)(batched_x)
#
#
# def get_latent_variables(model, state, batched_input):
#     """
#     Function to get the latent variables predicted by the model.
#     :param state: The train state of the trained model.
#     :param batched_input: Batched input data (unshuffled).
#     :return: (z, dz/dt) (2, time_steps)
#     """
#     z_list = []
#     dzdt_list = []
#     for i in range(num_batches):
#         batch = batched_input[i]
#         z, dzdt = predict_latent_batched(model, state.params, batch)
#         z_list.append(z)
#         dzdt_list.append(dzdt)
#
#     z = np.concatenate(z_list, axis=0).squeeze()
#     dzdt = np.concatenate(dzdt_list, axis=0)
#     print(z.shape, dzdt.shape)
#     latent_variables = np.stack((z, dzdt), axis=0)
#     return latent_variables
#
#
# key, subkey = random.split(key)
# test_data = pend.get_trajectory(subkey).T
# test_data_batched, num_batches = get_batched_data(key, test_data, batch_size, permute=False)
# latent_variables = get_latent_variables(model, state, test_data_batched)
#
# # Plot the results.
# plt.figure()
# plt.plot(latent_variables[0], latent_variables[1])
# plt.xlabel(r"$z$")
# plt.ylabel(r"$dz/dt$")
# plt.title("Phase space diagram")
# plt.show()
#
# num_time_steps = latent_variables.shape[1]
# time_steps = jnp.linspace(0., 10., test_data.shape[0])
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
# ax1.plot(time_steps[:num_time_steps], latent_variables[0])
# ax1.set_ylabel(r"$z$")
# ax1.set_xlabel("Time")
# ax1.set_title(r"How $z$ varies with time.")
#
# ax2.plot(time_steps[:num_time_steps], latent_variables[1])
# ax2.set_ylabel(r"$dz/dt$")
# ax2.set_xlabel("Time")
# ax2.set_title(r"How $dz/dt$ varies with time.")
# plt.show()
#