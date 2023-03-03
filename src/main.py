import jax
import jax.numpy as jnp
import jax.random as random

import optax
from flax.training.train_state import TrainState

from models import AutoEncoder
from data_generator import SimplePendulum, get_batched_data

# 0. Initialize hyperparameters
num_epochs = 1000
batch_size = 1000
encoder_widths = [64, 32, 16, 1]
decoder_widths = [1, 16, 32, 64]
learning_rate = 1e-3

# 1. Get the data
# train_data = jnp.load("train_data_pend.npy")
# key = random.PRNGKey(5)
# key, subkey = random.split(key)
# batched_data, num_batches = get_batched_data(subkey, train_data, batch_size)

key = random.PRNGKey(5)
key, subkey = random.split(key)
pend = SimplePendulum()
trajectory = pend.get_trajectory(subkey)
train_data = trajectory.T
batched_data, num_batches = get_batched_data(subkey, train_data, batch_size)

# 2. Define and initialize the model
input_shape = (2,)  # (x, y) is the input to the encoder.
model = AutoEncoder(encoder_widths,
                    decoder_widths,
                    input_shape)

init_data = jnp.ones((batch_size, *input_shape))
key, subkey = random.split(key)
variables = model.init(random.PRNGKey(0), init_data)

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


# 5. Train the model.
for epoch in range(num_epochs):
    epoch_loss = 0.
    for i in range(num_batches):
        batch = batched_data[i]
        state, loss = train_step(state, batch)
        epoch_loss += loss

    epoch_loss /= num_batches

    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Loss = {epoch_loss}")

# 6. Decide how to evaluate the model.
