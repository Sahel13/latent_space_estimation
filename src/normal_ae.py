import jax
import jax.random as random
import jax.numpy as jnp

import optax
from flax.training.train_state import TrainState

from models import AutoEncoder
from data_generator import SimplePendulum, get_batched_data

# 0. Initialize hyperparameters
num_epochs = 1000
batch_size = 64
encoder_widths = [64, 32, 16, 2]
decoder_widths = [2, 16, 32, 64]
learning_rate = 1e-4

# 1. Get the data
key = random.PRNGKey(5)
key, subkey = random.split(key)
pend = SimplePendulum()
trajectory = pend.get_trajectory(subkey)[2:]
train_data = trajectory.T
batched_data, num_batches = get_batched_data(subkey, train_data, batch_size)

# Define the model and the train state
input_shape = (4,)  # (x, y) is the input to the encoder.
model = AutoEncoder(encoder_widths,
                    decoder_widths,
                    input_shape)

init_data = jnp.ones((batch_size, *input_shape))

key, subkey = random.split(key)
variables = model.init(subkey, init_data)

key, subkey = random.split(key)
state = TrainState.create(
    apply_fn=model.apply,
    params=model.init(subkey, init_data)['params'],
    tx=optax.adam(learning_rate),
)


# 3. Define the loss function
def recon_loss_batched(params, batched_x):
    def recon_loss(x):
        x_hat = model.apply({'params': params}, x)
        return jnp.sum(jnp.square(x - x_hat))

    return jnp.mean(jax.vmap(recon_loss)(batched_x))


def evaluate(params, batched_data):
    eval_loss = 0.
    num_batches = batched_data.shape[0]
    for i in range(num_batches):
        batch = batched_data[i]
        eval_loss += recon_loss_batched(params, batch)

    eval_loss /= num_batches
    print(f"The evaluation loss is {eval_loss}.")
    return eval_loss


# _ = evaluate(state.params, batched_data)


# 4. Define the training loop.
@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(recon_loss_batched, argnums=0)
    loss, grads = grad_fn(state.params, batch)
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


# Evaluate the model

# test_data = jnp.load("test_data_pend.npy")
# key, subkey = random.split(key)
# test_batched_data, test_batches = get_batched_data(subkey, test_data, batch_size)
# _ = evaluate(state.params, test_batched_data)
