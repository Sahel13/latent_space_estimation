from lse import models
import jax
import jax.numpy as jnp


def test_mlp():
    """
    Checks if the MLP model definition is working by passing in a
    random input and checking the output shape.
    """
    model = models.MLP([12, 8, 4])
    batch = jnp.ones((32, 10))
    variables = model.init(jax.random.PRNGKey(0), batch)  # Initialize the weights
    output = model.apply(variables, batch)
    assert output.shape == (32, 4)


def test_autoencoder():
    """
    Checks if the AutoEncoder model definition is working by passing in a
    random input and checking if the input and output shapes match.
    """
    model = models.AutoEncoder(encoder_widths=[20, 10, 5],
                        decoder_widths=[5, 10, 20],
                        input_shape=(12,))
    batch = jnp.ones((32, 12))
    variables = model.init(jax.random.PRNGKey(0), batch)
    encoded = model.apply(variables, batch, method=model.encode)
    decoded = model.apply(variables, encoded, method=model.decode)
    assert decoded.shape == batch.shape
