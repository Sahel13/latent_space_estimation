from typing import Sequence

import numpy as np
import flax.linen as nn


class MLP(nn.Module):
    """
    Simple multi-layer perceptron model in Flax.
    From https://github.com/google/flax.
    """
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.tanh(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


class AutoEncoder(nn.Module):
    """
    Normal autoencoder in Flax.
    From https://github.com/google/flax.
    """
    encoder_widths: Sequence[int]
    decoder_widths: Sequence[int]
    input_shape: Sequence[int]

    def setup(self):
        input_dim = np.prod(self.input_shape)
        self.encoder = MLP(self.encoder_widths)
        self.decoder = MLP(self.decoder_widths + (input_dim,))

    def __call__(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        z = self.decoder(z)
        x = nn.sigmoid(z)
        return x
