import sys

sys.path.insert(0, '/u/69/iqbals3/unix/Work/Code/latent_space_estimation/src')

from data_generator import SimplePendulum, get_dataset
import math
import jax
import jax.numpy as jnp


class TestSimplePendulum():
    pend = SimplePendulum()

    def test_hamiltonian_fn(self):
        coords = jnp.array([math.pi/3, 1.])
        energy = self.pend.hamiltonian_fn(coords)
        assert energy == 9.925

    def test_get_initial_state(self):
        total_energy = 9.8
        true_coords = jnp.array([math.pi/3, 0.])
        coords = self.pend.get_initial_state(total_energy)
        assert jnp.allclose(coords, true_coords)

    def test_convert_to_cartesian(self):
        time_steps = 20
        coords = jnp.ones((2, time_steps))
        conv_coords = self.pend.convert_to_cartesian(coords)
        assert conv_coords.shape == (4, time_steps)

    def test_trajectory(self):
        key = jax.random.PRNGKey(seed=0)
        trajectory = self.pend.get_trajectory(key)
        assert trajectory.shape[0] == 6
