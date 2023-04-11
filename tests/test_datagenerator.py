import sys

sys.path.insert(0, '/home/sahel/Code/lse/src')

from data_generator import SimplePendulum, DoublePendulum
import math
import jax
import jax.numpy as jnp


class TestSimplePendulum():
    key = jax.random.PRNGKey(seed=0)
    pend = SimplePendulum(key)

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
        trajectory = self.pend.get_trajectory()
        assert trajectory.shape[0] == 6

    def test_get_dataset(self):
        key = jax.random.PRNGKey(seed=0)
        dataset = self.pend.get_dataset(5)
        assert dataset.shape[0] == 6


class TestDoublePendulum():
    pend = DoublePendulum()

    def test_convert_to_cartesian(self):
        time_steps = 20
        coords = jnp.ones((4, time_steps))
        conv_coords = self.pend.convert_to_cartesian(coords)
        assert conv_coords.shape == (8, time_steps)
