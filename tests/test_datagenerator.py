from lse import data_generator
import math
import numpy as np


class TestSimplePendulum():
    rng = np.random.default_rng(12)
    pend = data_generator.SimplePendulum(rng)

    def test_get_initial_state(self):
        total_energy = 9.8
        true_coords = np.array([math.pi/3, 0.])
        coords = self.pend.get_initial_state(total_energy)
        assert np.allclose(coords, true_coords)

    def test_convert_to_cartesian(self):
        time_steps = 20
        coords = np.ones((2, time_steps))
        conv_coords = self.pend.convert_to_cartesian(coords)
        assert conv_coords.shape == (4, time_steps)

    def test_trajectory(self):
        trajectory = self.pend.get_trajectory()
        assert trajectory.shape[0] == 6

    def test_get_dataset(self):
        dataset = self.pend.get_dataset(5)
        assert dataset.shape[0] == 6


# class TestDoublePendulum():
#     pend = DoublePendulum()
# 
#     def test_convert_to_cartesian(self):
#         time_steps = 20
#         coords = jnp.ones((4, time_steps))
#         conv_coords = self.pend.convert_to_cartesian(coords)
#         assert conv_coords.shape == (8, time_steps)
