import math
import numpy as np
import jax.numpy as jnp
import jax.random as random
from scipy.integrate import solve_ivp


class SimplePendulum:

    def __init__(self, key):
        self.m = 1.
        self.g = 9.8
        self.l = 2.
        self._key = key

    def get_key(self):
        """Convenience function to get a new random key."""
        key, subkey = random.split(self._key)
        self._key = key
        return subkey

    def dynamics_fn(self, t, coords):
        """
        :param coords: (theta, dtheta)
        :returns: The derivatives of the inputs (dtheta, ddtheta).
        """
        theta, dtheta = coords
        ddtheta = -(self.g / self.l) * jnp.sin(theta)
        return jnp.array([dtheta, ddtheta])

    def get_initial_state(self, energy):
        """
        Function to generate an initial state given total energy.
        Returns polar coordinates.
        """
        # We start from the edge (0 momentum position).
        # Have to assume there is such a position, ie, PE < 2mgl
        theta = math.acos(1 - energy / (self.m * self.g * self.l))
        dtheta = 0
        return jnp.array([theta, dtheta])

    def convert_to_cartesian(self, coords):
        """
        coords.shape = (2, time_steps)
        coords[0] = (theta, dtheta)
        """
        x = self.l * jnp.sin(coords[0])
        y = self.l * (1 - jnp.cos(coords[0]))
        dxdt = jnp.multiply(jnp.cos(coords[0]), coords[1]) * self.l
        dydt = jnp.multiply(jnp.sin(coords[0]), coords[1]) * self.l
        return jnp.stack([x, y, dxdt, dydt])

    def get_trajectory(self, t_span=(0, 10), timescale=100, noise_std=0.01, normalized=False):
        """
        Function to return a single trajectory.
        Returns a dictionary with both polar and cartesian coordinates.
        """
        t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))
        trajectory = np.empty((6, len(t_eval)))

        # Get initial state.
        total_energy = 9.8 + 2 * random.normal(self.get_key())
        initial_coords = self.get_initial_state(total_energy)

        pendulum_ivp = solve_ivp(
            fun=self.dynamics_fn,
            t_span=t_span,
            y0=initial_coords,
            t_eval=t_eval,
            rtol=1e-10)

        coords = pendulum_ivp['y']

        # Add noise
        coords += random.normal(self.get_key(), coords.shape) * noise_std

        trajectory[:2] = coords
        cartesian_coords = self.convert_to_cartesian(coords)
        trajectory[2:] = cartesian_coords

        if normalized:
            means = jnp.mean(trajectory, axis=1, keepdims=True)
            norm_trajectory = trajectory / means
            return norm_trajectory

        return trajectory

    def get_dataset(self, num_trajectories, t_span=(0, 10), timescale=100, noise_std=0.01, normalized=False):
        """
        Function to return a dataset of trajectories.
        """
        dataset = []
        for i in range(num_trajectories):
            dataset.append(self.get_trajectory(t_span, timescale, noise_std, normalized))

        return np.hstack(dataset)


def get_batched_data(key, train_data, batch_size, permute=True):
    """
    Should return a vector with (num_batches, batch_size, X.shape)
    where X is a single data point.
    """
    if permute:
        x_data = random.permutation(key, train_data)
    else:
        x_data = train_data

    num_data_points = x_data.shape[0]
    num_batches = math.floor(num_data_points / batch_size)
    batched_data = np.empty((num_batches, batch_size, x_data.shape[1]))
    for i in range(num_batches):
        low = i * batch_size
        high = (i + 1) * batch_size
        batched_data[i] = x_data[low:high]

    return batched_data


class DoublePendulum:
    m1: float = 1.
    m2: float = 1.
    g: float = 9.8
    l1: float = 1.
    l2: float = 1.

    def dynamics_fn(self, t, coords):
        """
        The dynamics of the double pendulum.
        :param coords: (theta1, theta2, dtheta1, dtheta2)
        :returns: The derivatives of the inputs (dtheta1, dtheta2, ddtheta1, ddtheta2).
        """

        theta1, theta2, dtheta1, dtheta2 = coords

        c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

        ddtheta1 = (self.m2 * self.g * np.sin(theta2) * c - self.m2 * s * (self.l1 * dtheta1 ** 2 * c + self.l2 * dtheta2 ** 2) -
                 (self.m1 + self.m2) * self.g * np.sin(theta1)) / self.l1 / (self.m1 + self.m2 * s ** 2)
        ddtheta2 = ((self.m1 + self.m2) * (self.l1 * dtheta1 ** 2 * s - self.g * np.sin(theta2) + self.g * np.sin(theta1) * c) +
                 self.m2 * self.l2 * dtheta2 ** 2 * s * c) / self.l2 / (self.m1 + self.m2 * s ** 2)

        return dtheta1, dtheta2, ddtheta1, ddtheta2

    def convert_to_cartesian(self, coords):
        """
        >>> coords.shape = (4, time_steps)
        :param coords: coords[0] = (theta1, theta2, dtheta1, dtheta2)
        :returns: (x1, y1, x2, y2, dx1dt, dy1dt, dx2dt, dy2dt)
        """
        theta1, theta2, dtheta1, dtheta2 = coords
        x1 = self.l1 * np.sin(theta1)
        y1 = -self.l1 * np.cos(theta1)
        x2 = x1 + self.l2 * np.sin(theta2)
        y2 = y1 - self.l2 * np.cos(theta2)

        dx1 = self.l1 * dtheta1 * np.cos(theta1)
        dy1 = self.l1 * dtheta1 * np.sin(theta1)
        dx2 = dx1 + self.l2 * dtheta2 * np.cos(theta2)
        dy2 = dy1 + self.l2 * dtheta2 * np.sin(theta2)

        return np.vstack([x1, y1, x2, y2, dx1, dy1, dx2, dy2])

    def get_initial_state(self, key):
        """
        Function to get the initial state of the double pendulum.
        """
        noise = random.normal(key, (4,))
        theta1 = (np.pi / 3 + noise[0]) % np.pi
        theta2 = (np.pi / 6 + noise[1]) % np.pi
        dtheta1 = noise[2] * 2
        dtheta2 = noise[3] * 2
        return np.array([theta1, theta2, dtheta1, dtheta2])

    def get_trajectory(self, key, t_span=(0, 10), timescale=100, noise_std=0., normalized=False):
        """
        Function to return a single trajectory.
        Returns a dictionary with both polar and cartesian coordinates.
        """
        t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))
        trajectory = np.empty((12, len(t_eval)))

        # Get initial state.
        key, subkey = random.split(key)
        initial_coords = self.get_initial_state(subkey)

        pendulum_ivp = solve_ivp(
            fun=self.dynamics_fn,
            t_span=t_span,
            y0=initial_coords,
            t_eval=t_eval,
            rtol=1e-10)

        coords = pendulum_ivp['y']

        # Add noise
        key, subkey = random.split(key)
        coords += random.normal(subkey, coords.shape) * noise_std

        trajectory[:4] = coords
        cartesian_coords = self.convert_to_cartesian(coords)
        trajectory[4:] = cartesian_coords

        if normalized:
            means = jnp.mean(trajectory, axis=1, keepdims=True)
            norm_trajectory = trajectory / means
            return norm_trajectory

        return trajectory
    def get_dataset(self, key, num_trajectories, t_span=(0, 10), timescale=100, noise_std=0.01, normalized=False):
        """
        Function to return a dataset of trajectories.
        """
        dataset = []
        for i in range(num_trajectories):
            key, subkey = random.split(key)
            dataset.append(self.get_trajectory(subkey, t_span, timescale, noise_std, normalized))

        return np.hstack(dataset)
