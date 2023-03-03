import math
import jax
import numpy as np
import jax.numpy as jnp
import jax.random as random
from scipy.integrate import solve_ivp
from chex import dataclass

@dataclass
class SimplePendulum:
    m: float = 1.
    g: float = 9.8
    l: float = 2.

    @jax.jit
    def hamiltonian_fn(self, coords):
        """
        The Hamiltonian function for an ideal pendulum in generalized coordinates.
        Returns the energy as a float.
        coords => jnp.array([theta, p_theta])
        """
        q, p = jnp.split(coords, 2)
        kin_energy = p**2 / (2 * self.m * self.l**2)
        pot_energy = self.m * self.g * self.l * (1 - jnp.cos(q))
        H = kin_energy + pot_energy
        return H.reshape()

    @jax.jit
    def dynamics_fn(self, t, coords):
        """
        Function that returns the time derivatives of the position and momentum.
        Uses polar coordinates.
        """
        dcoords = jax.grad(self.hamiltonian_fn)(coords)
        dHdq, dHdp = jnp.split(dcoords, 2)
        derivatives = jnp.concatenate([dHdp, -dHdq], axis=-1)
        return derivatives

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

    @jax.jit
    def convert_to_cartesian(self, coords):
        """
        coords.shape = (2, time_steps)
        coords[0] = (theta, p_theta, dtheta, dp_theta)
        """
        x = self.l * jnp.sin(coords[0])
        y = self.l * (1 - jnp.cos(coords[0]))
        dxdt = jnp.multiply(jnp.cos(coords[0]), coords[1]) / self.l
        dydt = jnp.multiply(jnp.sin(coords[0]), coords[1]) / self.l
        return jnp.stack([x, y, dxdt, dydt])

    def get_trajectory(self, key, t_span=[0, 10], timescale=1000, noise_std=0.01, **kwargs):
        """
        Function to return a single trajectory.
        Returns a dictionary with both polar and cartesian coordinates.
        """
        t_eval = np.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))
        trajectory = np.empty((6, len(t_eval)))

        # Get initial state.
        total_energy = 9.8
        initial_coords = self.get_initial_state(total_energy)

        pendulum_ivp = solve_ivp(
            fun=self.dynamics_fn,
            t_span=t_span,
            y0=initial_coords,
            t_eval=t_eval,
            rtol=1e-10,
            **kwargs)

        coords = pendulum_ivp['y']

        # Add noise
        key, subkey = random.split(key)
        coords += random.normal(subkey, coords.shape) * noise_std

        trajectory[:2] = coords
        cartesian_coords = self.convert_to_cartesian(coords)
        trajectory[2:] = cartesian_coords

        return trajectory


def get_batched_data(key, train_data, batch_size):
    """
    Should return a vector with (batch_dim, X.shape)
    where X is a single data point.
    """
    x_data = random.permutation(key, train_data)
    num_data_points = x_data.shape[0]
    num_batches = math.floor(num_data_points / batch_size)
    batched_data = np.empty((num_batches, batch_size, x_data.shape[1]))
    for i in range(num_batches):
        low = i * batch_size
        high = (i + 1) * batch_size
        batched_data[i] = x_data[low:high]

    return batched_data, num_batches
