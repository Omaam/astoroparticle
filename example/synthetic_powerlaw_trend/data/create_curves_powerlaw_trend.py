"""Create light cruve where parameters change with trend.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import partical_xspec as px


def simulate_powerlaw_parameters(num_timesteps):
    """Siulate powerlaw parameters.
    """
    xspec_param_size = 2
    xspec_param_offsets = np.array([1.0, 10])
    xspec_param_slopes = np.array([0.0, -0.05])
    xspec_param_noise_scales = np.array([0.1, 0.1])

    np.random.seed(0)
    times = np.arange(num_timesteps)
    xspec_param_noise_rv = xspec_param_noise_scales * np.random.randn(
        num_timesteps, xspec_param_size)

    xspec_param_ts = \
        xspec_param_offsets + \
        times[:, np.newaxis]*xspec_param_slopes + \
        xspec_param_noise_rv
    xspec_param_ts = np.expand_dims(xspec_param_ts, axis=-2)

    return times, xspec_param_ts


def plot_xspec_param_observations(
        times, xspec_param_ts, observations_ts):
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(times, xspec_param_ts[:, 0, 0])
    ax[1].plot(times, xspec_param_ts[:, 0, 1])
    ax[2].plot(times, observations_ts[:, 0, :])

    ax[1].set_yscale("log")

    plt.tight_layout()
    plt.show()


def main():

    dtype = tf.float32
    num_timesteps = 100
    times, xspec_param_ts = simulate_powerlaw_parameters(num_timesteps)

    px.xspec.set_energy(0.5, 10.0, 10)
    obseravtion_fn = px.get_observaton_function_xspec_poisson(
        "powerlaw", 2, 1)

    observations_ts = []
    for i in range(num_timesteps):
        x = tf.convert_to_tensor(xspec_param_ts[i], dtype=dtype)
        observation_dist = obseravtion_fn(None, x)
        observations_ts.append(observation_dist.sample(seed=i))
    observations_ts = np.array(observations_ts)

    np.savetxt("latents.txt", xspec_param_ts[..., 0, :])
    np.savetxt("observations.txt", observations_ts[..., 0, :])

    plot_xspec_param_observations(
        times, xspec_param_ts, observations_ts)


if __name__ == "__main__":
    main()
