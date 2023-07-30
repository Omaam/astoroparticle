"""Trial of the particle filter in TensorFlow Probability.
"""
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

import astropf as px

sns.set_style("whitegrid")


def plot_and_save_particle_distribution_with_latents(
        particles,
        latents_true=None,
        times=None,
        ylabels=None,
        savename=None,
        particle_quantiles=None,
        indicies_logy=None):

    if latents_true.shape[-2] != particles.shape[-3]:
        raise ValueError(
            "`latents_true.shape[-2]` must be `particles.shape[-3]`. "
            f"{latents_true.shape[-2]} != {particles.shape[-3]}.")

    if times is None:
        times = np.arange(particles.shape[-3])

    fig, ax = plt.subplots(2, sharex=True)

    if latents_true is not None:
        ax[0].plot(times, latents_true[:, 0], color="k")
        ax[1].plot(times, latents_true[:, 1], color="k")

    y_centers = np.quantile(particles, 0.5, axis=-2)
    ax[0].plot(times, y_centers[:, 0], color="r")
    ax[1].plot(times, y_centers[:, 1], color="r")

    if particle_quantiles is not None:
        for quantile in particle_quantiles:
            errors_sigma = np.quantile(particles, quantile, axis=-2)
            for i in range(2):
                ax[i].fill_between(
                    times, *errors_sigma[..., i], alpha=0.20,
                    facecolor="none", color="r", edgecolor="none")

    if indicies_logy is not None:
        for i in indicies_logy:
            ax[i].set_yscale("log")

    ax[-1].set_xlabel("Time")

    if ylabels is not None:
        for i, ylabel in enumerate(ylabels):
            ax[i].set_ylabel(ylabel)

    fig.align_ylabels()
    plt.tight_layout()

    if savename is not None:
        plt.savefig(savename, dpi=150)

    plt.show()
    plt.close()


def set_number_of_particle():
    try:
        if sys.argv[1] == "test":
            num_particles = 200
    except IndexError:
        num_particles = 10000
    return num_particles


def main():

    dtype = tf.float32
    px.xspec.set_energy(0.5, 10.0, 10)

    num_particles = set_number_of_particle()

    observations = tf.convert_to_tensor(
        np.loadtxt("data/observations.txt"),
        dtype=dtype)

    positive_bijector = tfb.Blockwise(
        [tfb.Exp(), tfb.Exp()])

    # Assumes that
    # parameter1 of xspec follows trend (1; random walk),
    # parameter2 of xspec follows trend (1; random walk).
    coefficients = tf.constant(
        [[[1.0, 0.0],
          [0.0, 1.0]],
         [[0.0, 0.0],
          [0.0, 0.0]]],
        dtype=dtype)
    order = coefficients.shape[0]
    xspec_param_size = coefficients.shape[1]
    transition_noise_cov = tf.linalg.diag([0.3, 0.1])
    var_trans = px.TransitionVectorAutoregressive(
        coefficients, transition_noise_cov, dtype=dtype)
    transition_function = var_trans.transition_function

    observation_function = px.get_observaton_function_xspec_poisson(
        "diskbb", xspec_param_size, num_particles, [0, 1],
        bijector=positive_bijector)

    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=tf.tile(positive_bijector.inverse(
            tf.constant([0.2, 1e6], dtype=dtype)), [order]),
        scale_diag=tf.ones(order*xspec_param_size, dtype=dtype))

    t0 = time.time()
    particles, log_weights, _, _ = tfp.experimental.mcmc.particle_filter(
        observations,
        initial_state_prior,
        transition_function,
        observation_function,
        num_particles,
        parallel_iterations=1,
        seed=0
    )
    t1 = time.time()
    print("Inference ran in {:.2f}s.".format(t1-t0))

    particles = positive_bijector.forward(
        particles[..., :xspec_param_size])

    latents = np.loadtxt("data/latents.txt")

    particle_quantiles = [[0.160, 0.840], [0.025, 0.975], [0.001, 0.999]]

    plot_and_save_particle_distribution_with_latents(
        particles,
        latents,
        ylabels=["diskbb.Tin (keV)", "diskbb.norm"],
        savename=".cache/figs/curve_particle_filtered.png",
        particle_quantiles=particle_quantiles,
        indicies_logy=[0, 1])

    weited_particles = px.ParticleNumpy(particles, log_weights)
    smoothed_particles = weited_particles.smooth_lag_fixed(20)
    plot_and_save_particle_distribution_with_latents(
        smoothed_particles,
        latents,
        ylabels=["diskbb.Tin (keV)", "diskbb.norm"],
        savename=".cache/figs/curve_particle_smoothed.png",
        particle_quantiles=particle_quantiles,
        indicies_logy=[0, 1])


if __name__ == "__main__":
    main()
