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

import partical_xspec as px
import util

sns.set_style("whitegrid")


def plot_and_save_particle_distribution_with_latents(
        particles, latents_true=None, times=None,
        particle_quantiles=None):

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

    ax[-1].set_xlabel("Time")
    ax[0].set_ylabel("powerlaw.PhoIndex")
    ax[1].set_ylabel("powerlaw.norm")
    fig.align_ylabels()
    plt.tight_layout()

    savepath = util.join_and_create_directory(
        ".cache", "figs", "curve_particle_filtered.png")
    plt.savefig(savepath, dpi=150)
    plt.show()
    plt.close()


def main():

    px.xspec.set_energy(0.5, 10.0, 10)

    try:
        if sys.argv[1] == "test":
            num_particles = 100
    except IndexError:
        num_particles = 10000

    dtype = tf.float32

    blockwise_bijector = tfb.Blockwise(
        bijectors=[tfb.Chain([tfb.Scale(1.0), tfb.Exp()]),
                   tfb.Chain([tfb.Scale(10.), tfb.Exp()])]
    )

    transition_function = px.get_transition_var(
        coefficients=np.tile(np.diag([0.1, 0.1]), (1, 1, 1)),
        noise_covariance=np.diag(
            tf.convert_to_tensor([0.1, 0.1], dtype=dtype)),
        dtype=dtype)

    observation_function = px.get_observaton_function_xspec_poisson(
        "powerlaw", 2, num_particles, blockwise_bijector)

    initial_state_prior = tfd.MultivariateNormalDiag(
        loc=tf.constant([0.1, 0.1], dtype=dtype),
        scale_diag=tf.constant([0.01, 0.01], dtype=dtype))

    observations = tf.convert_to_tensor(
        np.loadtxt(".cache/observations.txt"),
        dtype=dtype)

    t0 = time.time()
    particles, _, _, log_lik = tfp.experimental.mcmc.particle_filter(
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

    particles_bijectored = blockwise_bijector.forward(particles)

    latents = np.loadtxt(".cache/latents.txt")
    particle_quantiles = [[0.160, 0.840], [0.025, 0.975], [0.001, 0.999]]
    plot_and_save_particle_distribution_with_latents(
        particles_bijectored, latents,
        particle_quantiles=particle_quantiles)


if __name__ == "__main__":
    main()
