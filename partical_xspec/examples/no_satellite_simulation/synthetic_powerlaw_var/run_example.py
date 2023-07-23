"""
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

import partical_xspec as px
from partical_xspec import transitions as pxt
from partical_xspec import observations as pxo

import util


sns.set_style("whitegrid")


def plot_and_save_particle_distribution_with_latents(
        particles, latents_true=None, times=None,
        particle_quantiles=None, savepath=None):

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

    if savepath is not None:
        plt.savefig(savepath, dpi=150)
    plt.show()
    plt.close()


def set_particle_numbers():
    import sys
    try:
        if sys.argv[1] == "test":
            num_particles = 100
    except IndexError:
        num_particles = 10000
    return num_particles


def main():

    px.xspec.set_energy(0.5, 10.0, 10)

    dtype = tf.float32

    transition = pxt.VectorAutoregressive(
        coefficients=[[[0.1, 0.0], [0.0, 0.1]]],
        noise_covariance=tf.constant([[0.3, 0.0], [0.0, 0.3]])**2,
        dtype=dtype)

    observation = pxo.Observation(
        xspec_model_name="powerlaw",
        noise_distribution=tfd.Poisson,
        default_xspec_bijector=tfb.Blockwise([
            tfb.Chain([tfb.Scale(1.0), tfb.Exp()]),
            tfb.Chain([tfb.Scale(10.0), tfb.Exp()]),
            ])
    )

    pf = px.ParticleFilter(transition, observation)

    observed_values = tf.convert_to_tensor(
        np.loadtxt("data/observations.txt"), dtype=dtype)
    num_particles = set_particle_numbers()

    t0 = time.time()
    particles, log_weights = pf.sample(
        observed_values,
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=[0.5, 0.5]),
        num_particles=num_particles,
        seed=123)
    t1 = time.time()
    print("Inference ran in {:.2f}s.".format(t1-t0))

    latents = np.loadtxt("data/latents.txt")
    particle_quantiles = [[0.160, 0.840], [0.025, 0.975], [0.001, 0.999]]

    savepath = util.join_and_create_directory(
        ".cache", "figs", "curve_particle_filtered.png")
    plot_and_save_particle_distribution_with_latents(
        particles, latents,
        particle_quantiles=particle_quantiles,
        savepath=savepath)

    savepath = util.join_and_create_directory(
        ".cache", "figs", "curve_particle_smoothed.png")
    particle_obj = px.WeightedParticleNumpy(particles, log_weights)
    smoothed_particles = particle_obj.smooth_lag_fixed(20)
    plot_and_save_particle_distribution_with_latents(
        smoothed_particles, latents,
        particle_quantiles=particle_quantiles,
        savepath=savepath)


if __name__ == "__main__":
    main()
