"""Example plotting functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf


def seaborn_settings(style="whitegrid", context="talk"):
    sns.set_style(style)
    sns.set_context(context)


def plot_quantiles(times, particles, quantiles, ax):
    quantiles = np.array(quantiles)
    if len(quantiles.shape) == 1:
        quantiles = quantiles[np.newaxis, :]
    for quantile in quantiles:
        errors = np.quantile(particles, quantile, axis=-1)
        ax.fill_between(times, *errors, alpha=0.20,
                        facecolor="none", color="r",
                        edgecolor="none")
    return ax


def plot_and_save_particle_latent(particles,
                                  times=None,
                                  latent_labels=None,
                                  latents_true=None,
                                  quantiles=None,
                                  figsize=None,
                                  logy_indices=None,
                                  savepath=None,
                                  show=False):
    """Plot and save particle distributions.
    """

    if latents_true.shape[-2] != particles.shape[-3]:
        raise ValueError(
            "`latents_true.shape[-2]` must be `particles.shape[-3]`. "
            f"{latents_true.shape[-2]} != {particles.shape[-3]}.")

    latent_size = particles.shape[-1]

    if times is None:
        times = np.arange(particles.shape[-3])

    if latent_labels is None:
        latent_labels = [f"parameter {i}" for i in range(latent_size)]

    if figsize is None:
        figsize = (7, 1.5*latent_size)

    fig, ax = plt.subplots(latent_size, sharex=True, figsize=figsize)

    if latents_true is not None:
        for i in range(latent_size):
            ax[i].plot(times, latents_true[:, i], color="k")

    particle_dist_centers = np.quantile(particles, 0.5, axis=-2)
    for i in range(latent_size):
        ax[i].plot(times, particle_dist_centers[:, i], color="r")
        ax[i].set_ylabel(latent_labels[i])

        if logy_indices is not None:
            if i in logy_indices:
                ax[i].set_yscale("log")

        if quantiles is not None:
            ax[i] = plot_quantiles(times, particles[..., i],
                                   quantiles, ax[i])
    ax[-1].set_xlabel("Time")
    fig.align_ylabels()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_and_save_particle_observation(particles,
                                       observation_function,
                                       times=None,
                                       observation_labels=None,
                                       observation_true=None,
                                       quantiles=None,
                                       logy=False,
                                       logy_indices=None,
                                       figsize=None,
                                       savepath=None,
                                       show=False):
    observation_particles = tf.map_fn(
        lambda p: observation_function(None, p).mean(),
        particles)

    observation_size = observation_particles.shape[-1]
    num_timesteps = observation_particles.shape[-3]

    if figsize is None:
        figsize = (7, observation_size)

    fig, ax = plt.subplots(observation_size, sharex=True,
                           figsize=figsize,
                           constrained_layout=True)
    if times is None:
        times = tf.range(num_timesteps)

    if observation_labels is None:
        observation_labels = ["obs {}".format(i)
                              for i in range(observation_size)]

    observation_centers = np.quantile(observation_particles, 0.5, axis=-2)
    for i in range(observation_size):
        ax[i].plot(times, observation_true[:, i], color="k")
        ax[i].plot(times, observation_centers[:, i], color="r")
        ax[i].set_ylabel(observation_labels[i])
        if quantiles is not None:
            ax[i] = plot_quantiles(times, observation_particles[..., i],
                                   quantiles, ax[i])
        if logy:
            import warnings
            warnings.warn("you should not use `logy`, "
                          "instead use `logy_indices`.")
            ax[i].set_yscale("log")

        if logy_indices is not None:
            if i in logy_indices:
                ax[i].set_yscale("log")

    ax[-1].set_xlabel("Time")
    fig.align_ylabels()
    if savepath is not None:
        plt.savefig(savepath, dpi=150)
    if show:
        plt.show()
    plt.close()
