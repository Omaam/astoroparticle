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
                                  figsize=(8, 5),
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

    fig, ax = plt.subplots(latent_size, sharex=True, figsize=figsize)

    if latents_true is not None:
        ax[0].plot(times, latents_true[:, 0], color="k")
        ax[1].plot(times, latents_true[:, 1], color="k")

    particle_dist_centers = np.quantile(particles, 0.5, axis=-2)
    for i in range(latent_size):
        ax[i].plot(times, particle_dist_centers[:, i], color="r")
        ax[i].set_ylabel(latent_labels[i])
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
                                       figsize=None,
                                       savepath=None,
                                       show=False):
    observation_particles = tf.map_fn(
        lambda p: observation_function(None, p).mean(),
        particles)

    num_observation = observation_particles.shape[-1]
    num_timesteps = observation_particles.shape[-3]

    if figsize is None:
        figsize = (7, 1*num_observation)

    fig, ax = plt.subplots(num_observation, sharex=True,
                           figsize=figsize,
                           constrained_layout=True)
    if times is None:
        times = tf.range(num_timesteps)

    if observation_labels is None:
        observation_labels = ["obs {}".format(i)
                              for i in range(num_observation)]

    for j in range(num_observation):
        ax[j].plot(times, observation_true[:, j], color="k")
        ax[j].set_ylabel(observation_labels[j])
        if quantiles is not None:
            ax[j] = plot_quantiles(times, observation_particles[..., j],
                                   quantiles, ax[j])
        if logy:
            ax[j].set_yscale("log")

    ax[-1].set_xlabel("Time")
    if savepath is not None:
        plt.savefig(savepath, dpi=150)
    if show:
        plt.show()
    plt.close()
