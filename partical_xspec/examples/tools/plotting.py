"""Example plotting functions.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def seaborn_settings(style="whitegrid", context="talk"):
    sns.set_style(style)
    sns.set_context(context)


def plot_and_save_particle_distribution(
        particles,
        times=None,
        latent_labels=None,
        latents_true=None,
        particle_quantiles=None,
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

    fig, ax = plt.subplots(latent_size, sharex=True)

    if latents_true is not None:
        ax[0].plot(times, latents_true[:, 0], color="k")
        ax[1].plot(times, latents_true[:, 1], color="k")

    particle_dist_centers = np.quantile(particles, 0.5, axis=-2)
    for i in range(latent_size):
        ax[i].plot(times, particle_dist_centers[:, i], color="r")
        ax[i].set_ylabel(latent_labels[i])

    if particle_quantiles is not None:
        for quantile in particle_quantiles:
            errors_sigma = np.quantile(particles, quantile, axis=-2)
            for i in range(latent_size):
                ax[i].fill_between(
                    times, *errors_sigma[..., i], alpha=0.20,
                    facecolor="none", color="r", edgecolor="none")

    ax[-1].set_xlabel("Time")
    fig.align_ylabels()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=150)

    if show:
        plt.show()

    plt.close()
