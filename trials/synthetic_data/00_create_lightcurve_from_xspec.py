"""Create light curve by defining Xspec paramter's variation.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xspec
from statsmodels.tsa.vector_ar.var_model import VARProcess

import util

sns.set_style("whitegrid")


def generate_sample_varmodel(num_timesteps, ndim=1,
                             order=1, noise_scale=0.1,
                             random_seed=None):
    coefs = np.tile(np.diag(np.repeat(0.1, ndim)),
                    (order, 1, 1))
    noise_cov = np.diag(np.repeat(noise_scale, ndim))
    varmodel = VARProcess(coefs, coefs_exog=None,
                          sigma_u=noise_cov)
    data = varmodel.simulate_var(num_timesteps,
                                 seed=random_seed)
    return data


def plot_and_save_curve_parameter_observation(
        times, parameters, time_spectra, savename=None, show=False):
    time_spectra_for_plot = time_spectra - np.arange(time_spectra.shape[-1])
    fig, ax = plt.subplots(3, sharex="col", figsize=(7, 6))
    ax[0].plot(times, parameters[:, 0], color="k")
    ax[1].plot(times, parameters[:, 1], color="k")
    colors = sns.color_palette("Spectral", time_spectra.shape[-1])
    for i, curve in enumerate(time_spectra_for_plot.T):
        ax[2].plot(times, curve, color=colors[i])
    ax[0].set_ylabel("powerlaw.PhoIndex")
    ax[1].set_ylabel("powerlaw.norm")
    ax[2].set_ylabel("Flux")
    ax[-1].set_xlabel("Time")
    fig.align_ylabels()
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_and_save_energyspectra(energies, time_spectra, savename=None,
                                show=False):
    fig, ax = plt.subplots(1, sharex="col", figsize=(7, 5))
    colors = sns.color_palette("Spectral", time_spectra.shape[-2])
    for i, spectrum in enumerate(time_spectra):
        ax.plot(energies, spectrum, color=colors[i], alpha=0.30)
    ax.plot(energies, np.mean(time_spectra, axis=0), color="k")
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=150)
    if show:
        plt.show()
    plt.close()


def main():

    # Parameter settings.
    enery_kev_start = 0.5
    enery_kev_end = 10.0
    num_bands = 10
    num_timesteps = 100

    # Model settings.
    xspec.AllModels.setEnergies(
        f"{enery_kev_start} {enery_kev_end} {num_bands}")

    xspec_model = xspec.Model("powerlaw")

    params = generate_sample_varmodel(
        num_timesteps, ndim=2, order=1,
        random_seed=1)
    params = np.exp(params)
    params = params * np.array([1.0, 10.0])

    times = np.arange(num_timesteps)
    time_spectra = np.empty((num_timesteps, num_bands))
    for i, param in enumerate(params):
        xspec_model.powerlaw.PhoIndex = param[0]
        xspec_model.powerlaw.norm = param[1]
        flux = np.array(xspec_model.values(0))
        rate = np.random.poisson(flux)
        time_spectra[i] = rate

    savepath_latent = util.join_and_create_directory(
        ".cache", "latents.txt")
    np.savetxt(savepath_latent, params)
    savepath_observation = util.join_and_create_directory(
        ".cache", "observations.txt")
    np.savetxt(savepath_observation, time_spectra)

    save_curve_path = util.join_and_create_directory(
        ".cache", "figs", "curve_parameter_observation.png")
    plot_and_save_curve_parameter_observation(
        times, params, time_spectra, savename=save_curve_path)

    energy_edges = np.array(xspec_model.energies(0))
    energies = (energy_edges[1:] + energy_edges[:-1]) / 2
    save_spectra_path = util.join_and_create_directory(
        ".cache", "figs", "observed_energy_spectra.png")
    plot_and_save_energyspectra(
        energies, time_spectra, savename=save_spectra_path)


if __name__ == "__main__":
    main()
