"""Create light curve by defining Xspec paramter's variation.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from statsmodels.tsa.vector_ar.var_model import VARProcess

import astroparticle as ap
from astroparticle.examples.tools import util

ape = ap.experimental


sys.path.append("..")


sns.set_style("whitegrid")


def generate_sample_varmodel():

    num_timesteps = 100
    ndim = 2
    order = 1
    noise_scale = 0.3
    random_seed = 123

    coefs = np.tile(np.diag(np.repeat(0.1, ndim)), (order, 1, 1))
    noise_cov = np.diag(np.repeat(noise_scale**2, ndim))
    varmodel = VARProcess(coefs, coefs_exog=None, sigma_u=noise_cov)
    data = varmodel.simulate_var(num_timesteps, seed=random_seed)
    return data


def plot_and_save_curve_parameter_observation(
        times, parameters, time_spectra, savename=None, show=False):
    fig, ax = plt.subplots(3, sharex="col", figsize=(7, 6))
    ax[0].plot(times, parameters[:, 0], color="k")
    ax[1].plot(times, parameters[:, 1], color="k")
    colors = sns.color_palette("Spectral", time_spectra.shape[-1])
    for i, curve in enumerate(time_spectra.T):
        ax[2].plot(times, curve, color=colors[i])
    ax[0].set_ylabel("powerlaw.PhoIndex")
    ax[1].set_ylabel("powerlaw.norm")
    ax[2].set_ylabel("Flux")
    ax[2].set_yscale("log")
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
    ax.grid(which="minor", lw=0.5)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Flux")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=150)
    if show:
        plt.show()
    plt.close()


def main():

    # Parameter settings.
    energy_kev_start = 0.5
    energy_kev_end = 10.0
    num_bands = 10
    num_timesteps = 100

    params = generate_sample_varmodel()
    params = np.exp(params)
    params = params * np.array([1.0, 10.0])

    times = np.arange(num_timesteps)

    energy_splits_model = tf.linspace(0.1, 20.0, 3451+1)
    energy_splits_nicer = tf.linspace(0.1, 20.0, 1501+1)
    energy_splits_output = tf.linspace(energy_kev_start, energy_kev_end,
                                       num_bands+1)

    energy_edges_model = tf.stack([energy_splits_model[:-1],
                                   energy_splits_model[1:]],
                                  axis=1)

    flux = tf.zeros([num_timesteps, 3451], dtype=tf.float32)
    flux = ape.observations.PowerLaw(
            energy_edges_model, params[:, 0], params[:, 1])(flux)
    flux = ape.observations.ResponseNicerXti()(flux)
    flux = ape.observations.Rebin(
        energy_splits_old=energy_splits_nicer,
        energy_splits_new=energy_splits_output)(flux)
    flux = flux + tf.random.normal([num_timesteps, num_bands], 0.0, 10.)
    time_spectra = flux

    savepath_latent = util.join_and_create_directory(
        "latents.txt")
    np.savetxt(savepath_latent, params)
    savepath_observation = util.join_and_create_directory(
        "observations.txt")
    np.savetxt(savepath_observation, time_spectra)

    save_curve_path = util.join_and_create_directory(
        "..", ".cache", "figs", "curve_parameter_observation.png")
    plot_and_save_curve_parameter_observation(
        times, params, time_spectra, savename=save_curve_path,
        show=True)

    energies = (energy_splits_output[1:] + energy_splits_output[:-1]) / 2
    save_spectra_path = util.join_and_create_directory(
        "..", ".cache", "figs", "observed_energy_spectra.png")
    plot_and_save_energyspectra(
        energies, time_spectra, savename=save_spectra_path)


if __name__ == "__main__":
    main()
