"""Energy spectrum of powerlaw simulation.
"""
import tensorflow as tf
from tensorflow_probability import distributions as tfd

import astroparticle as ap
from astroparticle import experimental as ape


def main():

    # The number of group edges for nicerrmf is 3451.
    observation = ap.observations.Observation(
        "powerlaw", 3451, tfd.Poisson,
        energy_ranges_kev=[0.1, 20.])

    model_flux = observation.compute_observation(
        tf.convert_to_tensor([[[1.0, 1.0]]]))[0]

    response = ape.observations.ResponseNicerXti()

    # The number of channel edges for nicerarf is 1495.
    rebin = ape.observations.Rebin(
        energy_splits_old=tf.linspace(0.1, 20.0, 1495),
        energy_splits_new=tf.linspace(0.5, 10.0, 11))

    observed_flux = response(model_flux)
    rebinned_flux = rebin(observed_flux)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.step(rebin.energies, rebinned_flux[0],
            where="mid", label="powerlaw")
    ax.set_title("NICER observation simulation")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("energy (keV)")
    ax.set_ylabel("cts/s")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
