"""Example of VAR(1) model.
"""
import time

import numpy as np
import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

import partical_xspec as px
from partical_xspec import transitions as pxt
from partical_xspec import observations as pxo
from partical_xspec.examples import tools as extools


extools.seaborn_settings(context="notebook")


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

    # Load observations and true latents.
    observed_values = tf.convert_to_tensor(
        np.loadtxt("data/observations.txt"), dtype=dtype)
    latents = np.loadtxt("data/latents.txt")

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

    particle_quantiles = [[0.160, 0.840], [0.025, 0.975], [0.001, 0.999]]
    xspec_param_names = ["powerlaw.PhoIndex", "powerlaw.norm"]

    savepath = extools.join_and_create_directory(
        ".cache", "figs", "curve_particle_filtered.png")
    extools.plot_and_save_particle_distribution(
        particles,
        latent_labels=xspec_param_names,
        latents_true=latents,
        particle_quantiles=particle_quantiles,
        savepath=savepath)

    savepath = extools.join_and_create_directory(
        ".cache", "figs", "curve_particle_smoothed.png")
    smoothed_particles = px.WeightedParticleNumpy(
        particles, log_weights).smooth_lag_fixed(20)
    extools.plot_and_save_particle_distribution(
        smoothed_particles,
        latent_labels=xspec_param_names,
        latents_true=latents,
        particle_quantiles=particle_quantiles,
        savepath=savepath)


if __name__ == "__main__":
    main()