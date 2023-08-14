"""Example of VAR(1) model.
"""
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

import astroparticle as ap
from astroparticle.examples import tools as extools

apt = ap.transitions
aps = ap.spectrum

extools.seaborn_settings(context="notebook")


def set_particle_numbers():
    import sys
    try:
        if sys.argv[1] == "test":
            num_particles = 200
    except IndexError:
        num_particles = 10000
    return num_particles


class MyPhysicalModel(aps.PhysicalComponent):

    def __init__(self, energy_edges):
        self.powerlaw = aps.PowerLaw(energy_edges)
        self.diskbb = aps.DiskBB(energy_edges)
        self.phabs = aps.PhabsNicerXti(energy_edges, nh=[0.5])

    def __call__(self, flux):
        flux = self.powerlaw(flux)
        flux = self.diskbb(flux)
        flux = self.phabs(flux)
        return flux

    def _set_parameter(self, x):
        self.powerlaw.set_parameter(x[:, :2])
        self.diskbb.set_parameter(x[:, 2:4])


def main():

    dtype = tf.float32

    # Load observations and true latents.
    latent_size = 4
    observed_values = tf.convert_to_tensor(
        np.loadtxt("data/observations.txt"), dtype=dtype)

    transition = apt.VectorAutoregressive(
        coefficients=tf.eye(latent_size, batch_shape=(1,)),
        noise_covariance=0.3 * tf.eye(latent_size, batch_shape=(1,)),
        dtype=dtype)

    xray_spectrum_bijector = tfb.Blockwise([
        tfb.Chain([tfb.Scale(1.), tfb.Exp()]),
        tfb.Chain([tfb.Scale(10.), tfb.Exp()]),
        tfb.Chain([tfb.Scale(0.2), tfb.Exp()]),
        tfb.Chain([tfb.Scale(1e6), tfb.Exp()]),
    ])

    # Observation part.
    num_energy_obs = 10
    energy_range_obs = [0.5, 10.]
    energy_edges_obs = tf.linspace(energy_range_obs[0], energy_range_obs[1],
                                   num_energy_obs+1)

    response = aps.ResponseNicerXti()
    rebin = aps.Rebin(energy_edges_input=response.energy_edges_output,
                      energy_edges_output=energy_edges_obs)
    physical_model = MyPhysicalModel(response.energy_edges_input)

    @tf.function(jit_compile=False, autograph=False)
    def observation_fn(step, xray_spectrum_params):
        xray_spectrum_params = xray_spectrum_bijector.forward(
            xray_spectrum_params)

        physical_model.set_parameter(xray_spectrum_params)

        flux = tf.zeros(response.energy_size_input, dtype=dtype)
        flux = physical_model(flux)
        flux = response(flux)
        flux = rebin(flux)

        # Assume that the Signal-to-Noise Ratio for each band
        # is 10%, and this information is known.
        observation_dist = tfd.Independent(
            tfd.Normal(loc=flux, scale=0.1*flux),
            reinterpreted_batch_ndims=1)

        return observation_dist

    num_particles = set_particle_numbers()

    t0 = time.time()
    [
     particle,
     log_weights,
     parent_indices,
     incremental_log_marginal_likelihood,
    ] = tfp.experimental.mcmc.particle_filter(
        observed_values,
        initial_state_prior=tfd.MultivariateNormalDiag(
            scale_diag=tf.repeat(0.5, latent_size)),
        transition_fn=transition.get_function(),
        observation_fn=observation_fn,
        num_particles=num_particles,
        parallel_iterations=1,
        seed=123)
    t1 = time.time()
    print("Inference ran in {:.2f}s.".format(t1-t0))

    particle_bijectored = xray_spectrum_bijector.forward(particle)
    latent_values_true = tf.convert_to_tensor(
        np.loadtxt("data/latents.txt"), dtype=dtype)

    extools.plot_and_save_particle_latent(
        particle_bijectored,
        latents_true=latent_values_true,
        latent_labels=["powerlaw\nphoton_index", "powerlaw\nnorm",
                       "diskbb\ntin", "diskbb\nnorm",
                       "phabs\nnh"],
        quantiles=[[0.025, 0.975], [0.001, 0.999]],
        logy_indices=[1, 3, 4, 5, 6],
        savepath=".cache/figs/latent_values_particle.png",
        show=False)

    extools.plot_and_save_particle_observation(
        particle,
        observation_fn,
        observation_true=observed_values,
        quantiles=[[0.025, 0.975], [0.001, 0.999]],
        logy_indices=np.arange(observed_values.shape[1]),
        savepath=".cache/figs/observation_values_particle.png",
        show=False)


if __name__ == "__main__":
    main()