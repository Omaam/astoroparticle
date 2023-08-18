"""Example of VAR(1) model.
"""
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import astroparticle as ap
from astroparticle.examples import tools as extools

tfb = tfp.bijectors
tfd = tfp.distributions

ape = ap.experimental
apt = ap.transitions
aps = ap.spectrum

extools.seaborn_settings(context="notebook")


def set_particle_numbers():
    import sys
    try:
        if sys.argv[1] == "test":
            num_particles = 1000
    except IndexError:
        num_particles = 10000
    return num_particles


class MyObservationModel(aps.PhysicalComponent):

    def __init__(self, energy_edges_output):

        self.response = aps.ResponseNicerXti()
        self.rebin = aps.Rebin(
            energy_edges_input=self.response.energy_edges_output,
            energy_edges_output=energy_edges_output)

        energy_edges_model = self.response.energy_edges_input
        self.powerlaw = aps.PowerLaw(energy_edges_model)
        self.diskbb = aps.DiskBB(energy_edges_model)
        self.phabs = aps.PhabsNicerXti(energy_edges_model, nh=[0.5])

    @tf.function(autograph=False, jit_compile=False)
    def __call__(self, step, particles):

        dtype = particles.dtype

        particles = self.particles_bijector.forward(particles)

        self.set_parameter(particles)

        flux = tf.zeros(self.response.energy_size_input, dtype=dtype)
        flux = self.powerlaw(flux)
        flux = self.diskbb(flux)
        flux = self.phabs(flux)
        flux = self.response(flux)
        flux = self.rebin(flux)

        # Assume that the Signal-to-Noise Ratio for each band
        # is 10%, and this information is known.
        observation_dist = tfd.Independent(
            tfd.Normal(loc=flux, scale=0.1*flux),
            reinterpreted_batch_ndims=1)

        return observation_dist

    def _set_parameter(self, x):
        self.powerlaw.set_parameter(x[:, :2])
        self.diskbb.set_parameter(x[:, 2:4])

    @property
    def particles_bijector(self):
        return tfb.Blockwise([
            tfb.Chain([tfb.Scale(1.), tfb.Exp()]),
            tfb.Chain([tfb.Scale(10.), tfb.Exp()]),
            tfb.Chain([tfb.Scale(0.2), tfb.Exp()]),
            tfb.Chain([tfb.Scale(1e6), tfb.Exp()]),
            tfb.Chain([tfb.Exp()]),
            tfb.Chain([tfb.Exp()]),
            tfb.Chain([tfb.Exp()]),
            tfb.Chain([tfb.Exp()]),
        ])


def main():

    dtype = tf.float32

    # Load observations and true latents.
    observed_values = tf.convert_to_tensor(
        np.loadtxt("data/observations.txt"), dtype=dtype)

    num_dims = 4
    state_var_order = 1
    noise_trend_order = 1

    # Transition part.
    coefficients = 0.1 * tf.eye(num_dims, batch_shape=(state_var_order,))
    state_model = ape.transitions.VectorAutoregressive(
        coefficients, dtype=dtype, name="state_model_var")
    noise_model = ape.transitions.Trend(
        noise_trend_order,
        num_dims,
        noise_scale=[0.01, 0.01, 0.01, 0.01],
        dtype=dtype,
        name="noise_model_trend")
    transition_model = ape.transitions.SelfOrganizingLatentModel(
        tfd.Normal, state_model, noise_model)

    # Observation part.
    num_energy_obs = 10
    energy_range_obs = [0.5, 10.]
    energy_edges_obs = tf.linspace(energy_range_obs[0], energy_range_obs[1],
                                   num_energy_obs+1)
    observation_model = MyObservationModel(energy_edges_obs)

    t0 = time.time()
    [
     particles,
     log_weights,
     parent_indices,
     incremental_log_marginal_likelihood,
    ] = tfp.experimental.mcmc.particle_filter(
        observed_values,
        initial_state_prior=tfd.Independent(
            tfd.Normal(loc=[0.0, 0.0, 0.0, 0.0,   # Parameter
                            -1.2, -1.2, -1.2, -1.2],  # Noise
                       scale=[0.5, 0.5, 0.5, 0.5,
                              0.6, 0.6, 0.6, 0.6]),
            reinterpreted_batch_ndims=1),
        transition_fn=transition_model,
        observation_fn=observation_model,
        num_particles=set_particle_numbers(),
        parallel_iterations=1,
        seed=123)
    t1 = time.time()
    print("Inference ran in {:.2f}s.".format(t1-t0))

    # index of 5 is the equivalent hydrogen column, which set
    # to be constant.
    latent_values_true = tf.convert_to_tensor(
        np.loadtxt("data/latents.txt"), dtype=dtype)[:, :4]
    latent_values_true = tf.concat(
        [latent_values_true,
         0.3 * tf.ones(latent_values_true.shape, dtype=dtype)],
        axis=-1)

    particles_bijectored = observation_model.particles_bijector.forward(
        particles)
    extools.plot_and_save_particle_latent(
        particles_bijectored,
        latents_true=latent_values_true,
        latent_labels=["powerlaw\nphoton_index",
                       "powerlaw\nnorm",
                       "diskbb\ntin",
                       "diskbb\nnorm",
                       "noise scale\nphoton_index",
                       "noise_scale\nnorm",
                       "noise_scale\ntin",
                       "noise_scale\nnorm"],
        quantiles=[[0.025, 0.975], [0.001, 0.999]],
        logy_indices=[1, 3],
        savepath=".cache/figs/latent_values_particle.png",
        show=True)

    extools.plot_and_save_particle_observation(
        particles,
        observation_model,
        observation_true=observed_values,
        quantiles=[[0.025, 0.975], [0.001, 0.999]],
        logy_indices=np.arange(observed_values.shape[1]),
        savepath=".cache/figs/observation_values_particle.png",
        show=False)


if __name__ == "__main__":
    main()
