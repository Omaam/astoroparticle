"""Particle filter"""
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_config


np_config.enable_numpy_behavior()


class ParticleFilter:
    def __init__(self, transition, observation):

        self.transition = transition
        self.observation = observation

    def sample(self, observations, initial_state_prior,
               num_particles, seed=None):

        transition = self.transition
        observation = self.observation

        transition_fn = transition.get_function()

        observation_fn = observation.get_function(
            transition.default_latent_indicies)

        [
         particles,
         log_weights,
         _,
         _
        ] = tfp.experimental.mcmc.particle_filter(
                observations,
                initial_state_prior,
                transition_fn,
                observation_fn,
                num_particles,
                parallel_iterations=1,
                seed=seed
            )

        particles = self.observation.default_xspec_bijector.forward(
            particles[..., transition.default_latent_indicies])

        return particles, log_weights
