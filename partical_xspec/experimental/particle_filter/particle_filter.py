"""Particle filter"""
import tensorflow_probability as tfp


class ParticleFilter:
    def __init__(self, transition, observation):

        self.transiton = transition
        self.observation = observation

    def sample(self, observations, initial_state_prior, num_particles):

        transition_fn = self.transition.transition_function
        observation_fn = self.observation.obserationo_function

        # ids_latent = self.transition.default_latent_indicies

        [particles,
         log_weights,
         _,
         _] = tfp.experimental.mcmc.particle_filter(
                observations,
                initial_state_prior,
                transition_fn,
                observation_fn,
                num_particles,
                parallel_iterations=1,
                seed=0
            )

        return particles, log_weights
