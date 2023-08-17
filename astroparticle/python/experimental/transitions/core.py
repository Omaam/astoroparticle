"""Transition class.
"""
import tensorflow as tf
import tensorflow_probability as tfp

tfl = tf.linalg
tfd = tfp.distributions


class LatentModel(tf.Module):

    def __init__(self, num_dims):
        self.num_dims = num_dims

    def forward(self, step, particles):
        return self._forward(step, particles)

    def _forward(self, step, particles):
        raise NotImplementedError("forward not implemented.")

    def default_latent_indices(self, **kwargs):
        return self._default_latent_indices()

    def _default_latent_indices(self, **kwargs):
        raise NotImplementedError("default_latent_indices not implemented.")


class LinearLatentModel(LatentModel):

    def __init__(self,
                 transition_matrix,
                 num_dims,
                 name="LinearLatentModel"):
        with tf.name_scope(name) as name:
            self._transition_matrix = transition_matrix
            self._num_dims = num_dims
            self._latent_size = transition_matrix.shape[-1]

            super(LinearLatentModel, self).__init__(
                num_dims
            )

    def _forward(self, step, particles):
        return particles @ tf.linalg.matrix_transpose(
            self.transition_matrix)

    @property
    def latent_size(self):
        return self._latent_size

    @property
    def num_dims(self):
        return self._num_dims

    @property
    def transition_matrix(self):
        return self._transition_matrix


class NonLinearLatentModel(LatentModel):

    def __init__(self,
                 transition_func,
                 num_dims,
                 name="NonLinearLatentModel"):
        with tf.name_scope(name) as name:
            self._transitoin_func = transition_func
            super(LinearLatentModel, self).__init__(
                num_dims
            )

    def _forward(self, step, particles):
        return self.transition_function(step, particles)

    def _transitoin_function(self, **inputs):
        raise NotImplementedError(
            "transition function not implemented.")

    def transition_function(self, **inputs):
        return self._transitoin_function


class ConditionalDistribution(tf.Module):
    """Conditional distribution.

    x_n ~ R(*|x_{n-1})
    """

    def __init__(self, distribution):
        self.distribution = distribution

    def __call__(self, step, particles):
        return self.forward(step, particles)

    def forward(self, step, particles):
        """Forward particles.

        Args:
            step: the step number.
            particles: particles.

        Return:
            particles_updated: Updated particles.
        """
        return self._forward(step, particles)

    def _forward(self, step, particles):
        raise NotImplementedError("forward not implemented")


class TransitionModel(ConditionalDistribution):
    """Transition model.

    x_n = F(x_{n-1}, v_n).
    But, both linear and non-linear could be handled.
    """

    def __init__(self,
                 transition_dist: ConditionalDistribution,
                 state_model: LatentModel,
                 noise_model: LatentModel):
        super(TransitionModel, self).__init__(
            distribution=transition_dist
        )
        self.state_model = state_model
        self.noise_model = noise_model

    def _forward(self, step, particles):

        dtype = particles.dtype
        batch_shape = particles.shape[:-1]

        state_particles = tf.gather(
            particles,
            self.state_model.default_latent_indices(),
            axis=-1)
        noise_particles = tf.gather(
            particles,
            self.noise_model.default_latent_indices(
                ) + self.state_model.latent_size,
            axis=-1)
        state_particles_new = self.state_model.forward(step, state_particles)
        noise_particles_new = self.noise_model.forward(step, noise_particles)

        particles_latent_new = tf.concat(
            [state_particles_new, noise_particles_new],
            axis=-1)
        particles_latent_noise_new = tf.concat(
            [noise_particles_new,
             tf.zeros(
                (*batch_shape,
                 self.state_model.latent_size-self.state_model.num_dims),
                dtype=dtype),
             # add 0.01 % error for noise for resampling. If no error,
             # particles convege to one value.
             # tf.zeros((*batch_shape, self.noise_model.num_dims),
             #          dtype=dtype)
             1e-2 * tf.ones(
                (*batch_shape, self.noise_model.num_dims),
                dtype=dtype)
             ], axis=-1)

        return tfd.Independent(
            self.distribution(
                particles_latent_new,
                particles_latent_noise_new,
            ), reinterpreted_batch_ndims=1)
