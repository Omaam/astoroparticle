"""Transition class.
"""
import tensorflow as tf
import tensorflow_probability as tfp

tfl = tf.linalg
tfb = tfp.bijectors
tfd = tfp.distributions


class ParticleDistribution(tf.Module):
    """Particle distribution.

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


class LatentModel(tf.Module):

    def __init__(self,
                 num_dims,
                 noise_scale):

        if noise_scale is not None:
            noise_scale = tf.convert_to_tensor(noise_scale)

        self._num_dims = num_dims
        self._noise_scale = noise_scale

    def forward(self, step, particles):
        return self._forward(step, particles)

    def _forward(self, step, particles):
        raise NotImplementedError("forward not implemented.")

    def default_latent_indices(self, **kwargs):
        return self._default_latent_indices()

    def _default_latent_indices(self, **kwargs):
        raise NotImplementedError("default_latent_indices not implemented.")

    @property
    def num_dims(self):
        return self._num_dims

    @property
    def noise_scale(self):
        return self._noise_scale


class LinearLatentModel(LatentModel):

    def __init__(self,
                 num_dims,
                 transition_matrix,
                 noise_scale=None,
                 name="LinearLatentModel"):

        with tf.name_scope(name) as name:

            self._transition_matrix_transposed = tf.linalg.matrix_transpose(
                transition_matrix)
            self._num_dims = num_dims
            self._latent_size = transition_matrix.shape[-1]

            super(LinearLatentModel, self).__init__(
                num_dims,
                noise_scale
            )

    def _forward(self, step, particles):
        return particles @ self.transition_matrix_transposed

    @property
    def latent_size(self):
        return self._latent_size

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def transition_matrix_transposed(self):
        return self._transition_matrix_transposed


class NonLinearLatentModel(LatentModel):

    def __init__(self,
                 num_dims,
                 transition_function,
                 noise_scale,
                 name="NonLinearLatentModel"):
        with tf.name_scope(name) as name:

            self._transitoin_function = transition_function

            super(NonLinearLatentModel, self).__init__(
                num_dims,
                noise_scale
            )

    def _forward(self, step, particles):
        return self.transition_functiontion(step, particles)

    def _transitoin_functiontion(self, **inputs):
        raise NotImplementedError(
            "transition functiontion not implemented.")

    def transition_functiontion(self, **inputs):
        return self._transitoin_functiontion


class SelfOrganizingLatentModel(ParticleDistribution):
    """Transition model.

    x_n = F(x_{n-1}, v_n).
    But, both linear and non-linear could be handled.
    """

    def __init__(self,
                 transition_dist: tfd.Distribution,
                 state_model: LatentModel,
                 noise_model: LatentModel):

        """
        Raises:
            ValueError: if `noise_mode.noise_scale` is `LatentModel`.
        """

        if isinstance(noise_model.noise_scale, LatentModel):
            raise ValueError(
                "`noise_mode.noise_scale` must be float or "
                "list of float.")

        self.state_model = state_model
        self.noise_model = noise_model

        super(SelfOrganizingLatentModel, self).__init__(
            distribution=transition_dist
        )

    def _forward(self, step, particles):

        dtype = particles.dtype
        batch_shape = particles.shape[:-1]

        state_particles = tf.gather(particles,
                                    self.state_model.default_latent_indices(),
                                    axis=-1)
        noise_particles = tf.gather(particles,
                                    self.noise_model.default_latent_indices()
                                    + self.state_model.latent_size,
                                    axis=-1)

        state_particles_new = self.state_model.forward(step, state_particles)
        noise_particles_new = self.noise_model.forward(step, noise_particles)

        event_space_bijector = tfb.Chain([tfb.Log()])

        loc_particles = tf.concat(
            [state_particles_new,
             noise_particles_new],
            axis=-1)
        scale_particles = tf.concat(
            [
             # scale for state distribution.
             tf.gather(
                event_space_bijector.inverse(noise_particles_new),
                tf.range(self.state_model.num_dims),
                axis=-1),
             tf.zeros(
                (*batch_shape,
                 self.state_model.latent_size-self.state_model.num_dims),
                dtype=dtype),

             # scale for noise distribution
             tf.broadcast_to(
                tf.cast(self.noise_model.noise_scale, dtype),
                (*batch_shape, self.noise_model.num_dims)),
             tf.zeros(
                (*batch_shape,
                 self.noise_model.latent_size-self.noise_model.num_dims),
                dtype=dtype),
             ],
            axis=-1)

        return tfd.Independent(
            self.distribution(loc_particles, scale_particles),
            reinterpreted_batch_ndims=1)
