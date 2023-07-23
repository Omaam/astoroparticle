"""Observation base class."""
import xspec
import tensorflow as tf
from tensorflow_probability import distributions as tfd


class Observation:
    def __init__(self, xspec_model_name, noise_distribution,
                 default_xspec_bijector=None, dtype=tf.float32):

        self.xspec_model = xspec.Model(xspec_model_name)
        self.noise_distribution = noise_distribution
        self.default_xspec_bijector = default_xspec_bijector
        self.dtype = dtype

    def get_function(self, latent_indicies=None):
        return self._observation_function(latent_indicies)

    def _observation_function(self, latent_indicies=None):

        def _observation_fn(step, x):

            if latent_indicies is not None:
                x = tf.gather(x, latent_indicies, axis=-1)

            if self.default_xspec_bijector is not None:
                x = self.default_xspec_bijector.forward(x)

            particle_flux = []
            for i in range(x.shape[-2]):
                self.xspec_model.setPars(*x[i].tolist())
                particle_flux.append(self.xspec_model.values(0))
            particle_flux = tf.convert_to_tensor(
                particle_flux, dtype=self.dtype)

            observation_dist = tfd.Independent(
                self.noise_distribution(particle_flux),
                reinterpreted_batch_ndims=1)

            return observation_dist

        return _observation_fn
