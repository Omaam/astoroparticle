"""Observation base class."""
import xspec
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from astroparticle.python.xspec.settings import set_energy


class Observation:
    """Base class for observation models.
    """

    def __init__(self, xspec_model_name, observation_size,
                 noise_distribution, xspec_bijector=None,
                 energy_ranges_kev=None, dtype=tf.float32):
        """Construct a specification for an observation model.

        Args:
            xspec_model_name: The name of xspec name.
            noise_distrubuion: An instance of `tf.distributions` with
                shape of the size of xspec parameters.
            xspec_bijector: A bijector for parameters casting to xspec.
                This enables to constrain xspec parameter space.
            dtype: dtype used computation.
        """

        self.xspec_model = xspec.Model(xspec_model_name)
        self.observation_size = observation_size
        self.noise_distribution = noise_distribution
        self.xspec_bijector = xspec_bijector
        self.dtype = dtype

        self.observation_function = None

        self._set_energy_ranges(observation_size, energy_ranges_kev)

    def compute_observation(self, particles):
        """Compute approximate observation."""
        observation_particles = []
        for p in range(particles.shape[-2]):
            particle_flux = self._compute_xspec_flux(
                particles[:, p, :])
            observation_particles.append(particle_flux)

        observation_particles = tf.convert_to_tensor(
            observation_particles, dtype=self.dtype)
        return observation_particles

    def _compute_xspec_flux(self, x):
        """Compute xspec flux."""
        particle_flux = []
        for i in range(x.shape[-2]):
            self.xspec_model.setPars(*x[i].tolist())
            particle_flux.append(self.xspec_model.values(0))
        particle_flux = tf.convert_to_tensor(
            particle_flux, dtype=self.dtype)
        return particle_flux

    def get_function(self, latent_indicies=None):
        """Get observation function."""
        observation_function = self._observation_function(
            latent_indicies)
        self.observation_function = observation_function
        return observation_function

    def _observation_function(self, latent_indicies=None):
        """Return observation function."""
        def _observation_fn(step, x):

            if latent_indicies is not None:
                x = tf.gather(x, latent_indicies, axis=-1)

            if self.xspec_bijector is not None:
                x = self.xspec_bijector.forward(x)

            particle_flux = self._compute_xspec_flux(x)
            observation_dist = tfd.Independent(
                self.noise_distribution(particle_flux),
                reinterpreted_batch_ndims=1)

            return observation_dist

        return _observation_fn

    def _set_energy_ranges(self, observation_size, energy_ranges_kev):
        if energy_ranges_kev is None:
            energy_ranges_kev = ["", ""]
        if len(energy_ranges_kev) != 2:
            raise ValueError("`len(energy_ranges_kev)` must be 2.")
        set_energy(energy_ranges_kev[0], energy_ranges_kev[1],
                   self.observation_size)
