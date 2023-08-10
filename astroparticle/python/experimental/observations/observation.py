"""Observation base class."""
import tensorflow as tf
import tensorflow_probability as tfp

from astroparticle.python.experimental.spectrum.binning import Rebin

tfb = tfp.bijectors
tfd = tfp.distributions


class ObservationModel:
    """Base class for observation models.
    """

    def __init__(self,
                 physical_model,
                 detector_response,
                 energy_ranges_kev,
                 noise_distribution,
                 observation_size,
                 spectrum_param_bijectors=None,
                 dtype=tf.float32):
        """Construct a specification for an observation model.

        Args:
            xspec_model_name: The name of xspec name.
            noise_distrubuion: An instance of `tf.distributions` with
                shape of the size of xspec parameters.
            xspec_bijector: A bijector for parameters casting to xspec.
                This enables to constrain xspec parameter space.
            dtype: dtype used computation.
        """

        self.physical_model = physical_model
        self.detector_response = detector_response
        self.energy_ranges_kev = energy_ranges_kev
        self.noise_distribution = noise_distribution
        self.spectrum_param_bijector = tfb.Blockwise(
            spectrum_param_bijectors)
        self.dtype = dtype

        self.energy_edges_model = detector_response.energy_edges_model
        self.energy_edges_detector = detector_response.energy_edges_detector
        self.energy_edges_obs = tf.linspace(
            energy_ranges_kev[0], energy_ranges_kev[1],
            observation_size+1)

    def _get_model_empty_flux(self):
        return tf.zeros(self. self.energy_edges_model-1, dtype=self.dtype,
                        name="flux")

    def get_function(self, latent_indicies=None):
        """Get observation function."""
        observation_function = self._observation_function(
            latent_indicies)
        self.observation_function = observation_function
        return observation_function

    def _observation_function(self, latent_indicies=None):
        """Return observation function.
        """

        detector_response = self.detector_response
        rebin = Rebin(
            energy_edges_old=self.energy_edges_detector,
            energy_edges_new=self.energy_edges_obs)
        physical_model = self.physical_model

        @tf.function(jit_compile=True, autograph=False)
        def _observation_fn(step, x):

            if latent_indicies is not None:
                x = tf.gather(x, latent_indicies, axis=-1)

            if self.spectrum_param_bijector is not None:
                x = self.spectrum_param_bijector.forward(x)

            physical_model.set_model_param(x)

            flux = tf.zeros(detector_response.num_energies_input)
            flux = physical_model(flux)
            flux = detector_response(flux)
            flux = rebin(flux)

            observation_dist = tfd.Independent(
                tfd.Normal(loc=flux, scale=10.),
                reinterpreted_batch_ndims=1)

            return observation_dist

        return _observation_fn

    def compute_observation_from_particle(self, particles):
        """Compute approximate observation."""
        observation_particles = []
        for p in range(particles.shape[-2]):
            particle_flux = self._compute_flux(
                particles[:, p, :])
            observation_particles.append(particle_flux)

        observation_particles = tf.convert_to_tensor(
            observation_particles, dtype=self.dtype)
        return observation_particles
