"""Sums of time-series models."""

import tensorflow as tf

from astroparticle.python.transitions.transition import Transition


class Sum(Transition):
    def __init__(self, coefficients, noise_covariance, dtype=tf.float32):
        raise NotImplementedError()

    def _default_latent_indicies(self):
        return tf.range(self.latent_size)
