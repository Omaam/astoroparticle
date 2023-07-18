"""Particle tests.
"""
import unittest

import numpy as np

from partical_xspec.particle import particle


class ParticleTest(unittest.TestCase):

    def test_smoothed_particle_shape(self):
        raw_particles = np.load(".cache/particles.npy")
        log_weights = np.load(".cache/log_weights.npy")

        particles = particle.ParticleNumpy(raw_particles, log_weights)
        smoothed_particles = particles.smooth_lag_fixed(20)
        self.assertEqual(raw_particles.shape, smoothed_particles.shape)


if __name__ == "__main__":
    unittest.main()
