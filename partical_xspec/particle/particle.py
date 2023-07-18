"""Particle class.
"""
import numpy as np
from scipy import special


class ParticleNumpy:
    def __init__(self, particles, log_weights):
        self._particles = np.asarray(particles)
        self._log_weights = np.asarray(log_weights)

        self.num_timesteps = particles.shape[0]
        self.particle_size = particles.shape[1]

    def smooth_lag_fixed(self, num_fixed_lag=20):
        particles = self._particles
        log_weights = self._log_weights
        particle_size = self.particle_size

        smoothed_particle = np.copy(particles)
        for i in range(self.num_timesteps):
            probs = special.softmax(log_weights[i])
            selected_particle_ids = np.random.choice(
                particle_size, particle_size,
                replace=True, p=probs)

            idx_start = 0 if i < num_fixed_lag else i - num_fixed_lag
            smoothed_particle[idx_start:i+1] = \
                smoothed_particle[idx_start:i+1, selected_particle_ids]
        return smoothed_particle

    @property
    def values(self):
        return self._particles

    @property
    def shape(self):
        return self._particles.shape


class ParticleTF:
    pass
