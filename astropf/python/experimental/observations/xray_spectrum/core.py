"""Energy spectrum model base module.
"""
import tensorflow as tf


class XraySpectrum(tf.Module):
    def __init__(self, energy_edges, num_params, dtype=tf.float32):
        self._energy_edges = energy_edges
        self.num_params = num_params
        self.dtype = dtype

    def __call__(self, value):
        return self.forward(value)

    def _forward(self, x):
        """Subclass implementation for `forward` public function."""
        raise NotImplementedError("forward not implemented.")

    def forward(self, flux):
        return self._forward(flux)

    @property
    def energy_edges(self):
        return self._energy_edges

    @property
    def _energies(self):
        raise NotImplementedError("energies not implemented.")

    @property
    def energies(self):
        return self._energies
