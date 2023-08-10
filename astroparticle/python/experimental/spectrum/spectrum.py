"""Energy spectrum model base module.
"""
import tensorflow as tf


class Spectrum(tf.Module):
    def __init__(self,
                 energy_edges_input,
                 energy_edges_output):
        self._energy_edges_input = energy_edges_input
        self._energy_edges_output = energy_edges_output

    def __call__(self, value):
        return self.forward(value)

    def _forward(self, x):
        """Subclass implementation for `forward` public function."""
        raise NotImplementedError("forward not implemented.")

    def forward(self, flux):
        return self._forward(flux)

    @property
    def energy_edges_input(self):
        return self._energy_edges_input

    @property
    def energy_edges_output(self):
        return self._energy_edges_output

    @property
    def energy_size_input(self):
        return self._energy_edges_input.shape[0] - 1

    @property
    def energy_size_output(self):
        return self._energy_edges_output.shape[0] - 1
